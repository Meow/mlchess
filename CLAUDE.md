# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"NightyBot" — a UCI chess engine whose move generation is two small PyTorch MLPs trained by imitation learning on the [lichess open database](https://database.lichess.org/), combined with a hand-written piece-square-table evaluation at the search leaves. Designed to be dropped into `lichess-bot/engines/nightybot` with `run.sh` as the engine binary.

## Commands

No linter or build step exists. Everything is run directly with `python3`.

```bash
pip3 install torch numpy safetensors wandb chess   # deps are NOT installed in this checkout
python3 run.py            # UCI engine, reads commands on stdin (./run.sh is the lichess-bot entry point)
python3 test_engine.py    # sanity checks for the evaluation and the search
python3 train5.py         # train the "to" model  -> chess5.model
python3 train_from.py     # train the "from" model -> chess_from.model
python3 save.py           # export both .model files to .safetensors (for the Rust port)
python3 run_from.py       # one-off: dump the from-model's top squares for a hardcoded FEN
nix-shell shell.nix       # CUDA dev shell (NixOS, python310 + torchWithCuda)
```

Manual engine smoke test — `run.py` reads stdin in a loop, so you can pipe it:

```
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 winc 1000 binc 1000
```

`test_engine.py` loads the real checkpoints and plays out short searches, so it takes a few seconds to start and around a minute to run. It is the fastest way to tell whether a change to `evaluation.py` or the search broke something: it covers material ordering, the colour-mirror antisymmetry of `eval_pos`, promotion, mate scoring, stalemate and repetition handling, the time budget, and the UCI loop.

Both training scripts train in an infinite loop and autosave every 10000 steps; stop them with Ctrl-C. `train5.py` calls `wandb.init` unconditionally — comment it out (as `train_from.py` already does) or it will crash without a wandb login. Both also hardcode the PGN path (`/home/luna/Downloads/lichess_db_standard_rated_2023-10.pgn`), which must be edited before training.

## Architecture

Two models, both 1 input layer + 3 residual middle layers (620 features, GELU + LayerNorm) + 1 output layer:

- **`ChessFromModel`** (`chess_from.model`, input 1024 = 64 squares x 16-dim embedding) — outputs 128 logits, reshaped to `(2, 64)`: which piece to move, separately for white (row 0) and black (row 1).
- **`ChessModel`** (`chess5.model`, input 1088 = 1024 + a 64-dim embedding of the source square) — outputs 64 logits: where that piece should go.

`run.py` is the whole engine: UCI loop, search, and both model classes.

- `candidates()` turns the two nets into a shortlist of legal moves for a position: the top `search_pieces` source squares the from-model likes, crossed with the top `search_moves` destinations the to-model likes for each. `beam_size()` widens that shortlist in the opening and again in the endgame, where there are fewer legal moves to cover.
- `search()` is a plain negamax over that shortlist, scoring leaves with `eval_pos` from [evaluation.py](evaluation.py) (piece-square tables adapted from [blunder](https://github.com/algerbrex/blunder), MIT). Every node returns its score from the point of view of the side to move, and the caller negates.
- `find_best_move()` runs iterative deepening up to `search_depth`, fanning the root moves out over `threading.Thread`s. Each completed depth replaces the previous answer, so running out of time costs accuracy instead of the move. An iteration that does not finish is discarded whole.
- Time control comes from `go`: `time_budget()` reads `wtime`/`btime`/`winc`/`binc`/`movestogo`/`movetime` and returns a per-move budget. `search_depth`, `search_moves` and `search_pieces` at the top of the file are the ceilings, not the target — with a clock the engine usually stops short of them.

### Gotchas

- **`eval_pos` needs two separate index translations, and both look redundant.** The tables are in blunder's order (pawn, knight, bishop, rook, queen, king) while `encode()`'s alphabet is `".prnbqk"`, so `piece_to_table` remaps them — without it a rook scores off the knight table. And the tables are written from White's side with a8 first, which is already the orientation `encode()` produces, so White reads them straight and only Black is mirrored (`pov_list`) — the opposite of what blunder does, because blunder's own squares are a1-first. Both of these were wrong before, and "simplifying" either one back re-breaks the engine badly rather than subtly. `test_engine.py` checks material ordering and that `eval_pos` is antisymmetric under `board.mirror()`.
- **The model classes in `run_from.py` and `save.py` reference `nn.` but never import it.** This is not a bug to fix blindly: the models are saved with `torch.save(model)` (full pickled module), so `torch.load` resolves the class from `__main__` of whatever script is loading. `__init__` never runs on load, so the missing import is harmless — but it means **the `forward` that actually executes is the one in the loading script, not the one it was trained with**. `run.py`'s `ChessModel.forward` deliberately differs from `train5.py`'s: it adds `board_in.repeat(len(piece_in), 1)` to score one board against many candidate source squares in a batch. Keep class and attribute names identical across these files or loading breaks. `run.py` does import `nn` and additionally publishes both classes into `__main__` from `load_model()`, which is what lets `test_engine.py` import it.
- **`torch.load` needs `map_location` and `weights_only=False`.** The checkpoints are whole pickled modules saved from a CUDA box: without `map_location` they refuse to load on a CPU-only machine, and on torch 2.6+ `weights_only` defaults to `True` and rejects them outright. `load_model()` in `run.py` handles both; `run_from.py` and `save.py` still do not.
- **Two different square numberings are in play.** `encode()` returns the 64 squares in FEN reading order (index 0 = a8), while model *outputs* and labels use python-chess numbering (index 0 = a1, see `num_to_sqr`). Consistent between training and inference, so don't "fix" one side alone. `encode_board()` is the same encoding taken straight off a `chess.Board` to skip the FEN round-trip on the hot path — `sq ^ 56` is the conversion between the two numberings, and `test_engine.py` checks the two encoders agree.
- **`eval_pos`'s colour indices come from `encode()`, where 1 is white and 0 is black** — the opposite of `chess.WHITE`. `leaf_eval()` does that mapping in one place; there is no longer a `flip_idx` shuffle at the leaves.
- **A `chess.Move` with no promotion field is not a legal promotion.** Any code that builds moves as `chess.Move(from, to)` silently cannot queen a pawn, which is fatal in endgames. `candidates()` picks the matching legal move out of the move list instead of constructing one.
- Training samples only positions where the *winner* was to move, from games averaging 1900+ Elo (1600+ in the older `experiments/` scripts), and discards games under 6 plies. Nothing in the nets or the evaluation was trained on opening theory or endgame technique, so the search and `eval_pos` are carrying both.

## Layout

`experiments/` holds superseded training scripts and checkpoints (`train.py`–`train4.py`, smaller/narrower nets); 620x3 beat the 512x2 variant. Nothing in the live pipeline imports from it. `chess.safetensors` / `chess_from.safetensors` are `save.py` exports for consumption by a separate Rust implementation, and are not read by any Python here — note that the Rust port reimplements the evaluation, so the table-indexing fixes described above have to be carried across by hand.
