# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"NightyBot" — a UCI chess engine whose move generation is two small PyTorch MLPs trained by imitation learning on the [lichess open database](https://database.lichess.org/), combined with a hand-written piece-square-table evaluation at the search leaves. Designed to be dropped into `lichess-bot/engines/nightybot` with `run.sh` as the engine binary.

## Commands

No linter or build step exists. Everything is run directly with `python3`.

```bash
pip3 install torch numpy safetensors wandb chess   # deps are NOT installed in this checkout
python3 run.py            # UCI engine, reads commands on stdin (./run.sh is the lichess-bot entry point)
cargo build --release --manifest-path engine/Cargo.toml   # the same engine as a Rust binary; run.sh prefers it once built
engine/target/release/nightybot bench 4   # nodes/s at fixed depth; `logits FEN` / `candidates FEN` dump the nets
python3 test_engine_rs.py # checks the Rust engine against run.py (logits, shortlists, moves) and over UCI
python3 test_engine.py    # sanity checks for the evaluation and the search
python3 train5.py         # train the "to" model  -> chess5.model
python3 train_from.py     # train the "from" model -> chess_from.model
python3 save.py           # export both .model files to .safetensors (for the Rust port)
python3 run_from.py       # one-off: dump the from-model's top squares for a hardcoded FEN
nix-shell shell.nix       # CUDA dev shell (NixOS, python310 + torchWithCuda)

# reinforcement-learning engine (separate from the above; see "RL engine" below)
python3 train_rl.py       # self-play training -> nighty_rl.pt; Ctrl-C saves, reruns resume
python3 train_rl.py --out-dir /tmp/x --actors 2 --games-per-actor 8 --sims 16 --fast-sims 4 --min-buffer 100 --batch-size 32 --max-steps 5 --eval-every 0   # smoke run
python3 run_rl.py         # UCI engine for nighty_rl.pt (./run_rl.sh for lichess-bot)
python3 rl_eval.py --opponent greedy:2 --games 40   # arena; also nightybot, uci:<cmd>, rl:<file>
python3 test_rl.py        # ~1 min, needs no trained model; covers both search backends
python3 -m pip install ./rust   # build + install the Rust search for THIS python (or: cd rust && maturin develop --release)
```

Manual engine smoke test — `run.py` reads stdin in a loop, so you can pipe it:

```
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 winc 1000 binc 1000
```

`test_engine.py` loads the real checkpoints and plays out short searches, so it takes a few seconds to start and around a minute to run.
`test_engine_rs.py` does the same for the Rust binary and additionally diffs it against run.py position by position, so it needs `chess.safetensors` / `chess_from.safetensors` to be exports of the current `.model` files (rerun `save.py` after retraining). It is the fastest way to tell whether a change to `evaluation.py` or the search broke something: it covers material ordering, the colour-mirror antisymmetry of `eval_pos`, promotion, mate scoring, stalemate and repetition handling, the time budget, and the UCI loop.

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

### Rust build of the engine (`engine/`)

[engine/](engine/) is run.py as one binary, `nightybot`: the same nets run on the CPU from the safetensors exports ([engine/src/net.rs](engine/src/net.rs)), the same shortlist, search and time control ([engine/src/search.rs](engine/src/search.rs)), the same UCI loop ([engine/src/main.rs](engine/src/main.rs)). On the M5 Pro it searches ~60k nodes/s against ~7k for run.py (whose root threads are serialised by the GIL), so it completes depth 5 in half a second and depth 6 on a few-minute clock; its default `Depth` ceiling is 7 where run.py's is 5.

- **It is meant to give the same answer as run.py at the same depth**, and `test_engine_rs.py` checks that it does: both nets' logits agree with torch to ~1e-5, the shortlists are identical, and best move and score match at equal depth. The one thing it adds is alpha-beta inside each root subtree, which returns the same values (root children get a full window, ties go to the first candidate either way) for a fraction of the nodes. Keep it that way: a behaviour change belongs in both engines or in neither.
- **The evaluation is not duplicated.** `main.rs` includes `../rust/src/classical.rs` and `tables.rs` from the RL crate by `#[path]`, so `evaluation.py` → `rust/gen_tables.py` → both Rust engines is the only chain. The crate is separate from `rust/` because that one is a PyO3 cdylib that cannot also link as a normal binary.
- **Weights come from `save.py`, not the `.model` files.** The binary looks for the two `.safetensors` in the working directory, then next to itself and up to four directories up, or takes `--models DIR`. Retrain → rerun `save.py`, or the Rust engine keeps playing the old nets while the tests report a mismatch.
- **shakmaty refuses a position where the side not to move is in check** (python-chess did not care). Other oddities in a FEN — stale castling rights, impossible material — are ignored as python-chess ignored them. `test_engine.py`'s stalemate-trap position is one of those illegal ones; `test_engine_rs.py` uses a legal version.
- **`stop` interrupts a search; `quit` and end of input interrupt only `go infinite`.** stdin is read on a thread. A piped `go` + `quit` (or a pipe that just closes) still gets its full search, the same rule run_rl.py follows.

### Gotchas

- **`eval_pos` needs two separate index translations, and both look redundant.** The tables are in blunder's order (pawn, knight, bishop, rook, queen, king) while `encode()`'s alphabet is `".prnbqk"`, so `piece_to_table` remaps them — without it a rook scores off the knight table. And the tables are written from White's side with a8 first, which is already the orientation `encode()` produces, so White reads them straight and only Black is mirrored (`pov_list`) — the opposite of what blunder does, because blunder's own squares are a1-first. Both of these were wrong before, and "simplifying" either one back re-breaks the engine badly rather than subtly. `test_engine.py` checks material ordering and that `eval_pos` is antisymmetric under `board.mirror()`.
- **The model classes in `run_from.py` and `save.py` reference `nn.` but never import it.** This is not a bug to fix blindly: the models are saved with `torch.save(model)` (full pickled module), so `torch.load` resolves the class from `__main__` of whatever script is loading. `__init__` never runs on load, so the missing import is harmless — but it means **the `forward` that actually executes is the one in the loading script, not the one it was trained with**. `run.py`'s `ChessModel.forward` deliberately differs from `train5.py`'s: it adds `board_in.repeat(len(piece_in), 1)` to score one board against many candidate source squares in a batch. Keep class and attribute names identical across these files or loading breaks. `run.py` does import `nn` and additionally publishes both classes into `__main__` from `load_model()`, which is what lets `test_engine.py` import it.
- **`torch.load` needs `map_location` and `weights_only=False`.** The checkpoints are whole pickled modules saved from a CUDA box: without `map_location` they refuse to load on a CPU-only machine, and on torch 2.6+ `weights_only` defaults to `True` and rejects them outright. `load_model()` in `run.py` handles both; `run_from.py` and `save.py` still do not.
- **Two different square numberings are in play.** `encode()` returns the 64 squares in FEN reading order (index 0 = a8), while model *outputs* and labels use python-chess numbering (index 0 = a1, see `num_to_sqr`). Consistent between training and inference, so don't "fix" one side alone. `encode_board()` is the same encoding taken straight off a `chess.Board` to skip the FEN round-trip on the hot path — `sq ^ 56` is the conversion between the two numberings, and `test_engine.py` checks the two encoders agree.
- **`eval_pos`'s colour indices come from `encode()`, where 1 is white and 0 is black** — the opposite of `chess.WHITE`. `leaf_eval()` does that mapping in one place; there is no longer a `flip_idx` shuffle at the leaves.
- **A `chess.Move` with no promotion field is not a legal promotion.** Any code that builds moves as `chess.Move(from, to)` silently cannot queen a pawn, which is fatal in endgames. `candidates()` picks the matching legal move out of the move list instead of constructing one.
- Training samples only positions where the *winner* was to move, from games averaging 1900+ Elo (1600+ in the older `experiments/` scripts), and discards games under 6 plies. Nothing in the nets or the evaluation was trained on opening theory or endgame technique, so the search and `eval_pos` are carrying both.

## RL engine

A second engine, sharing nothing with the imitation one but `evaluation.py` (and that only for adjudication and the arena's `greedy`/`classical` opponents). AlphaZero-style: one `RLNet` with a policy head over `N_MOVES` = 4168 (from*64+to, plus 72 underpromotion slots) and a win/draw/loss head, driving a PUCT search. Two trunks share the class: `arch='transformer'` (the default for new runs: 64 square tokens, attention layers, policy read off from-square x to-square pairs, value off the mean token) and `arch='mlp'` (the first runs: flattened embeddings into a residual MLP; a config without `arch` is one). The MLP plateaued at ~1000 Elo because every pattern had to be learned once per square.

- [rl_model.py](rl_model.py) — `RLNet`, `encode()`/`move_index()`, save/load. Saved as `{'config', 'state_dict'}`, loaded with `weights_only=True` — none of the pickled-module business above applies.
- [rl_mcts.py](rl_mcts.py) — the search. `MCTS.step(trees, leaves_per_tree)` descends every tree, evaluates all new leaves in one batch, backs up. Self-play batches across many trees (1 leaf each); the engine batches within one tree using virtual loss. Also `game_over()`, `forced_mate()`.
- [train_rl.py](train_rl.py) — actors (spawned processes, CPU unless CUDA) play `--games-per-actor` games each and push finished games to a queue; the learner (main process, CUDA > MPS > CPU) trains from a replay buffer, capped by `--replay-ratio`, and publishes weights to a flat shared-memory tensor that actors re-pull when `version` changes. Optional evaluator process runs `rl_eval.play_match` periodically.
- [run_rl.py](run_rl.py) — UCI loop; stdin is read on a thread so `stop` can interrupt `go infinite`. Reuses the tree between moves when the new position extends the old one.
- [rl_eval.py](rl_eval.py) — players (`MCTSPlayer`, `GreedyPlayer`, `RandomPlayer`, `UCIPlayer`) and `play_match`, which plays paired games from shared random openings.
- [rl_backend.py](rl_backend.py) — the one interface the three above use for searching: `PythonBackend` over rl_mcts.py, `RustBackend` over [rust/](rust/) (PyO3 + shakmaty + rayon; installs as the module `nighty_rs`). The directory is not called `nighty_rs` on purpose: a directory of that name at the repo root shadows the module as an empty namespace package, and `rust_available()` also checks for `Searcher` for the same reason. `make_backend('auto')` takes Rust when built. Both trees keep a python-chess `Board` mirror (`tree.board`) that advances once per move, for adjudication, `forced_mate` and the engine's tree-reuse check.
- [rust/src/lib.rs](rust/src/lib.rs) — the per-simulation half of rl_mcts.py in Rust: a `Searcher` holds all trees of a process, `collect()` walks them in parallel and returns one batch of leaves, `apply()` takes the net's answers. Python only touches roots (moves, visits, encodings, game over).

With the Rust search, self-play is bound by inference: ~180–200k simulations/s on an M5 Pro from one process with 4096 games and the net on MPS (one thread alone ~44k; 8192 games ~300k in isolation), so with Rust the defaults are two actors of 4096 games with the cores split between them (~300k together; one actor ~200k because its walk/net/backup chain is serial and leaves both CPU and GPU idle in turns; a third adds little), net on the best device. ~9 GB RSS per actor at 4096 games. With the Python search it is bound by Python instead: roughly 65k simulations/s over 16 actors (~8k for one actor alone). python-chess legal-move generation is the largest single cost (~38%); the forward pass is ~20% of one actor's time and ~30% with 16 actors, because a call's latency rises under contention. MPS actors measured no faster since a GPU call has a ~1.2 ms floor that grows to ~4 ms with 16 processes sharing the GPU — hence actors default to CPU on a Mac. A Rust search (shakmaty legal movegen measured at 40–95 ns/position, ~300x python-chess) would make inference the limit; it only pays off fully with large GPU batches (≥512 leaves per call), i.e. one inference process rather than one small batch per actor.

### RL gotchas

- **The RL encoding is side-to-move relative and python-chess numbered (a1 = 0)** — the board is flipped (`sq ^ 56`) and colours swapped when Black is to move, and moves are flipped the same way. That is a third convention next to the two above; don't mix `encode()` from run.py and from rl_model.py. `test_rl.py` checks `encode(b) == encode(b.mirror())`.
- **Inside the tree a single repetition is a draw, but never at the root.** `descend()` skips `terminal_value` for the root, and `Tree.advance()` refuses to reuse a child marked terminal; otherwise a game that has repeated once gets a root that can never be expanded.
- **Untrained-net ties.** The policy head is deliberately *not* zero-initialised, and `best_child(node, rng)` breaks ties at random in self-play. With exactly uniform priors, argmax descends into the first legal move everywhere and early self-play is ~90% threefold repetitions.
- **Every own move becomes a row, but only full searches carry a policy target** (playout cap randomisation: `--full-prob` of moves get `--sims`, the rest `--fast-sims`; a quick search's visits are stored with `weight` 0 and train the value head only). `losses()` weights the policy term by that column; `policy_positions_per_s` in the log is the rows with a target, about a quarter of positions/s.
- **What leaves the learner is a running average of the weights** (`--ema`, 0 = raw): actors, evaluator, snapshots and `nighty_rl.pt` all get it, the checkpoint keeps both. Resuming an old checkpoint without `averaged` starts the average from the raw weights.
- **Games that reach `--max-plies` are adjudicated with `evaluation.py`** — the one place hand-written knowledge enters training. Changing the eval changes the RL value targets for those games.
- **Games are played against an opponent mix, not only self-play.** `OpponentPool` in train_rl.py loads the newest `--snapshot-pool` files from `rl_snapshots/` as extra backends (each snapshot net gets its own `Searcher`; all share rayon's global pool via `nighty_rs.set_threads`), plus `random` and `greedy` movers that do not search. `SelfPlayGame` swaps its tree to whichever backend plays the side to move (fresh tree from the board when the net changes; `RustTree` starts from the last irreversible move only, and Rust's `ply()` comes from the FEN's move number for that reason). Only the current net's moves become rows; `record['ours']` feeds the per-opponent win rates.
- **`rust/src/classical.rs` is a port of `eval_pos`, and `rust/src/tables.rs` is generated from evaluation.py by `rust/gen_tables.py`.** Change the Python evaluation → rerun the generator and check the eval terms; `test_rl.py` diffs the two scores over 3000 positions × both sides. The `classical` opponent kind (`nighty_rs.classical_moves`, batched, rayon) is this eval under a `--classical-depth`-ply alpha-beta with no quiescence.
- **Stockfish is a yardstick, never an opponent.** `evaluator_main` plays it every `--stockfish-every` steps via `rl_eval.stockfish_player` (UCI_LimitStrength/UCI_Elo, clamped to the binary's range) and logs `eval/stockfish_elo`; the level follows the last estimate. `OpponentPool` deliberately has no stockfish kind.
- **Resignation is gated by a live false-resign rate.** `--no-resign-frac` of games can never resign and instead report whether a side that crossed `--resign` actually lost; the learner flips the shared `resign_on` flag only while that error rate is under `--resign-false-max`. Measured 37% wrong after 12 minutes of training, so an unconditional threshold poisons early labels.
- **In run_rl.py only `stop` interrupts a search; `quit` interrupts only `go infinite`.** stdin is read ahead on a thread, so a piped `go movetime N` + `quit` would otherwise stop after one step and play the raw prior.
- **The Rust and Python encodings must stay identical** (`encode`, `move_index`, castling as e1g1, promotion numbers) — a net trained with one is played with the other. `test_rl.py` diffs them over 3000 random positions; change both or neither.
- **`RustBackend` pipelines over `lanes` searchers, and a tree with leaves out at the net is not done.** Each lane keeps one net call in flight (`Evaluator.submit`/`fetch`): a lane's batch is collected and submitted in one `step()` and applied at the start of the next, so walking one lane overlaps the GPU pass of the other. Rust's `done()` therefore requires `pending` to be empty, and `advance`/`reset_search`/`drop_tree` refuse a tree with leaves in flight; `RustTree` settles its lane first when a search is cut short (the engine's `stop`). Actors default to 2 lanes on CUDA and 1 elsewhere (`--actor-lanes`): on MPS a call has a ~1 ms floor, so halving the batch per call lost more than the overlap gained. All lanes share rayon's global pool and are walked in turn, so they never contend. A `RustTree.close()` frees the Rust tree; `SelfPlayGame.finish()` reads `ply()` before closing.

## Layout

`experiments/` holds superseded training scripts and checkpoints (`train.py`–`train4.py`, smaller/narrower nets); 620x3 beat the 512x2 variant. Nothing in the live pipeline imports from it. `chess.safetensors` / `chess_from.safetensors` are `save.py` exports; nothing in Python reads them, `engine/` (the Rust build of the imitation engine, above) does. `engine/target/` and `rust/target/` are build output.
