# ML-based chess bot

Trained on [lichess open database](https://database.lichess.org/), filtering for games between 1600+ rated players for initial training, and 1900+ rated players for subsequent runs.

Two small nets choose the moves and a hand-written piece-square evaluation scores the positions the search reaches. Nothing here was trained on opening theory or endgame technique — there's no opening book and no tablebase, so the search and the evaluation carry both ends of the game.

### Training

You don't have to train it, pre-trained snapshots of the model are provided with the repo. `chess5.model` for "which moves are best for a specific piece in this position" and `chess_from.model` for "which pieces are best to move in this position". The output of the first model are 64 tensors, representing the chess board, where indices go like "0 1 2 3 ..." = "A1 B1 C1 D1 ...". The output of the second model are 128 tensors, the first 64 represent the best moves for white, the last 64 represent the best moves for black.

Training scripts are written with my system in mind. You'll have to edit the `train5.py` and `train_from.py` scripts, and change the path to the dataset. You'll also have to comment out the wandb stuff, or initialize it on your own, otherwise you'll crash.

The type of training used is immitation learning.

To train, simply run

```
python3 train5.py
python3 train_from.py
```

### Running

1. `pip3 install torch numpy safetensors wandb chess`
2. `python3 run.py`
3. The script speaks UCI. Feed it UCI commands like:

```
position startpos moves g1f3 d7d5 e2e3 c8e6 f1b5 b8c6 b5c6 b7c6 f3d4 d8d6 b1c3 e8c8 d4e6 f7e6 e1g1 d5d4 e3d4 d6d4 f1e1 d8d5 e1e6 d5c5 e6e4 d4d6 d1g4 e7e6
go wtime 60000 btime 60000 winc 1000 binc 1000
```

Sample output:

```
info depth 1 score cp 40 time 3 pv g1f3
info depth 2 score cp 0 time 17 pv g1f3
info depth 3 score cp 14 time 161 pv g1f3
info depth 4 score cp -3 time 1659 pv d2d4
bestmove d2d4
```

It reads the clock out of `go` (`wtime`/`btime`/`winc`/`binc`/`movestogo`/`movetime`) and deepens until the budget runs out, so it will not flag in a long game. Without a clock it falls back to a 30 second cap per move.

`search_depth`, `search_moves` and `search_pieces` at the top of `run.py` are the ceilings on how deep and how wide it will go — with a clock it usually stops short of them. They're also exposed as the `Depth`, `AnalyzeMoves` and `AnalyzePieces` UCI options if your GUI would rather set them that way.

Run `python3 test_engine.py` to check an install, or after touching the evaluation or the search. It loads the real checkpoints and plays out short searches, so it takes about a minute.

It's designed to work with `lichess-bot`. Put this folder in `lichess-bot/engines/nightybot`. The config should be:

```yml
engine:
  dir: "./engines/nightybot/"
  name: "run.sh"
  working_dir: "./engines/nightybot/"
  protocol: "uci"
```

### How it picks a move

The two nets don't play the game on their own — they propose. For a position, the "from" net ranks the squares worth moving from and the "to" net ranks the destinations for each of those, and the top few of each become a shortlist of legal moves. The engine then searches that shortlist with negamax and scores the leaves with the piece-square tables in `evaluation.py` (adapted from [blunder](https://github.com/algerbrex/blunder), MIT).

Two things are deliberately not left to the nets, because a shortlist that misses them loses games outright: a mate in one is always played if one exists, and draws (stalemate, insufficient material, the fifty-move rule, and repeating a position already on the board) are scored as draws rather than as whatever the tables happen to say. Everything else is the nets' choice.

### Model layer layout

Both models have an input layer, 3 middle layers, and 1 output layer. The feature size for all of them is 620 tensors. The input dimension is 1024 for the "from" model, and 1088 (1024 + 64) for the "where" model. The feature size (the size of the middle layers) was picked arbitrarily at first, and after experimenting, 3x 620 tensor middle layers has shown to train better than 512 tensor 2 middle layer variant.
