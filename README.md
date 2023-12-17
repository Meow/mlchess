# ML-based chess bot

Trained on [lichess open database](https://database.lichess.org/), filtering for games between 1600+ rated players for initial training, and 1900+ rated players for subsequent runs.

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

**You can run it on Linux only. If you're on Windows, use WSL.**

1. `pip3 install torch numpy safetensors wandb chess`
2. `python3 run.py`
3. The script speaks UCI. Feed it UCI commands like:

```
position startpos moves g1f3 d7d5 e2e3 c8e6 f1b5 b8c6 b5c6 b7c6 f3d4 d8d6 b1c3 e8c8 d4e6 f7e6 e1g1 d5d4 e3d4 d6d4 f1e1 d8d5 e1e6 d5c5 e6e4 d4d6 d1g4 e7e6
go aaa
```

Sample output:

```
info string [(Move.from_uci('d6d7'), 559.0), (Move.from_uci('c5f5'), 428.0), (Move.from_uci('c5f5'), 484.0), (Move.from_uci('d6d7'), 649.0), (Move.from_uci('g8f6'), 518.0), (Move.from_uci('c5f5'), 453.0), (Move.from_uci('c5f5'), 324.0), (Move.from_uci('c5f5'), 309.0), None, (Move.from_uci('c5f5'), 397.0), (Move.from_uci('g8f6'), 388.0), None]
info string Highest Eval: 649.0
info string Playing: (1, 1)
bestmove e4e6
```

You might want to edit the `run.py` script and change `search_moves`, `search_pieces` and `search_depth` variables if it runs too slowly on your machine, especially `search_depth`. The values of 4 or 5 seem the most sane.

It's designed to work with `lichess-bot`. Put this folder in `lichess-bot/engines/nightybot`. The config should be:

```yml
engine:
  dir: "./engines/nightybot/"
  name: "run.sh"
  working_dir: "./engines/nightybot/"
  protocol: "uci"
```

### Model layer layout

Both models have an input layer, 3 middle layers, and 1 output layer. The feature size for all of them is 620 tensors. The input dimension is 1024 for the "from" model, and 1088 (1024 + 64) for the "where" model. The feature size (the size of the middle layers) was picked arbitrarily at first, and after experimenting, 3x 620 tensor middle layers has shown to train better than 512 tensor 2 middle layer variant.
