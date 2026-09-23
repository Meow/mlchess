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

### Reinforcement-learning engine

There's a second, separate engine that learns entirely from self-play, AlphaZero style: one net (`nighty_rl.pt`) with a policy head (which move) and a value head (win/draw/loss), and a PUCT tree search that uses both. It doesn't read the lichess database and doesn't use the piece-square tables to choose moves. The one exception: self-play games still going at 300 plies are scored with `evaluation.py`, because otherwise almost every early game ends in a draw and the value head has nothing to learn from (`--adjudicate-cp 0` turns that off).

Train it:

```
python3 train_rl.py                      # uses every core, Ctrl-C to stop, resumes from where it left off
python3 train_rl.py --max-minutes 480    # e.g. overnight
python3 train_rl.py --help               # all the knobs
```

It starts from a blank net, so expect it to lose to everything for the first hour or so. Training is parallel in three ways:

- **Self-play.** With the Rust search (below) two processes each play 4096 games at once: the tree search runs across the cores, and one forward pass on the GPU scores a position from every game. Two rather than one because each step is a chain (walk the trees, run the net, back up) and a second process fills the gaps the first leaves on the CPU and the GPU. Without the Rust search, all cores but two each run a Python process of 48 games.
- **The learner** trains on the GPU while the games are being played. It uses CUDA, or MPS on a Mac, and hands new weights to the actors through shared memory.
- **An evaluator** process plays the current net against a fixed opponent every 20 minutes, so you can see whether it's improving.

#### Opponents

Pure self-play has a known weakness: a net that only ever plays its current self can forget how to beat its earlier selves and drift in circles. So each game is played against something drawn from a mix — `self` (both sides the current net), `snapshot` (an earlier net from `rl_snapshots/`, uniform over the newest eight), `random`, or `greedy` (the 1-ply `evaluation.py` player) — with a random colour. Only the current net's own moves become training rows; every game's result trains the value head. The mix slides from `--opponents` (default `self=0.7,snapshot=0.2,random=0.1`) to `--opponents-final` (default `self=0.7,snapshot=0.3`) over `--opponents-steps` training steps. The progress line then carries win rates per opponent (`vs random 96% snapshot 58%`), which is a strength meter you get for free; wandb has them as `vs/*`. `greedy` runs in Python at a millisecond or two per move, and at 16k games a 10% share was measured to halve throughput: use `greedy=0.02` or so. `--fresh` moves the previous run's checkpoint, weights and snapshots to `rl_previous_<time>/` so a new net neither overwrites them nor trains against them.

#### The Rust search

The per-simulation work (move generation, encoding, PUCT selection, backup) is also implemented in Rust, in `rust/`, behind exactly the same interface (`rl_backend.py`). Everything picks it up automatically once it is built, and `--backend python` gets the pure-Python one back. To build it you need a Rust toolchain ([rustup](https://rustup.rs)); pip does the rest, and it must be the same `python3` you run the scripts with:

```
python3 -m pip install ./rust
python3 test_rl.py          # checks the two agree on every token and move index
```

(For hacking on the Rust itself, `pip3 install maturin` and `cd rust && maturin develop --release` rebuilds in place.) `train_rl.py` says which backend it is using on its first line; if it says `python` when you expected `rust`, the extension isn't installed for that interpreter.

On an M5 Pro it moved self-play from ~65k simulations/s (16 Python processes on the CPU) to ~300k (two processes of 4096 games, net on the GPU; ~200k with one), and one search thread alone does ~44k against ~8k for a Python process. The net is now the limit, so this is where a bigger GPU pays. Each actor holds about 9 GB at 4096 games; `--actors 2 --games-per-actor 8192` reached ~340k here at twice that. Expect the machine to look half idle even so: the hand-offs between search, net and back-up are serial within each process.

Progress lines look like this:

```
step 529 | buffer 67,671 | games 1,818 (+252, 250.5/min) | positions/s 174 | sims/s 55,399 | plies 165 | W/D/B 30/28/40% | mate 30% resign 28% adjudicated 12% repetition 10% material 10% fifty 5% max-plies 2% | policy 2.879 (kl 0.277) value 0.449 | steps/s 1.4
eval vs greedy (net v12, 100 sims): +2 =0 -18  score 10.0%  elo -382 +/- 254  [20s]
```

(Those are from the first twelve minutes of a run on a 16-core Mac. `resign on/off (wrong N%)` also appears once there is data: resigning is only allowed while the games that may never resign show it would have been right at least 95% of the time, because a wrongly resigned game is a wrong training label — and a young value head gets that wrong about a third of the time.)

It writes `nighty_rl.pt` (the weights), `rl_checkpoint.pt` (to resume), and a copy of the weights every 10000 steps in `rl_snapshots/`. The replay buffer isn't saved, so after a restart it spends a couple of minutes refilling before it trains again.

With `--wandb` it logs everything in those progress lines, the evaluation results and the resignation stats to [Weights & Biases](https://wandb.ai) (`pip3 install wandb`, `wandb login` once), syncs `nighty_rl.pt` whenever it is rewritten, and uploads the final weights as a model artifact called `nighty_rl` when it stops. The run id is stored in the checkpoint, so stopping and restarting continues the same run. `--wandb-project` and `--wandb-name` name things; `WANDB_MODE=offline` logs to a local `wandb/` directory you can `wandb sync` later.

Play it:

```
python3 run_rl.py        # same UCI commands as run.py
```

For lichess-bot, use `name: "run_rl.sh"` instead of `run.sh` in the config above. It spends its whole time budget searching, and also understands `go nodes N`. `go depth` is ignored, because depth doesn't mean anything to a tree search. It uses the Rust search when built (`--backend python` otherwise); `--threads` is its search threads.

Measure it:

```
python3 rl_eval.py --opponent greedy:2 --games 40
python3 rl_eval.py --opponent nightybot                  # vs the imitation engine in this repo
python3 rl_eval.py --opponent rl:rl_snapshots/step_0010000.pt
python3 rl_eval.py --opponent uci:stockfish --opponent-time 0.05
```

Every 1000 training steps the evaluator also plays 20 games against [Stockfish](https://stockfishchess.org) held down with its own `UCI_Elo` handicap (`brew install stockfish` / `dnf install stockfish`; the Docker image has it), picks the level nearest the last estimate so the score stays informative, and logs the result as `eval/stockfish_elo` in wandb. That is the number to quote. Stockfish is only ever a yardstick, never an opponent in training: training against it would just be distilling Stockfish. `--stockfish-every 0` turns it off; `rl_eval.py --opponent stockfish:1500` runs one match by hand. Note the estimate is as good as Stockfish's handicap calibration, which assumes a normal time control; at `--stockfish-time 0.1` treat it as a consistent scale rather than a lichess rating.

`python3 test_rl.py` checks all of this in about a minute without needing a trained model.

#### On a Linux box with an NVIDIA GPU (Docker)

`Dockerfile` and `docker-compose.yml` build an image with PyTorch + CUDA and the Rust search, and mount the checkout at `/app` so checkpoints land in it as usual. On the host you need the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (Fedora: `sudo dnf install nvidia-container-toolkit`, `sudo nvidia-ctk runtime configure --runtime=docker`, restart docker). Then:

```
cp .env.example .env            # WANDB_API_KEY, and your uid/gid so written files are yours
docker compose build
docker compose run --rm test                                   # test_rl.py, on both backends
docker compose run --rm train --wandb --max-minutes 480        # anything after `train` goes to train_rl.py
docker compose run --rm eval --opponent greedy:2 --games 40
docker compose run --rm engine                                 # UCI on stdin
```

With a 4090 the defaults (two actors of 4096 games, net on CUDA, bf16 training) should be a starting point rather than the ceiling: `--games-per-actor 8192` or more is the first thing to try, since the GPU is the limit and bigger batches feed it better. The CUDA paths have not been run yet — everything here was developed on a Mac — so `test` is the first thing to run there. Podman users: replace the `deploy` GPU block with `devices: [nvidia.com/gpu=all]` (CDI) and drop `user:`, since podman maps the container's root to you already.

On a Mac, the learner uses the GPU (MPS) automatically. Self-play and the engine default to the CPU. `--actor-device mps` and `run_rl.py --device mps` work too, but on an M5 Pro they weren't any faster (self-play measured 62k vs 65k simulations/s). A self-play process spends about a fifth of its time in the net when it runs alone and about 30% when sixteen of them share the machine; the rest is the Python tree search and python-chess generating legal moves. A GPU call costs about a millisecond however small the batch, and several times that with sixteen processes queueing for the GPU, so it doesn't win at these batch sizes. The same goes for the engine with the Python search: on the CPU it searches about 5–7k nodes/s (`Batch` 8–32), on MPS 3–5k; with the Rust search it does 19k (`Batch` 16) to 50k (`Batch` 64) on the CPU, and the CPU still beats MPS for a single tree. The Rust search changes the picture, because a Rust move generator does a position in 40–95 ns against 13–37 µs for python-chess: self-play then wants the GPU (see above), and the numbers under "The Rust search" are with it.

On a CUDA machine the actors put their inference on the GPU by default, spread over every card. Each actor process holds its own CUDA context, which costs GPU memory and, more so, host RAM (roughly 1–2 GB per process); with many actors, start NVIDIA's MPS daemon (`nvidia-cuda-mps-control -d`) so they share the GPU without time-slicing it, or lower `--actors`.

### Model layer layout

Both models have an input layer, 3 middle layers, and 1 output layer. The feature size for all of them is 620 tensors. The input dimension is 1024 for the "from" model, and 1088 (1024 + 64) for the "where" model. The feature size (the size of the middle layers) was picked arbitrarily at first, and after experimenting, 3x 620 tensor middle layers has shown to train better than 512 tensor 2 middle layer variant.
