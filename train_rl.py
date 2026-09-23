"""Train NightyBot's reinforcement-learning model (nighty_rl.pt) from self-play.

AlphaZero-style: the net plays itself with a PUCT search, then learns to
predict what the search chose (policy) and who went on to win (value). No
lichess games and no hand-written evaluation go in -- apart from one
fallback, see adjudicate().

  python3 train_rl.py                     # every core; resumes from rl_checkpoint.pt
  python3 train_rl.py --actors 8 --sims 400
  python3 train_rl.py --help              # everything else

Stop it with Ctrl-C; it saves before exiting.

Everything runs at once, in separate processes:

  actors     --actors of them, default all cores but two. Each plays
             --games-per-actor games at the same time and scores one leaf from
             every game in a single forward pass, so the net always sees a real
             batch even though the search itself is plain Python. This is
             where nearly all the time goes, so it gets nearly all the cores.
  learner    this process. Holds a replay buffer of recent positions and
             trains on it on the fastest device there is (CUDA, then MPS),
             handing new weights to the actors through shared memory.
  evaluator  one more, every --eval-every minutes: plays the current net
             against a fixed opponent (rl_eval.py) so there is a number to
             watch going up.

Written to --out-dir:
  nighty_rl.pt        the weights run_rl.py plays with
  rl_checkpoint.pt    weights and optimizer state, for resuming
  rl_snapshots/       a copy of the weights every --snapshot-every steps
"""

import argparse
import contextlib
import os
import queue
import signal
import time
from collections import Counter, deque

import chess
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import rl_eval
from rl_backend import make_backend, rust_available
from rl_mcts import game_over
from rl_model import RLNet, cpu_state, load_net, pick_device, save_net, torch_load

# Training rows keep at most this many moves. No position from a real game
# comes near it (the record is 218, and ~40 is typical); past it, the least
# visited moves are the ones dropped.
MAX_MOVES = 128

def parse_args(argv=None):
  cores = os.cpu_count() or 2
  p = argparse.ArgumentParser(
    description='Self-play reinforcement learning for nighty_rl.pt. '
                'See the top of train_rl.py for how the pieces fit together.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  g = p.add_argument_group('parallelism')
  g.add_argument('--backend', default='auto', choices=['auto', 'rust', 'python'],
                 help='tree search: the Rust one (nighty_rs, ~4-5x faster) if it is built, else Python')
  g.add_argument('--actors', type=int, default=None,
                 help='self-play processes (default: 2 with Rust, all cores but two with Python)')
  g.add_argument('--games-per-actor', type=int, default=None,
                 help='games each actor plays at once; also its inference batch size '
                      '(default: 4096 with Rust, 48 with Python)')
  g.add_argument('--actor-device', default='auto',
                 help="where actors run the net. 'auto': with Rust the best device there is "
                      "(CUDA, MPS, CPU); with Python CUDA if any, else CPU")
  g.add_argument('--actor-threads', type=int, default=0,
                 help='search threads per Rust actor; 0 = the cores shared out between the actors')
  g.add_argument('--learner-device', default='auto', help="'auto' is CUDA, then MPS, then CPU")
  g.add_argument('--learner-threads', type=int, default=2,
                 help='torch threads for the learner, if it ends up on the CPU')

  g = p.add_argument_group('self-play')
  g.add_argument('--sims', type=int, default=200,
                 help='simulations for a full search, the only kind that becomes training data')
  g.add_argument('--fast-sims', type=int, default=40,
                 help='simulations for a fast search, which only moves the game along')
  g.add_argument('--full-prob', type=float, default=0.25,
                 help='chance a move gets a full search; 1 turns fast searches off')
  g.add_argument('--c-puct', type=float, default=1.75)
  g.add_argument('--fpu-reduction', type=float, default=0.25)
  g.add_argument('--dirichlet-alpha', type=float, default=0.3)
  g.add_argument('--dirichlet-frac', type=float, default=0.25)
  g.add_argument('--temp-plies', type=int, default=30,
                 help='plies at the start of each game where moves are sampled, not picked')
  g.add_argument('--max-plies', type=int, default=300,
                 help='games this long are stopped and adjudicated')
  g.add_argument('--adjudicate-cp', type=int, default=300,
                 help='at --max-plies, a static eval this far ahead wins; 0 calls every one a draw')
  g.add_argument('--resign', type=float, default=-0.9,
                 help='resign when the best move is worth less than this; -1 never resigns')
  g.add_argument('--no-resign-frac', type=float, default=0.2,
                 help='fraction of games never allowed to resign; they measure how often a '
                      'resignation would have been wrong')
  g.add_argument('--resign-false-max', type=float, default=0.05,
                 help='resigning is only allowed while the measured rate of wrong resignations '
                      'is below this')

  g = p.add_argument_group('opponents')
  g.add_argument('--opponents', default='self=0.7,snapshot=0.2,random=0.1',
                 help='what each game is played against, with weights: self (both sides the '
                      'current net), snapshot (an earlier net from rl_snapshots/), random, greedy '
                      '(1-ply evaluation.py; it runs in Python at ~1 ms a move, so at thousands of '
                      'games a 10%% share halves throughput -- use 0.02 or so)')
  g.add_argument('--opponents-final', default='self=0.7,snapshot=0.3',
                 help='the mix at --opponents-steps and after; interpolated linearly until then')
  g.add_argument('--opponents-steps', type=int, default=20000)
  g.add_argument('--snapshot-pool', type=int, default=8,
                 help='how many of the newest snapshots opponents are drawn from')
  g.add_argument('--snapshot-refresh', type=float, default=5,
                 help='minutes between looks at rl_snapshots/ for new ones')

  g = p.add_argument_group('model (ignored when resuming: the checkpoint wins)')
  g.add_argument('--width', type=int, default=512)
  g.add_argument('--blocks', type=int, default=4)

  g = p.add_argument_group('learning')
  g.add_argument('--batch-size', type=int, default=512)
  g.add_argument('--lr', type=float, default=1e-3)
  g.add_argument('--weight-decay', type=float, default=1e-4)
  g.add_argument('--buffer', type=int, default=250000, help='replay buffer size, in positions')
  g.add_argument('--min-buffer', type=int, default=20000,
                 help='positions to collect before the first training step')
  g.add_argument('--replay-ratio', type=float, default=4.0,
                 help='times each position is trained on, on average; the learner waits for '
                      'the actors rather than go over it')
  g.add_argument('--publish-every', type=int, default=50,
                 help='steps between handing new weights to the actors')

  g = p.add_argument_group('evaluation')
  g.add_argument('--eval-every', type=float, default=20,
                 help='minutes between evaluation matches; 0 turns the evaluator off')
  g.add_argument('--eval-opponent', default='greedy',
                 help='random | greedy[:depth] | nightybot | uci:<command> | rl:<weights>')
  g.add_argument('--eval-games', type=int, default=20)
  g.add_argument('--eval-sims', type=int, default=100)
  g.add_argument('--stockfish-every', type=int, default=1000,
                 help='training steps between Elo measurements against a handicapped Stockfish '
                      '(never trained on); 0 turns them off. Needs a stockfish binary.')
  g.add_argument('--stockfish-games', type=int, default=20)
  g.add_argument('--stockfish-time', type=float, default=0.1, help="Stockfish's seconds per move")

  g = p.add_argument_group('output')
  g.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
  g.add_argument('--save-every', type=int, default=1000, help='steps between checkpoints')
  g.add_argument('--snapshot-every', type=int, default=5000,
                 help='steps between copies kept in rl_snapshots/ (also the opponent pool); 0 keeps none')
  g.add_argument('--log-every', type=float, default=30, help='seconds between progress lines')
  g.add_argument('--max-steps', type=int, default=0,
                 help='stop after this many training steps in this run; 0 runs until Ctrl-C')
  g.add_argument('--max-minutes', type=float, default=0,
                 help='stop after this long, e.g. 480 for overnight; 0 runs until Ctrl-C')
  g.add_argument('--fresh', action='store_true',
                 help='start from a new net; the existing checkpoint, weights and snapshots '
                      'are moved to rl_previous_<time>/ first')
  g.add_argument('--wandb', action='store_true',
                 help='log to Weights & Biases (pip3 install wandb; wandb login). A resumed '
                      'checkpoint continues its run. Set WANDB_MODE=offline to log locally.')
  g.add_argument('--wandb-project', default='mlchess')
  g.add_argument('--wandb-name', default=None, help='run name (default: wandb picks one)')
  g.add_argument('--seed', type=int, default=None)
  args = p.parse_args(argv)
  # A search of fewer than 2 simulations never visits a child, and there is
  # no move to pick from that.
  args.sims = max(2, args.sims)
  args.fast_sims = max(2, args.fast_sims)
  if args.backend == 'auto':
    args.backend = 'rust' if rust_available() else 'python'
  # The Rust search runs every game of a process on all cores and wants a
  # batch big enough to keep a GPU busy; the Python one is one core per
  # process and cannot use a GPU well at any batch size. Two Rust actors
  # rather than one because a step is a chain -- walk trees, run the net,
  # back up -- and with one process the cores idle while the GPU works and
  # vice versa; a second process fills those gaps (measured 1.5x, a third
  # adds little). Each one holds ~9 GB at 4096 games.
  rust = args.backend == 'rust'
  if args.actors is None:
    args.actors = 2 if rust else max(1, cores - 2)
  if args.games_per_actor is None:
    args.games_per_actor = 4096 if rust else 48
  if rust and args.actor_threads == 0:
    args.actor_threads = max(1, cores // args.actors)
  args.opponents = parse_mix(args.opponents)
  args.opponents_final = parse_mix(args.opponents_final)
  return args

OPPONENT_KINDS = ('self', 'snapshot', 'random', 'greedy')

def parse_mix(spec):
  """'self=0.7,snapshot=0.3' -> {'self': 0.7, 'snapshot': 0.3, ...}, normalised."""
  mix = {kind: 0.0 for kind in OPPONENT_KINDS}
  for part in spec.split(','):
    kind, _, weight = part.partition('=')
    if kind.strip() not in mix:
      raise SystemExit(f'unknown opponent {kind!r}; choose from {", ".join(OPPONENT_KINDS)}')
    mix[kind.strip()] = float(weight or 1)
  total = sum(mix.values())
  if total <= 0:
    raise SystemExit('the opponent mix needs at least one positive weight')
  return {k: v / total for k, v in mix.items()}

def mix_at(args, step):
  """The opponent mix for a training step: --opponents sliding to --opponents-final."""
  t = 1.0 if args.opponents_steps <= 0 else min(1.0, step / args.opponents_steps)
  return {k: (1 - t) * args.opponents[k] + t * args.opponents_final[k] for k in OPPONENT_KINDS}

# --- weights, shared between processes ----------------------------------------
#
# The learner owns the real net. Actors each keep a private copy for inference
# and refresh it from one flat shared-memory tensor whenever `version` moves.

def publish(net, shared, lock, version):
  flat = parameters_to_vector(net.parameters()).detach().float().cpu()
  with lock:
    shared.copy_(flat)
    version.value += 1

def pull(net, shared, lock, version):
  with lock:
    flat = shared.clone()
    seen = version.value
  vector_to_parameters(flat.to(next(net.parameters()).device), net.parameters())
  return seen

# --- self-play ----------------------------------------------------------------

def adjudicate(board, margin):
  """Games that reach --max-plies are scored by evaluation.py's material and
  piece placement. This is the one place hand-written knowledge gets in: an
  untrained net shuffles pieces around for hundreds of moves, and without it
  nearly all of the early games would be draws with nothing in them for the
  value head to learn. Once the net can finish games itself it rarely comes
  up. --adjudicate-cp 0 calls them all draws instead."""
  if margin <= 0:
    return 0, 'max-plies'
  score = rl_eval.static_eval(board)
  if abs(score) < margin:
    return 0, 'max-plies'
  return (1 if score > 0 else -1), 'adjudicated'

class OpponentPool:
  """What a game can be played against besides the current net: earlier nets
  from rl_snapshots/ (each with its own search backend), a random mover, or
  the 1-ply greedy player from rl_eval.py.

  Playing only against itself, a net can forget how to beat what it used to
  be and drift in circles; a mix of old selves and fixed players keeps the
  games varied and gives a strength meter for free (the win rates in the
  progress line)."""

  def __init__(self, args, device, rng, snapshot_dir):
    self.args = args
    self.device = device
    self.rng = rng
    self.snapshot_dir = snapshot_dir
    self.snapshots = {}      # file name -> backend playing that net
    self.greedy = rl_eval.GreedyPlayer(1)
    self.checked = 0.0
    self.refresh()

  def refresh(self):
    if time.time() - self.checked < self.args.snapshot_refresh * 60 and self.checked:
      return
    self.checked = time.time()
    try:
      names = sorted(n for n in os.listdir(self.snapshot_dir) if n.endswith('.pt'))
    except FileNotFoundError:
      names = []
    names = names[-self.args.snapshot_pool:]
    for name in list(self.snapshots):
      if name not in names:
        del self.snapshots[name]
    for name in names:
      if name not in self.snapshots:
        try:
          net = load_net(os.path.join(self.snapshot_dir, name), self.device).eval()
        except Exception as e:  # half-written, or from another net shape
          print(f'skipping snapshot {name}: {e}', flush=True)
          continue
        self.snapshots[name] = make_backend(self.args.backend, net, self.device,
                                            self.args.c_puct, self.args.fpu_reduction)

  def backends(self):
    return list(self.snapshots.values())

  def choose(self, step):
    """(kind, backend or None) for a new game."""
    mix = mix_at(self.args, step)
    kind = self.rng.choice(OPPONENT_KINDS, p=[mix[k] for k in OPPONENT_KINDS])
    if kind == 'snapshot':
      if not self.snapshots:
        return 'self', None
      name = self.rng.choice(sorted(self.snapshots))
      return 'snapshot', self.snapshots[name]
    return str(kind), None

class SelfPlayGame:
  """One game in progress inside an actor, and the training rows it has
  produced so far. The current net plays both sides, or one side against an
  opponent from the pool; only the current net's own moves become rows."""

  def __init__(self, args, rng, resign_on, backend, pool=None, step=0):
    self.args = args
    self.rng = rng
    self.resign_on = resign_on  # shared flag the learner sets, see actor_main
    self.backend = backend
    self.pool = pool
    self.kind, self.opponent = pool.choose(step) if pool else ('self', None)
    self.us = chess.WHITE if self.kind == 'self' or rng.random() < 0.5 else chess.BLACK
    self.board = chess.Board()
    self.tree = None
    self.tree_backend = None
    self.rows = []
    self.may_resign = rng.random() >= args.no_resign_frac
    self.would_resign = set()   # colours that crossed the threshold in a no-resign game
    self.start_search()

  def ours(self):
    return self.kind == 'self' or self.board.turn == self.us

  def mover_backend(self):
    """The search backend for the side to move; None if it does not search."""
    if self.ours():
      return self.backend
    return self.opponent if self.kind == 'snapshot' else None

  def start_search(self):
    a = self.args
    backend = self.mover_backend()
    if backend is None:
      self.full = False
      return
    # The tree lives on from move to move while the same net is searching;
    # when the other side's net takes over it starts afresh from the board.
    if self.tree_backend is not backend:
      if self.tree is not None:
        self.tree.close()
      self.tree = backend.tree(self.board)
      self.tree_backend = backend
    # Playout cap randomisation (KataGo): most moves get a quick search that
    # only keeps the game going, a few get a full one that becomes a training
    # row. More finished games per hour, which is what the value head eats.
    # Opponents only ever get the quick search, and no noise.
    self.full = self.ours() and self.rng.random() < a.full_prob
    noise = (a.dirichlet_alpha, a.dirichlet_frac) if self.full else None
    self.tree.reset_search(a.sims if self.full else a.fast_sims, noise)

  def ready(self, done_ids):
    """Is there a move to play? done_ids maps a backend to its finished tree
    ids (or None, meaning ask the tree)."""
    if self.mover_backend() is None:
      return True
    done = done_ids.get(id(self.tree_backend))
    return self.tree.done() if done is None else self.tree.id in done

  def search_done(self):
    return self.ready({})

  def play_move(self):
    """Record the finished search, play a move, and return the game's record
    if that was the end of it."""
    a = self.args
    board = self.board
    turn = board.turn
    if self.mover_backend() is None:
      if self.kind == 'greedy':
        move = self.pool.greedy.best(board)
      else:
        legal = list(board.legal_moves)
        move = legal[self.rng.integers(len(legal))]
      # Through our tree if we have one, so it stays on the same position as
      # the board (and keeps what it searched below this move).
      if self.tree is not None:
        self.tree.advance(move)
        over = self.tree.game_over()
      else:
        board.push(move)
        over = game_over(board)
    else:
      tree = self.tree
      visits = tree.visits()
      if self.full and visits.sum() > 0:
        self.rows.append(training_row(tree, visits, turn))
      if self.ours() and tree.q(tree.best()) < a.resign:
        if self.may_resign and self.resign_on.value:
          return self.finish(-1 if turn == chess.WHITE else 1, 'resign')
        if not self.may_resign:
          self.would_resign.add(turn)
      temperature = 1.0 if board.ply() < a.temp_plies else 0.0
      tree.advance(tree.moves()[tree.sample(temperature)])
      over = tree.game_over()

    if over is None and board.ply() >= a.max_plies:
      over = adjudicate(board, a.adjudicate_cp)
    if over is not None:
      return self.finish(*over)
    self.start_search()
    return None

  def finish(self, result, reason):
    """Package the game for the learner. result is White's score: 1, 0, -1."""
    n = len(self.rows)
    plies = self.board.ply()
    if self.tree is not None:
      self.tree.close()
      self.tree = None
    record = {'result': result, 'reason': reason, 'plies': plies,
              'positions': n, 'opponent': self.kind,
              # our score against a pool opponent, for the win rates
              'ours': None if self.kind == 'self' else result * (1 if self.us == chess.WHITE else -1),
              # for each side that would have resigned: did it actually lose?
              'resign_checks': [result == (-1 if colour == chess.WHITE else 1)
                                for colour in self.would_resign]}
    tokens = np.zeros((n, 64), np.int8)
    halfmove = np.zeros(n, np.float32)
    legal = np.full((n, MAX_MOVES), -1, np.int16)
    probs = np.zeros((n, MAX_MOVES), np.float16)
    wdl = np.zeros(n, np.int8)
    for j, (tok, hm, idx, p, turn) in enumerate(self.rows):
      tokens[j] = tok
      halfmove[j] = hm
      legal[j, :len(idx)] = idx
      probs[j, :len(p)] = p
      # win / draw / loss, for the side to move in that position
      if result == 0:
        wdl[j] = 1
      else:
        wdl[j] = 0 if (result == 1) == (turn == chess.WHITE) else 2
    record.update(tokens=tokens, halfmove=halfmove, legal=legal, probs=probs, wdl=wdl)
    return record

def training_row(tree, visits, turn):
  tokens, halfmove, idx = tree.encoding()
  idx = np.asarray(idx, np.int16)
  visits = np.asarray(visits, np.float64)
  if len(idx) > MAX_MOVES:
    keep = np.argsort(-visits, kind='stable')[:MAX_MOVES]
    idx, visits = idx[keep], visits[keep]
  return tokens, halfmove, idx, visits / visits.sum(), turn

def actor_device(name, rank, backend='python'):
  if name != 'auto':
    return name
  if torch.cuda.is_available():
    return f'cuda:{rank % torch.cuda.device_count()}'
  return pick_device('auto') if backend == 'rust' else 'cpu'

def actor_main(rank, args, shared, lock, version, games_q, sims, resign_on, step_now, stop):
  # Ctrl-C reaches every process in the group; the learner handles it and
  # tells the actors to stop through `stop`.
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  # The Rust search has its own thread pool; the Python one is one process
  # per core already, and intra-op threads would only fight them.
  torch.set_num_threads(1)
  seed = None if args.seed is None else [args.seed, rank]
  rng = np.random.default_rng(seed)
  device = actor_device(args.actor_device, rank, args.backend)

  net = RLNet(**args.model_config).to(device).eval()
  seen = pull(net, shared, lock, version)
  backend = make_backend(args.backend, net, device, args.c_puct, args.fpu_reduction,
                         seed=None if args.seed is None else args.seed * 1000 + rank,
                         threads=args.actor_threads)
  pool = OpponentPool(args, device, rng, os.path.join(args.out_dir, 'rl_snapshots'))
  games = [SelfPlayGame(args, rng, resign_on, backend, pool, step_now.value)
           for _ in range(args.games_per_actor)]
  by_tree = {}   # (backend id, tree id) -> game index, for games that are searching
  idle = set()   # games whose side to move does not search (random, greedy)
  def track(n):
    game = games[n]
    if game.mover_backend() is None:
      idle.add(n)
    else:
      by_tree[(id(game.tree_backend), game.tree.id)] = n
  for n in range(len(games)):
    track(n)

  # Games are not kept in step with each other: whenever one finishes its
  # search it plays its move and starts the next, so every step() has a leaf
  # from every game and the batch never shrinks.
  while not stop.is_set():
    if version.value != seen:
      seen = pull(net, shared, lock, version)
    pool.refresh()
    backends = [backend] + pool.backends()
    # Only the games with a move to play: with thousands of games in flight
    # that is a few percent of them, and asking every one costs more than
    # the moves do.
    ready = list(idle)
    for b in backends:
      done = b.done_ids()
      if done is None:
        done = [t for (bid, t) in by_tree if bid == id(b) and games[by_tree[(bid, t)]].tree.done()]
      for t in done:
        n = by_tree.pop((id(b), t), None)
        if n is not None:
          ready.append(n)
    idle.clear()
    for n in ready:
      record = games[n].play_move()
      if record is not None:
        record['version'] = seen
        games_q.put(record)
        games[n] = SelfPlayGame(args, rng, resign_on, backend, pool, step_now.value)
      track(n)
    for b in backends:
      b.step(None)
    # Live, rather than counted off finished games, which would read low for
    # the first few minutes while every game is still in progress.
    sims[rank] = sum(b.sims for b in backends)
  # Don't let unread records hold up this process's exit.
  games_q.cancel_join_thread()

def evaluator_main(args, shared, lock, version, results_q, step_now, stop):
  """Two yardsticks, in one process so they never compete with each other:
  the fixed --eval-opponent every --eval-every minutes, and a handicapped
  Stockfish every --stockfish-every steps for an Elo number. Stockfish is
  measured against, never trained against -- otherwise the net would just be
  learning to be Stockfish."""
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  torch.set_num_threads(1)
  net = RLNet(**args.model_config).eval()
  opponent = rl_eval.make_player(args.eval_opponent, 'cpu', args.eval_sims)
  stockfish_at = args.stockfish_every if args.stockfish_every > 0 else None
  if stockfish_at and not rl_eval.stockfish_path():
    print('no stockfish binary found, so no Elo measurements (install it or set STOCKFISH)',
          flush=True)
    stockfish_at = None
  # Start against the weakest level Stockfish offers, then follow the net up:
  # a match only says something when the score is not 0% or 100%.
  level = 1320
  next_stockfish = (step_now.value // args.stockfish_every + 1) * args.stockfish_every if stockfish_at else None

  def match(against, games, seen):
    player = rl_eval.MCTSPlayer(net, 'cpu', args.eval_sims, args.c_puct, args.fpu_reduction,
                                backend=args.backend, threads=2)
    started = time.time()
    summary = rl_eval.play_match(player, against, games, seed=seen, should_stop=stop.is_set)
    if summary is not None:
      summary.update(version=seen, seconds=time.time() - started)
    return summary

  try:
    due = time.time()
    while not stop.wait(timeout=2.0):
      if next_stockfish is not None and step_now.value >= next_stockfish:
        seen = pull(net, shared, lock, version)
        stockfish = rl_eval.stockfish_player(level, args.stockfish_time)
        try:
          summary = match(stockfish, args.stockfish_games, seen)
        finally:
          stockfish.close()
        if summary is None:
          break
        estimate = rl_eval.elo_estimate(summary, stockfish.elo)
        summary.update(kind='stockfish', level=stockfish.elo, estimate=estimate, step=step_now.value)
        results_q.put(summary)
        level = int(round(estimate / 50) * 50)
        next_stockfish = (step_now.value // args.stockfish_every + 1) * args.stockfish_every
      if time.time() < due:
        continue
      seen = pull(net, shared, lock, version)
      summary = match(opponent, args.eval_games, seen)
      if summary is None:
        break
      summary['kind'] = 'fixed'
      results_q.put(summary)
      due = time.time() + args.eval_every * 60
  finally:
    opponent.close()
  results_q.cancel_join_thread()

# --- learning -----------------------------------------------------------------

class Replay:
  """A ring buffer of the most recent positions, as flat numpy arrays."""

  def __init__(self, capacity):
    self.capacity = capacity
    self.tokens = np.zeros((capacity, 64), np.int8)
    self.halfmove = np.zeros(capacity, np.float32)
    self.legal = np.full((capacity, MAX_MOVES), -1, np.int16)
    self.probs = np.zeros((capacity, MAX_MOVES), np.float16)
    self.wdl = np.zeros(capacity, np.int8)
    self.size = 0
    self.pos = 0

  def add(self, record):
    n = record['positions']
    for start in range(0, n, self.capacity):
      chunk = slice(start, min(n, start + self.capacity))
      m = chunk.stop - chunk.start
      at = (self.pos + np.arange(m)) % self.capacity
      for name in ('tokens', 'halfmove', 'legal', 'probs', 'wdl'):
        getattr(self, name)[at] = record[name][chunk]
      self.pos = (self.pos + m) % self.capacity
      self.size = min(self.capacity, self.size + m)

  def sample(self, n, rng):
    at = rng.integers(0, self.size, n)
    return (self.tokens[at], self.halfmove[at], self.legal[at],
            self.probs[at], self.wdl[at])

def to_device(batch, device):
  tokens, halfmove, legal, probs, wdl = batch
  pin = device.startswith('cuda')
  def move(a, dtype):
    t = torch.from_numpy(a).to(dtype)
    return t.pin_memory().to(device, non_blocking=True) if pin else t.to(device)
  return (move(tokens, torch.long), move(halfmove, torch.float32),
          move(legal, torch.long), move(probs, torch.float32), move(wdl, torch.long))

def losses(net, batch):
  tokens, halfmove, legal, probs, wdl = batch
  logits, wdl_logits = net(tokens, halfmove)
  # The policy is only trained over the legal moves: their logits are pulled
  # out and softmaxed on their own, and padding (-1) is masked away.
  picked = logits.float().gather(1, legal.clamp(min=0))
  picked = picked.masked_fill(legal < 0, -1e9)
  policy = -(probs * torch.log_softmax(picked, 1)).sum(1).mean()
  value = F.cross_entropy(wdl_logits.float(), wdl)
  # What the policy loss would be if the net matched the search exactly, so
  # the log can show how far off it actually is.
  entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(1).mean()
  return policy, value, entropy

class Stats:
  """Counts since the last progress line."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.started = time.time()
    self.games = 0
    self.positions = 0
    self.plies = 0
    self.results = Counter()
    self.reasons = Counter()
    self.vs = {}           # opponent kind -> [games, score for us]
    self.steps = 0
    self.policy = self.value = self.entropy = 0.0

  def add_game(self, record):
    self.games += 1
    self.positions += record['positions']
    self.plies += record['plies']
    self.results[record['result']] += 1
    self.reasons[record['reason']] += 1
    if record.get('ours') is not None:
      tally = self.vs.setdefault(record['opponent'], [0, 0.0])
      tally[0] += 1
      tally[1] += (record['ours'] + 1) / 2

  def add_step(self, policy, value, entropy):
    self.steps += 1
    self.policy += policy
    self.value += value
    self.entropy += entropy

def main():
  args = parse_args()
  ctx = mp.get_context('spawn')
  os.makedirs(args.out_dir, exist_ok=True)
  weights_path = os.path.join(args.out_dir, 'nighty_rl.pt')
  checkpoint_path = os.path.join(args.out_dir, 'rl_checkpoint.pt')
  snapshot_dir = os.path.join(args.out_dir, 'rl_snapshots')

  device = pick_device(args.learner_device)
  torch.set_num_threads(args.learner_threads)
  rng = np.random.default_rng(args.seed)
  if args.seed is not None:
    torch.manual_seed(args.seed)

  checkpoint = None
  if args.fresh:
    # A fresh net should not be overwriting the previous run's weights at its
    # first save, nor playing the previous run's snapshots as if they were its
    # own past: move that run out of the way, whole.
    old = [p for p in (checkpoint_path, weights_path, snapshot_dir) if os.path.exists(p)]
    if old:
      previous = os.path.join(args.out_dir, 'rl_previous_' + time.strftime('%Y%m%d_%H%M%S'))
      os.makedirs(previous)
      for p in old:
        os.replace(p, os.path.join(previous, os.path.basename(p)))
      print(f'--fresh: moved the previous run ({", ".join(os.path.basename(p) for p in old)}) '
            f'to {previous}', flush=True)
  elif os.path.exists(checkpoint_path):
    checkpoint = torch_load(checkpoint_path, 'cpu')
  config = checkpoint['config'] if checkpoint else {'width': args.width, 'blocks': args.blocks}
  args.model_config = config
  net = RLNet(**config)
  opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  step = games_total = positions_total = 0
  if checkpoint:
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    opt.load_state_dict(checkpoint['optimizer'])
    for group in opt.param_groups:
      group['lr'] = args.lr
      group['weight_decay'] = args.weight_decay
    step = checkpoint['step']
    games_total = checkpoint['games']
    positions_total = checkpoint['positions']
    print(f'resuming {checkpoint_path} at step {step:,} ({games_total:,} games so far)')
  net.to(device).train()
  n_params = sum(p.numel() for p in net.parameters())
  print(f'net {config}, {n_params / 1e6:.1f}M parameters; learner on {device}, '
        f'{args.actors} {args.backend} actor{"s" if args.actors > 1 else ""} x '
        f'{args.games_per_actor} games, net on '
        f'{actor_device(args.actor_device, 0, args.backend).split(":")[0]}', flush=True)

  if device.startswith('cuda') and torch.cuda.is_bf16_supported():
    autocast = torch.autocast('cuda', dtype=torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
  else:
    autocast = contextlib.nullcontext()

  if args.wandb:
    try:
      import wandb
    except ImportError:
      raise SystemExit('--wandb needs the wandb package: pip3 install wandb')
    # The run id lives in the checkpoint, so stopping and restarting training
    # keeps drawing the same curves instead of starting a new run.
    run_id = checkpoint.get('wandb_id') if checkpoint else None
    wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, id=run_id,
                           resume='allow' if run_id else None, job_type='rl',
                           config={k: v for k, v in vars(args).items() if k != 'model_config'},
                           tags=[args.backend, device])
    wandb_run.config.update({'model': config, 'parameters': n_params}, allow_val_change=True)
    # Everything is plotted against the training step, and the current
    # weights are synced whenever they are rewritten.
    wandb.define_metric('step')
    wandb.define_metric('*', step_metric='step')
    wandb.save(weights_path, base_path=args.out_dir, policy='live')

  shared = parameters_to_vector(net.parameters()).detach().float().cpu().share_memory_()
  lock = ctx.Lock()
  version = ctx.Value('q', 1)
  stop = ctx.Event()
  games_q = ctx.Queue()
  sims = ctx.Array('q', args.actors, lock=False)  # per actor, running total
  # Resigning saves a lot of self-play time, but a wrongly resigned game is a
  # wrong training label. An untrained value head gets this wrong a third of
  # the time, so it stays off until the games that may never resign show the
  # net would have been right often enough (--resign-false-max).
  resign_on = ctx.Value('i', 0)
  step_now = ctx.Value('q', step)  # actors read it to pick the opponent mix
  resign_checks = deque(maxlen=200)
  eval_q = ctx.Queue()

  actors = [ctx.Process(target=actor_main, name=f'actor-{rank}', daemon=True,
                        args=(rank, args, shared, lock, version, games_q, sims, resign_on,
                              step_now, stop))
            for rank in range(args.actors)]
  helpers = []
  if args.eval_every > 0:
    helpers.append(ctx.Process(target=evaluator_main, name='evaluator', daemon=True,
                               args=(args, shared, lock, version, eval_q, step_now, stop)))
  for proc in actors + helpers:
    proc.start()

  replay = Replay(args.buffer)
  stats = Stats()
  sims_seen = 0
  steps_run = fresh = 0

  def save(final=False):
    save_net(net, weights_path)
    tmp = checkpoint_path + '.tmp'
    state = {'config': net.config, 'state_dict': cpu_state(net),
             'optimizer': opt.state_dict(), 'step': step,
             'games': games_total, 'positions': positions_total}
    if args.wandb:
      state['wandb_id'] = wandb_run.id
    torch.save(state, tmp)
    os.replace(tmp, checkpoint_path)
    if final:
      print(f'saved {weights_path} and {checkpoint_path} at step {step:,}', flush=True)
      if args.wandb:
        # The weights as a versioned artifact, so any run's final net can be
        # pulled back with `wandb artifact get`.
        artifact = wandb.Artifact('nighty_rl', type='model',
                                  metadata={'step': step, 'games': games_total, **config})
        artifact.add_file(weights_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

  def take(record):
    nonlocal fresh, games_total, positions_total
    stats.add_game(record)
    resign_checks.extend(record['resign_checks'])
    if len(resign_checks) >= 50:
      wrong = 1 - sum(resign_checks) / len(resign_checks)
      resign_on.value = int(wrong < args.resign_false_max and args.resign > -1)
    games_total += 1
    positions_total += record['positions']
    if record['positions']:
      replay.add(record)
      fresh += record['positions']

  def report():
    nonlocal sims_seen
    s = stats
    secs = max(time.time() - s.started, 1e-9)
    sims_total = sum(sims)
    sims_rate = (sims_total - sims_seen) / secs
    sims_seen = sims_total
    games = max(s.games, 1)
    steps = max(s.steps, 1)
    line = (f'step {step:,} | buffer {replay.size:,} | games {games_total:,} '
            f'(+{s.games}, {s.games * 60 / secs:.1f}/min) | positions/s {s.positions / secs:.0f} '
            f'| sims/s {sims_rate:,.0f} | plies {s.plies / games:.0f}')
    if s.games:
      w, d, b = (s.results[k] * 100 // s.games for k in (1, 0, -1))
      ends = ' '.join(f'{k} {v * 100 // s.games}%' for k, v in s.reasons.most_common())
      line += f' | W/D/B {w}/{d}/{b}% | {ends}'
    if s.vs:
      line += ' | vs ' + ' '.join(f'{k} {v[1] * 100 / v[0]:.0f}%' for k, v in sorted(s.vs.items()))
    if resign_checks:
      wrong = 1 - sum(resign_checks) / len(resign_checks)
      line += f' | resign {"on" if resign_on.value else "off"} (wrong {wrong * 100:.0f}%)'
    if s.steps:
      line += (f' | policy {s.policy / steps:.3f} (kl {(s.policy - s.entropy) / steps:.3f}) '
               f'value {s.value / steps:.3f} | steps/s {s.steps / secs:.1f}')
    elif replay.size < args.min_buffer:
      line += f' | filling buffer to {args.min_buffer:,}'
    print(line, flush=True)
    if args.wandb:
      log = {'step': step, 'games': games_total, 'buffer': replay.size,
             'games_per_min': s.games * 60 / secs, 'positions_per_s': s.positions / secs,
             'sims_per_s': sims_rate, 'plies': s.plies / games}
      if s.games:
        log.update({f'result/{k}': s.results[v] / s.games
                    for k, v in (('white', 1), ('draw', 0), ('black', -1))})
        log.update({f'end/{k}': v / s.games for k, v in s.reasons.items()})
      log.update({f'vs/{k}': v[1] / v[0] for k, v in s.vs.items()})
      log.update({f'opponents/{k}': v for k, v in mix_at(args, step).items()})
      if resign_checks:
        log['resign/wrong'] = 1 - sum(resign_checks) / len(resign_checks)
        log['resign/on'] = int(resign_on.value)
      if s.steps:
        log.update({'loss/total': (s.policy + s.value) / steps,
                    'loss/policy': s.policy / steps, 'loss/value': s.value / steps,
                    'loss/policy_kl': (s.policy - s.entropy) / steps,
                    'steps_per_s': s.steps / secs, 'lr': opt.param_groups[0]['lr'],
                    'net_version': version.value})
      wandb.log(log)
    s.reset()

  run_started = last_report = time.time()
  try:
    while True:
      # Take whatever the actors have finished without waiting for more.
      while True:
        try:
          take(games_q.get_nowait())
        except queue.Empty:
          break
      while True:
        try:
          result = eval_q.get_nowait()
        except queue.Empty:
          break
        if result['kind'] == 'stockfish':
          print(f'elo vs stockfish@{result["level"]} (net v{result["version"]}, '
                f'{args.eval_sims} sims): {rl_eval.describe(result)}  '
                f'=> about {result["estimate"]:.0f} Elo  [{result["seconds"]:.0f}s]', flush=True)
          if args.wandb:
            wandb.log({'step': step, 'eval/stockfish_elo': result['estimate'],
                       'eval/stockfish_level': result['level'],
                       'eval/stockfish_score': result['score']})
          continue
        print(f'eval vs {args.eval_opponent} (net v{result["version"]}, '
              f'{args.eval_sims} sims): {rl_eval.describe(result)}  '
              f'[{result["seconds"]:.0f}s]', flush=True)
        if args.wandb:
          log = {'step': step, 'eval/score': result['score'], 'eval/wins': result['wins'],
                 'eval/draws': result['draws'], 'eval/losses': result['losses']}
          if result['elo'] is not None:
            log['eval/elo'] = result['elo']
            log['eval/elo_margin'] = result['elo_margin']
          wandb.log(log)

      if time.time() - last_report >= args.log_every:
        report()
        last_report = time.time()
        dead = [p.name for p in actors + helpers if not p.is_alive()]
        if dead:
          raise RuntimeError(f'{", ".join(dead)} died; see the traceback above')

      if args.max_steps and steps_run >= args.max_steps:
        break
      if args.max_minutes and time.time() - run_started >= args.max_minutes * 60:
        break

      if (replay.size < args.min_buffer
          or steps_run * args.batch_size >= fresh * args.replay_ratio):
        # Ahead of the actors: wait for another game rather than overfit
        # the ones already in the buffer.
        try:
          take(games_q.get(timeout=0.5))
        except queue.Empty:
          pass
        continue

      batch = to_device(replay.sample(args.batch_size, rng), device)
      with autocast:
        policy, value, entropy = losses(net, batch)
      opt.zero_grad(set_to_none=True)
      (policy + value).backward()
      torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
      opt.step()
      step += 1
      steps_run += 1
      step_now.value = step
      stats.add_step(policy.item(), value.item(), entropy.item())

      if step % args.publish_every == 0:
        publish(net, shared, lock, version)
      if step % args.save_every == 0:
        save()
      if args.snapshot_every and step % args.snapshot_every == 0:
        os.makedirs(snapshot_dir, exist_ok=True)
        save_net(net, os.path.join(snapshot_dir, f'step_{step:07d}.pt'))
  except KeyboardInterrupt:
    print('\nstopping...', flush=True)
  finally:
    stop.set()
    save(final=True)
    for proc in actors + helpers:
      proc.join(timeout=10)
      if proc.is_alive():
        proc.terminate()

if __name__ == '__main__':
  main()
