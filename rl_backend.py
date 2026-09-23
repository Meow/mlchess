"""One interface over the two tree searches: rl_mcts.py (pure Python) and
nighty_rs (the same search in Rust, built from rust/). train_rl.py,
rl_eval.py and run_rl.py only talk to this, so they neither know nor care
which one is underneath.

  backend = make_backend('auto')          # Rust if it is built, else Python
  tree = backend.tree(board)              # a search tree for a position
  tree.reset_search(200)
  while backend.step([tree]): pass        # or step many trees at once
  move = tree.moves()[tree.best()]

The Rust search is ~5x faster per core than the Python one and, being free
of the GIL, runs every tree of a process on all cores at once, so one process
with a thousand games and its inference on the GPU replaces sixteen Python
actors. Both keep a python-chess Board alongside the tree: it is what the
game logic (results, encodings for training rows, adjudication) reads, and
it only changes once per move, not per simulation.
"""

import chess
import numpy as np
import torch

import rl_mcts
from rl_model import Evaluator, encode, move_index

try:
  import nighty_rs
  if not hasattr(nighty_rs, 'Searcher'):
    # Something else answered to the name (e.g. an empty directory on the
    # path); the extension is not built for this interpreter.
    nighty_rs = None
except ImportError:
  nighty_rs = None

BUILD_HINT = ('the Rust search is not built for this Python: '
              'run `python3 -m pip install ./rust` with the same python3 (needs a Rust toolchain)')

def rust_available():
  return nighty_rs is not None

def make_backend(name, net, device, c_puct=1.75, fpu_reduction=0.25, seed=None, threads=0, lanes=1):
  """name: 'auto', 'rust' or 'python'. threads and lanes are for Rust only:
  threads 0 = all cores; lanes is how many searchers the trees are spread
  over, each with a net call in flight (see RustBackend)."""
  if name == 'auto':
    name = 'rust' if rust_available() else 'python'
  if name == 'rust':
    if nighty_rs is None:
      raise RuntimeError(BUILD_HINT)
    return RustBackend(net, device, c_puct, fpu_reduction, seed, threads, lanes)
  if name == 'python':
    return PythonBackend(net, device, c_puct, fpu_reduction, seed)
  raise ValueError(f'unknown backend {name!r}')

def to_move(triple):
  src, dst, promotion = triple
  return chess.Move(src, dst, promotion=promotion or None)

# --- Python -------------------------------------------------------------------

class PythonBackend:
  name = 'python'

  def __init__(self, net, device, c_puct, fpu_reduction, seed):
    self.rng = np.random.default_rng(seed)
    self.mcts = rl_mcts.MCTS(Evaluator(net, device), c_puct, fpu_reduction, self.rng)
    self.live = []  # trees not yet closed, for step(None)

  def tree(self, board):
    tree = PythonTree(self, board)
    self.live.append(tree)
    return tree

  def step(self, trees=None, leaves_per_tree=1):
    if trees is None:
      trees = self.live
    return self.mcts.step([t.tree for t in trees], leaves_per_tree)

  def run(self, trees, leaves_per_tree=1):
    while self.step(trees, leaves_per_tree):
      pass

  def finished(self, trees):
    """Indices into `trees` of the searches that have reached their target."""
    return [n for n, t in enumerate(trees) if t.done()]

  def done_ids(self):
    """Ids of finished searches, or None when trees must be asked one by one."""
    return None

  def set_c_puct(self, value):
    self.mcts.c_puct = value

  @property
  def sims(self):
    return self.mcts.sims

  @property
  def evals(self):
    return self.mcts.evals

class PythonTree:
  def __init__(self, backend, board):
    self.backend = backend
    self.board = board
    self.tree = rl_mcts.Tree(board)

  def reset_search(self, target, noise=None):
    self.tree.reset_search(target, noise)

  @property
  def sims(self):
    return self.tree.sims

  @property
  def target(self):
    return self.tree.target

  def done(self):
    return self.tree.sims >= self.tree.target

  @property
  def id(self):
    return id(self)

  def expanded(self):
    return self.tree.root.P is not None

  def moves(self):
    return self.tree.root.moves

  def visits(self):
    return self.tree.root.N

  def q(self, i):
    return rl_mcts.child_q(self.tree.root, i)

  def best(self, random_ties=False):
    return rl_mcts.best_child(self.tree.root, self.backend.rng if random_ties else None)

  def sample(self, temperature):
    return rl_mcts.sample_child(self.tree.root, temperature, self.backend.rng)

  def advance(self, move):
    self.tree.advance(move)

  def pv(self):
    return rl_mcts.principal_variation(self.tree.root)

  def mate_in(self):
    return rl_mcts.mate_in(self.tree.root)

  def encoding(self):
    """The root as the net sees it: (tokens, halfmove, indices of moves())."""
    tokens, halfmove = encode(self.board)
    turn = self.board.turn
    return tokens, halfmove, np.array([move_index(m, turn) for m in self.moves()], np.int64)

  def game_over(self):
    return rl_mcts.game_over(self.board)

  def ply(self):
    return self.board.ply()

  def close(self):
    try:
      self.backend.live.remove(self)
    except ValueError:
      pass

# --- Rust ---------------------------------------------------------------------

class RustBackend:
  """The process's trees, spread over `lanes` searchers, with one net call in
  flight per lane. A step is a chain -- walk the trees, run the net, back up
  -- and with one searcher the cores idle while the GPU works and vice versa.
  With two lanes the walk and back-up of one overlap the forward pass of the
  other: submit() queues the net's work and only fetch() waits for it, so
  each lane's batch is collected, submitted, and only picked up on the next
  step, after the other lane has been walked. All lanes share rayon's global
  pool, and are walked one after another, so they never fight for cores;
  what a second lane costs is half the batch per GPU call, which a CUDA
  card does not mind and MPS (a ~1 ms floor per call) does -- hence lanes=1
  unless the actor is on CUDA.

  The invariant that makes this safe: a tree with leaves out at the net is
  not done (done() checks) and is not touched -- Python settles the lane
  (applies its batch) before advancing, resetting or dropping such a tree,
  and Rust refuses if it did not."""

  name = 'rust'

  def __init__(self, net, device, c_puct, fpu_reduction, seed, threads, lanes=1):
    self.evaluate = Evaluator(net, device)
    # Every searcher in the process shares rayon's global pool; the first one
    # to ask sizes it (0 leaves it at every core).
    if threads:
      nighty_rs.set_threads(threads)
    self.lanes = [Lane(nighty_rs.Searcher(c_puct, fpu_reduction, None if seed is None else seed + i, 0))
                  for i in range(max(1, lanes))]
    self.next_lane = 0

  def tree(self, board):
    # Round robin keeps the lanes the same size, as games start and end.
    lane = self.lanes[self.next_lane]
    self.next_lane = (self.next_lane + 1) % len(self.lanes)
    return RustTree(self, lane, board)

  def step(self, trees=None, leaves_per_tree=1):
    """One batch for every lane this backend holds (the `trees` argument is
    only there to match PythonBackend: the searchers already know them all).
    Each lane first takes in the answers to its previous batch, then sends
    the next. Returns False, having done nothing, once nothing is out at
    the net and no tree wants more simulations."""
    worked = False
    for lane in self.lanes:
      if lane.handle is not None:
        self.settle(lane)
        worked = True
      if lane.searcher.busy():
        tokens, halfmove, indices = lane.searcher.collect(leaves_per_tree)
        if len(halfmove):
          lane.handle = self.evaluate.submit(tokens, halfmove, indices)
        worked = True
    return worked

  def settle(self, lane):
    """Apply the lane's batch in flight, if any (waits for the GPU)."""
    if lane.handle is None:
      return
    logits, values = self.evaluate.fetch(lane.handle)
    lane.handle = None
    lane.searcher.apply(np.ascontiguousarray(logits, dtype=np.float32),
                        np.ascontiguousarray(values, dtype=np.float32))

  def run(self, trees, leaves_per_tree=1):
    while self.step(trees, leaves_per_tree):
      pass

  def finished(self, trees):
    # One call for all of them: asking each of thousands of trees in turn was
    # a measurable slice of every step.
    done = self.done_ids()
    return [n for n, t in enumerate(trees) if t.id in done]

  def done_ids(self):
    return {(lane.index, r) for lane in self.lanes for r in lane.searcher.done()}

  def set_c_puct(self, value):
    for lane in self.lanes:
      lane.searcher.set_c_puct(value)

  @property
  def sims(self):
    return sum(lane.searcher.total_sims() for lane in self.lanes)

  @property
  def evals(self):
    return sum(lane.searcher.total_evals() for lane in self.lanes)

class Lane:
  """A searcher and the handle of its batch at the net, if one is out."""
  __slots__ = ('searcher', 'handle', 'index')
  count = 0

  def __init__(self, searcher):
    self.searcher = searcher
    self.handle = None
    self.index = Lane.count   # distinct across the process, so tree ids are too
    Lane.count += 1

class RustTree:
  def __init__(self, backend, lane, board):
    self.backend = backend
    self.lane = lane
    self.searcher = lane.searcher
    self.board = board
    # Only the moves since the last capture or pawn move go to Rust: nothing
    # before them can take part in a repetition, and walking a whole game's
    # move stack for every new tree adds up when trees are made per move.
    k = min(board.halfmove_clock, len(board.move_stack))
    if k:
      recent = board.copy(stack=k)
      start = recent.root().fen()
      moves = [m.uci() for m in recent.move_stack]
    else:
      start, moves = board.fen(), []
    self.rid = self.searcher.new_tree(start, moves)
    self.id = (lane.index, self.rid)
    self.target = 0
    self._moves = None

  def _settled(self):
    """Take in this tree's leaves at the net, if any, before reading or
    changing its root. In self-play a tree is only touched once done, when
    nothing of it is out; this is for a search stopped part way (the engine
    on `stop`, or a clock)."""
    if self.lane.handle is not None and self.searcher.in_flight(self.rid):
      self.backend.settle(self.lane)

  def reset_search(self, target, noise=None):
    self._settled()
    alpha, fraction = noise if noise else (0.0, 0.0)
    self.searcher.reset_search(self.rid, target, alpha, fraction)
    self.target = target

  @property
  def sims(self):
    return self.searcher.sims(self.rid)

  def done(self):
    """Reached its target, with every leaf's answer backed up."""
    return self.searcher.sims(self.rid) >= self.target and not self.searcher.in_flight(self.rid)

  def expanded(self):
    return self.searcher.expanded(self.rid)

  def _root(self):
    self._settled()
    triples, N, Q, P = self.searcher.root(self.rid)
    self._moves = [to_move(t) for t in triples]
    return N, Q, P

  def moves(self):
    if self._moves is None:
      self._root()
    return self._moves

  def visits(self):
    return self._root()[0]

  def q(self, i):
    return float(self._root()[1][i])

  def best(self, random_ties=False):
    self._settled()
    return self.searcher.best_child(self.rid, random_ties)

  def sample(self, temperature):
    self._settled()
    return self.searcher.sample_child(self.rid, temperature)

  def advance(self, move):
    self._settled()
    self.searcher.advance(self.rid, move.uci())
    self.board.push(move)
    self._moves = None

  def pv(self):
    self._settled()
    return [to_move(t) for t in self.searcher.pv(self.rid)]

  def mate_in(self):
    self._settled()
    return self.searcher.mate_in(self.rid)

  def encoding(self):
    tokens, halfmove, indices = self.searcher.root_encoding(self.rid)
    return tokens, halfmove, indices

  def game_over(self):
    return self.searcher.game_over(self.rid)

  def ply(self):
    return self.searcher.ply(self.rid)

  def close(self):
    if self.rid is not None:
      self._settled()
      self.searcher.drop_tree(self.rid)
      self.rid = None
