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

def make_backend(name, net, device, c_puct=1.75, fpu_reduction=0.25, seed=None, threads=0):
  """name: 'auto', 'rust' or 'python'. threads is for Rust only (0 = all cores)."""
  if name == 'auto':
    name = 'rust' if rust_available() else 'python'
  if name == 'rust':
    if nighty_rs is None:
      raise RuntimeError(BUILD_HINT)
    return RustBackend(net, device, c_puct, fpu_reduction, seed, threads)
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

  def tree(self, board):
    return PythonTree(self, board)

  def step(self, trees, leaves_per_tree=1):
    return self.mcts.step([t.tree for t in trees], leaves_per_tree)

  def run(self, trees, leaves_per_tree=1):
    while self.step(trees, leaves_per_tree):
      pass

  def finished(self, trees):
    """Indices into `trees` of the searches that have reached their target."""
    return [n for n, t in enumerate(trees) if t.done()]

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
    pass

# --- Rust ---------------------------------------------------------------------

class RustBackend:
  """One searcher for every tree of the process, one forward pass per step.
  (Splitting the trees over two searchers so one batch could be on the GPU
  while the other was walked was measured slower: two thread pools fight,
  and the GPU cares more about batch size than about overlap.)"""

  name = 'rust'

  def __init__(self, net, device, c_puct, fpu_reduction, seed, threads):
    self.evaluate = Evaluator(net, device)
    self.searcher = nighty_rs.Searcher(c_puct, fpu_reduction, seed, threads)

  def tree(self, board):
    return RustTree(self, self.searcher, board)

  def step(self, trees, leaves_per_tree=1):
    """One batch over every tree this backend holds (the `trees` argument is
    only there to match PythonBackend: the searcher already knows them all).
    Returns False, having done nothing, once none wants more simulations."""
    if not self.searcher.busy():
      return False
    tokens, halfmove, indices = self.searcher.collect(leaves_per_tree)
    if len(halfmove):
      logits, values = self.evaluate(tokens, halfmove, indices)
      self.searcher.apply(np.ascontiguousarray(logits, dtype=np.float32),
                          np.ascontiguousarray(values, dtype=np.float32))
    return True

  def run(self, trees, leaves_per_tree=1):
    while self.step(trees, leaves_per_tree):
      pass

  def finished(self, trees):
    # One call for all of them: asking each of thousands of trees in turn was
    # a measurable slice of every step.
    done = set(self.searcher.done())
    return [n for n, t in enumerate(trees) if t.id in done]

  def set_c_puct(self, value):
    self.searcher.set_c_puct(value)

  @property
  def sims(self):
    return self.searcher.total_sims()

  @property
  def evals(self):
    return self.searcher.total_evals()

class RustTree:
  def __init__(self, backend, searcher, board):
    self.backend = backend
    self.searcher = searcher
    self.board = board
    self.id = self.searcher.new_tree(board.root().fen(), [m.uci() for m in board.move_stack])
    self.target = 0
    self._moves = None

  def reset_search(self, target, noise=None):
    alpha, fraction = noise if noise else (0.0, 0.0)
    self.searcher.reset_search(self.id, target, alpha, fraction)
    self.target = target

  @property
  def sims(self):
    return self.searcher.sims(self.id)

  def done(self):
    return self.searcher.sims(self.id) >= self.target

  def expanded(self):
    return self.searcher.expanded(self.id)

  def _root(self):
    triples, N, Q, P = self.searcher.root(self.id)
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
    return self.searcher.best_child(self.id, random_ties)

  def sample(self, temperature):
    return self.searcher.sample_child(self.id, temperature)

  def advance(self, move):
    self.searcher.advance(self.id, move.uci())
    self.board.push(move)
    self._moves = None

  def pv(self):
    return [to_move(t) for t in self.searcher.pv(self.id)]

  def mate_in(self):
    return self.searcher.mate_in(self.id)

  def encoding(self):
    tokens, halfmove, indices = self.searcher.root_encoding(self.id)
    return tokens, halfmove, indices

  def game_over(self):
    return self.searcher.game_over(self.id)

  def ply(self):
    return self.searcher.ply(self.id)

  def close(self):
    if self.id is not None:
      self.searcher.drop_tree(self.id)
      self.id = None
