"""PUCT tree search (AlphaZero-style) over RLNet, batched across trees.

The search itself is plain Python, so the net is only worth calling on a batch.
There are two ways of getting one, and step() does both:

  * many trees in lockstep -- self-play runs dozens of games per process, and
    each step() takes a leaf from every one of them, so a single forward pass
    serves them all;
  * several leaves per tree -- the UCI engine only has one tree, so it
    descends it several times per step with a virtual loss on the way down,
    which steers each descent away from the paths already waiting on the net.

Values are always in [-1, 1]. A node's value is for the side to move at that
node; an edge's W is for the side choosing that edge, which is the same side,
so every node picks the child with the highest Q + U and backup flips the sign
once per ply.
"""

from collections import namedtuple
import math

import chess
import numpy as np

from rl_model import encode, move_index

class Node:
  __slots__ = ('moves', 'P', 'N', 'W', 'children', 'visits', 'value_sum',
               'terminal', 'pending')

  def __init__(self):
    self.moves = None      # legal moves, set on expansion
    self.P = None          # prior per move
    self.N = None          # visits per move
    self.W = None          # summed value per move
    self.children = None
    self.visits = 0        # this node's own evaluation + sum(N)
    self.value_sum = 0.0
    self.terminal = None   # exact value, if the game is over here
    self.pending = False   # sent to the net and not back yet

Leaf = namedtuple('Leaf', 'tree path node legal indices tokens halfmove')

def terminal_value(board, legal):
  """Exact value for the side to move if the game is over here, else None.

  Inside the tree one repetition already counts as a draw, not three: whoever
  could have avoided it chose not to, and that is what lets the search see a
  perpetual coming."""
  if not legal:
    return -1.0 if board.is_check() else 0.0
  if board.halfmove_clock >= 100 or board.is_insufficient_material():
    return 0.0
  # A repetition needs four reversible plies, so below that there is nothing
  # for is_repetition's walk down the move stack to find.
  if board.halfmove_clock >= 4 and board.is_repetition(2):
    return 0.0
  return None

def game_over(board):
  """(White's score, reason) if a real game has ended, else None. Every draw
  is claimed as soon as it can be."""
  if board.is_checkmate():
    return (-1 if board.turn == chess.WHITE else 1), 'mate'
  if board.is_stalemate():
    return 0, 'stalemate'
  if board.is_insufficient_material():
    return 0, 'material'
  if board.halfmove_clock >= 100:
    return 0, 'fifty'
  if board.is_repetition(3):
    return 0, 'repetition'
  return None

def forced_mate(board):
  """Mate in one, if there is one. Any search finds it eventually, but only a
  check can be mate, so looking costs a push per checking move."""
  for move in board.legal_moves:
    if not board.gives_check(move):
      continue
    board.push(move)
    try:
      if board.is_checkmate():
        return move
    finally:
      board.pop()
  return None

class Tree:
  """A board and the search tree grown from it.

  The board is borrowed: the search pushes and pops moves on it but always
  leaves it where it found it, and advance() is the only thing that moves it on
  for good. It keeps its whole move stack, which is how the search sees
  repetitions of positions from earlier in the game."""

  def __init__(self, board, root=None):
    self.board = board
    self.root = root or Node()
    self.target = 0         # simulations wanted for this move
    self.sims = 0           # simulations done for this move
    self.noise = None       # (alpha, fraction) of Dirichlet noise at the root
    self.root_prior = None  # the root's priors with that noise mixed in

  def reset_search(self, target, noise=None):
    self.target = target
    self.sims = 0
    self.noise = noise
    self.root_prior = None

  def advance(self, move):
    """Play a move, keeping whatever was already searched below it."""
    child = None
    if self.root.moves is not None:
      for i, m in enumerate(self.root.moves):
        if m == move:
          child = self.root.children[i]
          break
    self.board.push(move)
    # A child the tree scored as a repetition draw is a live position once it
    # is the real one, so it has to be searched from scratch.
    keep = child is not None and not child.pending and child.terminal is None
    self.root = child if keep else Node()
    self.root_prior = None
    self.sims = 0

def best_child(node, rng=None):
  """The most visited move, ties broken by value. Before any visits (a search
  of one simulation only expands the root) that is the net's top prior.

  With an rng, whatever is still tied after that is picked at random. Self-play
  needs this: an untrained net scores every move the same, and always taking
  the first of a tie walks the same piece back and forth until the game is a
  threefold repetition -- which is how nearly every early game used to end."""
  N = node.N
  if N.max() <= 0:
    top = np.flatnonzero(node.P == node.P.max())
  else:
    top = np.flatnonzero(N == N.max())
    if len(top) > 1:
      q = node.W[top] / N[top]
      top = top[q == q.max()]
  if len(top) == 1 or rng is None:
    return int(top[0])
  return int(rng.choice(top))

def sample_child(node, temperature, rng):
  """A move drawn in proportion to visits ** (1 / temperature); 0 is the most
  visited, ties broken at random."""
  if temperature <= 0:
    return best_child(node, rng)
  weights = node.N.astype(np.float64) ** (1.0 / temperature)
  total = weights.sum()
  if total <= 0:
    return best_child(node)
  return int(rng.choice(len(weights), p=weights / total))

def child_q(node, i):
  """Value of move i for the side choosing it."""
  return float(node.W[i] / node.N[i]) if node.N[i] else 0.0

def principal_variation(node, limit=24):
  return pv_and_end(node, limit)[0]

def pv_and_end(node, limit=24):
  """The line of most-visited moves, and the node it ends on."""
  line = []
  while (node is not None and node.N is not None and len(line) < limit
         and node.N.max() > 0):
    i = best_child(node)
    line.append(node.moves[i])
    node = node.children[i]
  return line, node

def mate_in(node):
  """Moves to mate if the principal variation ends in one, signed for the
  side to move at `node` (negative: it is the one getting mated), else None."""
  line, end = pv_and_end(node, limit=64)
  if end is None or end.terminal != -1.0:
    return None
  plies = len(line)
  return (plies + 1) // 2 if plies % 2 else -(plies // 2)

class MCTS:
  def __init__(self, evaluate, c_puct=1.75, fpu_reduction=0.25, rng=None):
    self.evaluate = evaluate
    self.c_puct = c_puct
    self.fpu_reduction = fpu_reduction
    self.rng = rng if rng is not None else np.random.default_rng()
    self.sims = 0   # simulations run, for throughput stats
    self.evals = 0  # ...of which reached a new leaf and went to the net

  def run(self, trees, leaves_per_tree=1):
    """Search every tree until it has done its target number of simulations."""
    while self.step(trees, leaves_per_tree):
      pass

  def step(self, trees, leaves_per_tree=1):
    """One batch: descend each unfinished tree up to leaves_per_tree times,
    score every new leaf in a single forward pass, back them all up. Returns
    False, having done nothing, once no tree wants more simulations."""
    batch = []
    busy = False
    for tree in trees:
      want = min(leaves_per_tree, tree.target - tree.sims)
      if want <= 0:
        continue
      busy = True
      if (tree.noise is not None and tree.root_prior is None
          and tree.root.P is not None):
        self.add_noise(tree)
      for _ in range(want):
        leaf = self.descend(tree)
        if leaf is None:
          # Ran into a leaf that is already waiting on the net. Nothing new
          # to learn from this tree until that comes back.
          break
        tree.sims += 1
        self.sims += 1
        if leaf is not True:
          batch.append(leaf)
    if batch:
      self.expand(batch)
    return busy

  def descend(self, tree):
    """Walk from the root to a leaf. A game-over leaf is backed up on the spot
    and gives True; a new one comes back as a Leaf for the net; None means the
    walk collided with a leaf already in this batch and was undone."""
    node = tree.root
    board = tree.board
    path = []
    try:
      while True:
        if node.terminal is not None:
          self.backup(path, node, node.terminal)
          return True
        if node.P is None:
          if node.pending:
            self.undo(path)
            return None
          legal = list(board.legal_moves)
          # The root is the real game, which only a lack of moves ends: a
          # position repeated once there is still a position to play.
          value = terminal_value(board, legal) if path or not legal else None
          if value is not None:
            node.terminal = value
            self.backup(path, node, value)
            return True
          node.pending = True
          tokens, halfmove = encode(board)
          turn = board.turn
          return Leaf(tree, path, node, legal,
                      [move_index(m, turn) for m in legal], tokens, halfmove)
        i = self.select(tree, node)
        # Virtual loss: count the visit now and score it as a loss until the
        # real value comes back, so other descents this batch look elsewhere.
        node.N[i] += 1
        node.W[i] -= 1
        node.visits += 1
        path.append((node, i))
        board.push(node.moves[i])
        child = node.children[i]
        if child is None:
          child = node.children[i] = Node()
        node = child
    finally:
      for _ in path:
        board.pop()

  def select(self, tree, node):
    prior = node.P
    if node is tree.root and tree.root_prior is not None:
      prior = tree.root_prior
    N = node.N
    # Unvisited moves are assumed a little worse than this node is doing so
    # far (first-play urgency), so the search deepens good lines instead of
    # trying every move once.
    fpu = node.value_sum / node.visits - self.fpu_reduction
    q = np.where(N > 0, node.W / np.maximum(N, 1.0), fpu)
    u = (self.c_puct * math.sqrt(node.visits)) * prior / (1.0 + N)
    return int(np.argmax(q + u))

  def expand(self, batch):
    tokens = np.array([leaf.tokens for leaf in batch], dtype=np.int64)
    halfmove = np.array([leaf.halfmove for leaf in batch], dtype=np.float32)
    indices = np.zeros((len(batch), max(len(leaf.indices) for leaf in batch)), np.int64)
    for row, leaf in zip(indices, batch):
      row[:len(leaf.indices)] = leaf.indices
    logits, values = self.evaluate(tokens, halfmove, indices)
    self.evals += len(batch)
    for leaf, row, value in zip(batch, logits, values):
      n = len(leaf.legal)
      p = row[:n]
      p = np.exp(p - p.max())
      node = leaf.node
      node.moves = leaf.legal
      node.P = (p / p.sum()).astype(np.float32)
      node.N = np.zeros(n, np.float32)
      node.W = np.zeros(n, np.float32)
      node.children = [None] * n
      node.pending = False
      self.backup(leaf.path, node, float(value))

  def backup(self, path, leaf, value):
    leaf.visits += 1
    leaf.value_sum += value
    for node, i in reversed(path):
      value = -value
      node.W[i] += 1.0 + value  # the 1 takes the virtual loss back out
      node.value_sum += value

  def undo(self, path):
    for node, i in path:
      node.N[i] -= 1
      node.W[i] += 1
      node.visits -= 1

  def add_noise(self, tree):
    alpha, fraction = tree.noise
    prior = tree.root.P
    noise = self.rng.dirichlet([alpha] * len(prior))
    tree.root_prior = ((1 - fraction) * prior + fraction * noise).astype(np.float32)
