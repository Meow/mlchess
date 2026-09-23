"""UCI engine for the reinforcement-learning model (nighty_rl.pt).

Where run.py has two imitation nets propose moves and piece-square tables judge
them, this has one self-taught net do both, the way AlphaZero plays: PUCT
search over its policy, scored by its value head, no hand-written evaluation.
It speaks the same UCI as run.py, so lichess-bot can run either -- point it at
run_rl.sh instead of run.sh.

  python3 run_rl.py [--weights nighty_rl.pt] [--device cpu]

Searching is not all-or-nothing the way iterative deepening is, so it uses its
whole time budget and simply stops; the move it plays is the most visited one.
`go nodes N` caps the search by simulations instead of time. `go depth` means
nothing to a tree search and is ignored.
"""

import argparse
import math
import os
import queue
import re
import sys
import threading
import time

import chess
import torch

from rl_backend import make_backend
from rl_mcts import forced_mate
from rl_model import load_net, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))

move_overhead = 0.1
default_budget = 10.0
batch = 32         # leaves per forward pass, gathered with virtual loss
c_puct = 1.75
fpu_reduction = 0.25

def send(line):
  # stdout is a pipe under lichess-bot, so it is block buffered unless we flush.
  sys.stdout.write(line + '\n')
  sys.stdout.flush()

def time_budget(board, params):
  """Seconds to spend on this move. Same arithmetic as run.py's."""
  if 'movetime' in params:
    return max(0.02, params['movetime'] / 1000.0 - move_overhead)
  clock = 'wtime' if board.turn == chess.WHITE else 'btime'
  bonus = 'winc' if board.turn == chess.WHITE else 'binc'
  if clock not in params:
    return None if 'nodes' in params else default_budget
  left = params[clock] / 1000.0
  inc = params.get(bonus, 0) / 1000.0
  togo = params.get('movestogo', 0)
  share = max(1, min(togo, 30)) if togo else 28
  budget = left / share + inc * 0.75
  return max(0.02, min(budget, left * 0.4) - move_overhead)

def score_string(tree, i):
  mate = tree.mate_in()
  if mate is not None:
    return f'mate {mate}'
  # Expected score to centipawns, with Leela's old conversion so the numbers
  # look like what a GUI expects rather than a probability.
  q = max(-0.999, min(0.999, tree.q(i)))
  return f'cp {round(111.714640912 * math.tan(1.5620688421 * q))}'

class Input:
  """stdin on its own thread, so `stop` can land in the middle of a search.

  Commands still run in order off the queue. The counters are how a search
  knows a stop is waiting behind it in the queue: every stop read bumps
  `seen`, every one taken off the queue bumps `handled`. `quit` is counted
  separately and only cuts an infinite search short: a script that pipes
  `go movetime 3000` followed by `quit` wants the search, not the exit --
  and the exit comes right after anyway."""

  def __init__(self):
    self.lines = queue.Queue()
    self.seen = {'stop': 0, 'quit': 0}
    self.handled = {'stop': 0, 'quit': 0}
    threading.Thread(target=self.read, daemon=True).start()

  def read(self):
    for line in sys.stdin:
      line = line.strip()
      if line in self.seen:
        self.seen[line] += 1
      self.lines.put(line)
    self.seen['quit'] += 1
    self.lines.put('quit')

  def get(self):
    line = self.lines.get()
    if line in self.handled:
      self.handled[line] += 1
    return line

  def stop_requested(self, infinite=False):
    if self.seen['stop'] > self.handled['stop']:
      return True
    return infinite and self.seen['quit'] > self.handled['quit']

class Engine:
  def __init__(self, weights, device, backend='auto', threads=0):
    self.device = device
    self.net = load_net(weights, device).eval()
    self.backend = make_backend(backend, self.net, device, c_puct, fpu_reduction, threads=threads)
    self.board = chess.Board()
    self.tree = None
    self.tree_start = None

  def new_game(self):
    self.board.reset()
    self.forget()

  def forget(self):
    if self.tree is not None:
      self.tree.close()
    self.tree = None

  def tree_for(self, board):
    """The previous search's tree, walked forward to this position if it leads
    here, so the visits already spent on it are not thrown away."""
    start = board.root().fen()
    old = self.tree
    if old is not None and self.tree_start == start:
      had = old.board.move_stack
      now = board.move_stack
      if len(now) >= len(had) and now[:len(had)] == had:
        for move in now[len(had):]:
          old.advance(move)
        return old
    self.forget()
    self.tree = self.backend.tree(board.copy())
    self.tree_start = start
    return self.tree

  def search(self, params, infinite, stop_requested):
    board = self.board
    legal = list(board.legal_moves)
    if not legal:
      return None
    mate = forced_mate(board)
    if mate is not None:
      send(f'info depth 1 score mate 1 pv {mate.uci()}')
      return mate

    started = time.time()
    budget = None if infinite else time_budget(board, params)
    deadline = started + budget if budget else None
    max_nodes = params.get('nodes', 0)
    tree = self.tree_for(board)
    tree.reset_search(max_nodes or 10 ** 9)
    if len(legal) == 1 and not infinite:
      return legal[0]

    last_info = started
    while True:
      self.backend.step([tree], batch)
      now = time.time()
      if stop_requested(infinite) or tree.done():
        break
      if deadline is not None:
        if now >= deadline or self.settled(tree, started, now, deadline):
          break
      if now - last_info >= 1.0:
        self.info(tree, started)
        last_info = now
    self.info(tree, started)
    return tree.moves()[tree.best()]

  @staticmethod
  def settled(tree, started, now, deadline):
    """True once the runner-up could not catch the best move even if it got
    every simulation left in the budget -- the rest would be wasted clock."""
    if tree.sims < 100 or not tree.expanded():
      return False
    N = tree.visits()
    if len(N) < 2:
      return False
    first, second = sorted(N)[-2:][::-1]
    remaining = tree.sims / max(now - started, 1e-6) * (deadline - now)
    return first - second > remaining

  def info(self, tree, started):
    if not tree.expanded() or tree.visits().max() <= 0:
      return
    i = tree.best()
    pv = tree.pv()
    elapsed = max(time.time() - started, 1e-6)
    send(f'info depth {len(pv)} nodes {tree.sims} nps {int(tree.sims / elapsed)} '
         f'score {score_string(tree, i)} time {int(elapsed * 1000)} '
         f'pv {" ".join(m.uci() for m in pv)}')

def set_position(board, cmd):
  args = cmd.split()
  if len(args) < 2:
    return
  i = 1
  if args[1] == 'startpos':
    board.reset()
    i = 2
  elif args[1] == 'fen':
    fields = []
    i = 2
    while i < len(args) and args[i] != 'moves':
      fields.append(args[i])
      i += 1
    try:
      board.set_fen(' '.join(fields))
    except ValueError as e:
      send(f'info string bad fen: {e}')
      return
  else:
    return
  if i < len(args) and args[i] == 'moves':
    for uci in args[i + 1:]:
      try:
        board.push_uci(uci)
      except ValueError as e:
        send(f'info string bad move {uci}: {e}')
        return

def parse_go(args):
  params = {}
  known = ('wtime', 'btime', 'winc', 'binc', 'movestogo', 'movetime', 'nodes')
  i = 1
  while i < len(args):
    if args[i] in known and i + 1 < len(args):
      try:
        params[args[i]] = int(args[i + 1])
      except ValueError:
        pass
      i += 2
    else:
      i += 1
  return params

def set_option(engine, cmd):
  global move_overhead, batch, c_puct
  m = re.match(r'^setoption\s+name\s+(.+?)\s+value\s+(.+)$', cmd, re.IGNORECASE)
  if not m:
    return
  name = m.group(1).strip().lower()
  value = m.group(2).strip()
  try:
    if name == 'move overhead':
      move_overhead = max(0.0, int(value) / 1000.0)
    elif name == 'batch':
      batch = max(1, int(value))
    elif name == 'cpuct':
      c_puct = max(0.01, int(value) / 100.0)
      engine.backend.set_c_puct(c_puct)
    elif name == 'threads':
      torch.set_num_threads(max(1, int(value)))
    else:
      return
  except ValueError:
    return
  send(f'info string {name} set to {value}')

def run(engine):
  stdin = Input()
  while True:
    cmd = stdin.get()
    if not cmd:
      continue
    if cmd == 'uci':
      send('id name NightyBot RL')
      send('id author Nighty')
      send(f'option name Batch type spin default {batch} min 1 max 256')
      send(f'option name CPuct type spin default {int(c_puct * 100)} min 1 max 1000')
      send(f'option name Threads type spin default {torch.get_num_threads()} min 1 max 64')
      send(f'option name Move Overhead type spin default {int(move_overhead * 1000)} min 0 max 5000')
      send('uciok')
    elif cmd == 'isready':
      send('readyok')
    elif cmd == 'ucinewgame':
      engine.new_game()
    elif cmd.startswith('position'):
      set_position(engine.board, cmd)
    elif cmd.startswith('go'):
      args = cmd.split()
      move = engine.search(parse_go(args), 'infinite' in args, stdin.stop_requested)
      if move is None:
        send('info string no legal moves')
        send('bestmove (none)')
      else:
        # UCI says an infinite search reports only once it is told to stop;
        # by the time search() returns, it has been.
        send(f'bestmove {move.uci()}')
    elif cmd == 'stop':
      pass  # nothing running; a stop during a search ends it from inside
    elif cmd.startswith('setoption'):
      set_option(engine, cmd)
    elif cmd == 'quit':
      return
    else:
      send('info string Unrecognized Command')

def main():
  p = argparse.ArgumentParser(description='UCI engine for nighty_rl.pt')
  p.add_argument('--weights', default=os.environ.get('NIGHTY_RL_WEIGHTS',
                                                     os.path.join(HERE, 'nighty_rl.pt')))
  p.add_argument('--device', default='cpu',
                 help="'cpu' (default), 'mps', 'cuda', or 'auto' for the best available")
  p.add_argument('--threads', type=int, default=1,
                 help='torch threads for inference (Python search), or search threads (Rust)')
  p.add_argument('--backend', default='auto', choices=['auto', 'rust', 'python'],
                 help='the Rust tree search (nighty_rs) if it is built, else the Python one')
  args = p.parse_args()
  torch.set_num_threads(args.threads)
  run(Engine(args.weights, pick_device(args.device), args.backend, args.threads))

if __name__ == '__main__':
  main()
