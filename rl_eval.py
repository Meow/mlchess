"""Pit the RL model against another player and report the score.

  python3 rl_eval.py                                  # nighty_rl.pt vs a 1-ply greedy player
  python3 rl_eval.py --opponent greedy:2 --games 40 --sims 400
  python3 rl_eval.py --opponent classical:3               # the Rust alpha-beta over evaluation.py
  python3 rl_eval.py --opponent nightybot             # vs the imitation engine (run.py), over UCI
  python3 rl_eval.py --opponent stockfish:1500          # Stockfish handicapped to 1500 (UCI_Elo)
  python3 rl_eval.py --opponent uci:stockfish --opponent-time 0.05
  python3 rl_eval.py --opponent rl:rl_snapshots/step_0010000.pt

Games are played in pairs from the same random opening, once with each colour,
so neither side profits from a lucky one. All of the RL side's games are
searched together in one batch, so twenty games cost little more than one.
train_rl.py runs the same match in the background while it trains.
"""

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import time

import chess
import torch

from evaluation import eval_pos
from rl_backend import make_backend
from rl_mcts import forced_mate, game_over
from rl_model import load_net, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))

# evaluation.py wants run.py's encode(): FEN reading order (a8 = 0), with 1-6
# black ".prnbqk" and 7-12 white. Rebuilt here from the board, rather than
# imported, because importing run.py loads its models.
FEN_ALPHABET = '.prnbqkPRNBQK'

def fen_order(board):
  out = [0] * 64
  for sq, piece in board.piece_map().items():
    out[sq ^ 56] = FEN_ALPHABET.index(piece.symbol())
  return out

def static_eval(board, color=chess.WHITE):
  """evaluation.py's score in centipawns, from `color`'s point of view."""
  return eval_pos(fen_order(board), 1 if color == chess.WHITE else 0)

# --- players ------------------------------------------------------------------
#
# A player takes a list of boards and returns a move for each. Boards are
# borrowed: a player may push and pop on them but must hand them back as it
# found them.

class MCTSPlayer:
  def __init__(self, net, device, sims, c_puct=1.75, fpu_reduction=0.25,
               backend='auto', threads=0):
    self.backend = make_backend(backend, net, device, c_puct, fpu_reduction, threads=threads)
    self.sims = sims

  def choose(self, boards):
    moves = [None] * len(boards)
    trees = []
    for n, board in enumerate(boards):
      moves[n] = forced_mate(board)
      if moves[n] is None:
        tree = self.backend.tree(board.copy())
        tree.reset_search(self.sims)
        trees.append((n, tree))
    self.backend.run([tree for _, tree in trees])
    for n, tree in trees:
      moves[n] = tree.moves()[tree.best()]
      tree.close()
    return moves

  def close(self):
    pass

class RandomPlayer:
  def __init__(self, seed=None):
    self.rng = random.Random(seed)

  def choose(self, boards):
    return [self.rng.choice(list(b.legal_moves)) for b in boards]

  def close(self):
    pass

class GreedyPlayer:
  """Alpha-beta to a fixed depth over every legal move, scored with
  evaluation.py, in Python. Depth 1 grabs anything hanging and never sees the
  recapture; depth 2 does. Neither is strong, which is the point: a yardstick
  an RL net should pass early, and whose score keeps meaning the same thing.
  ClassicalPlayer is the same engine in Rust and is what actually gets used
  when the extension is built; this is its fallback."""

  MATE = 1000000

  def __init__(self, depth=1):
    self.depth = depth

  def choose(self, boards):
    return [self.best(board) for board in boards]

  def best(self, board):
    best, alpha = None, -math.inf
    for move in self.ordered(board):
      board.push(move)
      try:
        score = -self.negamax(board, self.depth - 1, -math.inf, -alpha, 1)
      finally:
        board.pop()
      if best is None or score > alpha:
        best, alpha = move, score
    return best

  def negamax(self, board, depth, alpha, beta, ply):
    legal = list(board.legal_moves)
    if not legal:
      return -(self.MATE - ply) if board.is_check() else 0
    if (board.halfmove_clock >= 100 or board.is_insufficient_material()
        or (board.halfmove_clock >= 4 and board.is_repetition(2))):
      return 0
    if depth <= 0:
      return static_eval(board, board.turn)
    for move in self.ordered(board, legal):
      board.push(move)
      try:
        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
      finally:
        board.pop()
      if score >= beta:
        return score
      alpha = max(alpha, score)
    return alpha

  @staticmethod
  def ordered(board, legal=None):
    legal = list(board.legal_moves) if legal is None else legal
    return sorted(legal, key=lambda m: not board.is_capture(m))

  def close(self):
    pass

class ClassicalPlayer:
  """The small classical engine in nighty_rs: evaluation.py's evaluation
  under a shallow alpha-beta, searched in parallel for a whole list of boards.
  A different kind of opponent from any net -- materialist, tactically sharp
  to its horizon and blind past it -- and cheap enough to be a real share of
  self-play. Falls back to GreedyPlayer without the Rust build."""

  def __init__(self, depth=2, seed=0):
    import rl_backend
    self.depth = depth
    self.seed = seed
    self.fallback = None if rl_backend.rust_available() else GreedyPlayer(depth)

  def choose(self, boards):
    if self.fallback is not None:
      return self.fallback.choose(boards)
    import nighty_rs
    positions = []
    for board in boards:
      k = min(board.halfmove_clock, len(board.move_stack))
      if k:
        recent = board.copy(stack=k)
        positions.append((recent.root().fen(), [m.uci() for m in recent.move_stack]))
      else:
        positions.append((board.fen(), []))
    self.seed += len(boards)
    return [chess.Move(f, t, promotion=p or None)
            for f, t, p in nighty_rs.classical_moves(positions, self.depth, self.seed)]

  def close(self):
    pass

class UCIPlayer:
  def __init__(self, command, seconds, cwd=None, options=None):
    import chess.engine
    self.engine = chess.engine.SimpleEngine.popen_uci(
      command, cwd=cwd, stderr=subprocess.DEVNULL, timeout=60)
    if options:
      self.engine.configure(options)
    self.limit = chess.engine.Limit(time=seconds)

  def choose(self, boards):
    # One engine, so one position at a time. It gets the whole game, not just
    # the position, so it can see repetitions.
    return [self.engine.play(board, self.limit).move for board in boards]

  def close(self):
    try:
      self.engine.quit()
    except Exception:
      pass

def stockfish_path():
  """$STOCKFISH, else the binary on PATH, else where the usual packages put it
  (Debian and Ubuntu's apt package lives in /usr/games, which is not on PATH
  in most containers)."""
  found = os.environ.get('STOCKFISH') or shutil.which('stockfish')
  for candidate in ('/usr/games/stockfish', '/usr/local/bin/stockfish',
                    '/opt/homebrew/bin/stockfish', '/usr/bin/stockfish'):
    if found:
      break
    if os.access(candidate, os.X_OK):
      found = candidate
  return found

def stockfish_player(elo, seconds):
  """Stockfish held down to `elo` with its own UCI_LimitStrength handicap,
  clamped to what the binary allows (1320-3190 in Stockfish 16+). The player
  keeps the level it actually got as `.elo`."""
  path = stockfish_path()
  if not path:
    raise FileNotFoundError('no stockfish binary: install it (brew/apt/dnf install stockfish) '
                            'or set STOCKFISH=/path/to/it')
  player = UCIPlayer([path], seconds)
  option = player.engine.options['UCI_Elo']
  player.elo = int(max(option.min, min(option.max, elo)))
  player.engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': player.elo})
  return player

def elo_estimate(summary, opponent_elo):
  """Our Elo from a match score against an opponent of known Elo. A whitewash
  either way is read as if one game had gone the other way, so it gives a
  bound instead of infinity."""
  n = summary['games']
  score = min(max(summary['score'], 0.5 / n), 1 - 0.5 / n)
  return opponent_elo + 400 * math.log10(score / (1 - score))

def make_player(spec, device='cpu', sims=200, seconds=0.5, backend='auto'):
  """random | greedy[:depth] | classical[:depth] | nightybot | stockfish[:elo] | uci:<command> | rl:<weights file>"""
  kind, _, arg = spec.partition(':')
  if kind == 'random':
    return RandomPlayer()
  if kind == 'stockfish':
    return stockfish_player(int(arg) if arg else 1320, seconds)
  if kind == 'greedy':      # the classical engine one ply deep
    return ClassicalPlayer(int(arg) if arg else 1)
  if kind == 'classical':
    return ClassicalPlayer(int(arg) if arg else 2)
  if kind == 'nightybot':
    return UCIPlayer([sys.executable, os.path.join(HERE, 'run.py')], seconds, cwd=HERE)
  if kind == 'uci':
    return UCIPlayer(arg.split(), seconds)
  if kind == 'rl':
    return MCTSPlayer(load_net(arg, device).eval(), device, sims, backend=backend)
  raise ValueError(f'unknown player {spec!r}')

# --- matches ------------------------------------------------------------------

def random_opening(rng, plies):
  while True:
    board = chess.Board()
    for _ in range(plies):
      board.push(rng.choice(list(board.legal_moves)))
      if board.is_game_over():
        break
    if not board.is_game_over():
      return board

def play_match(player, opponent, games, opening_plies=4, max_plies=400,
               seed=0, should_stop=None, on_game=None):
  """Play `games` games and return player's result, or None if should_stop()
  said to give up partway. Games past max_plies are drawn."""
  rng = random.Random(seed)
  boards = []
  while len(boards) < games:
    opening = random_opening(rng, opening_plies)
    boards += [opening.copy(), opening.copy()]
  boards = boards[:games]
  player_white = [n % 2 == 0 for n in range(games)]
  results = [None] * games
  reasons = [None] * games

  while any(r is None for r in results):
    if should_stop is not None and should_stop():
      return None
    for side, is_player in ((player, True), (opponent, False)):
      todo = [n for n in range(games) if results[n] is None
              and ((boards[n].turn == chess.WHITE) == player_white[n]) == is_player]
      if not todo:
        continue
      for n, move in zip(todo, side.choose([boards[n] for n in todo])):
        board = boards[n]
        board.push(move)
        over = game_over(board)
        if over is None and board.ply() >= max_plies:
          over = (0, 'max-plies')
        if over is not None:
          white_score, reasons[n] = over
          results[n] = white_score if player_white[n] else -white_score
          if on_game is not None:
            on_game(n, results[n], reasons[n], board, player_white[n])
  return summarize(results)

def summarize(results):
  n = len(results)
  wins = results.count(1)
  draws = results.count(0)
  losses = results.count(-1)
  score = (wins + 0.5 * draws) / n
  elo = margin = None
  if 0 < score < 1:
    elo = -400 * math.log10(1 / score - 1)
    # Delta method on the per-game score, for a rough 95% interval.
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2
           + losses * score ** 2) / n
    margin = 1.96 * math.sqrt(var / n) * 400 / (math.log(10) * score * (1 - score))
    if margin == 0:  # every game the same result: the sample says nothing about spread
      margin = None
  return {'wins': wins, 'draws': draws, 'losses': losses, 'games': n,
          'score': score, 'elo': elo, 'elo_margin': margin}

def describe(summary):
  s = summary
  text = f"+{s['wins']} ={s['draws']} -{s['losses']}  score {s['score'] * 100:.1f}%"
  if s['elo'] is None:
    return text + ('  (won everything)' if s['score'] >= 1 else '  (lost everything)')
  if s['elo_margin'] is None:
    return text + f"  elo {s['elo']:+.0f} (all draws)"
  return text + f"  elo {s['elo']:+.0f} +/- {s['elo_margin']:.0f}"

def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--weights', default=os.path.join(HERE, 'nighty_rl.pt'))
  p.add_argument('--opponent', default='greedy',
                 help='random | greedy[:depth] | classical[:depth] | nightybot | stockfish[:elo] | '
                      'uci:<command> | rl:<weights>')
  p.add_argument('--games', type=int, default=20)
  p.add_argument('--sims', type=int, default=200, help='simulations per move for the RL side(s)')
  p.add_argument('--opponent-time', type=float, default=0.5,
                 help='seconds per move for a UCI opponent')
  p.add_argument('--opening-plies', type=int, default=4,
                 help='random plies before each pair of games')
  p.add_argument('--max-plies', type=int, default=400)
  p.add_argument('--device', default='cpu',
                 help="'cpu' is fastest for these batch sizes on Apple silicon; 'auto' picks CUDA/MPS")
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--backend', default='auto', choices=['auto', 'rust', 'python'])
  args = p.parse_args()

  device = pick_device(args.device)
  torch.set_num_threads(1)
  player = MCTSPlayer(load_net(args.weights, device).eval(), device, args.sims,
                      backend=args.backend)
  opponent = make_player(args.opponent, device, args.sims, args.opponent_time, args.backend)
  print(f'{os.path.basename(args.weights)} ({args.sims} sims) vs {args.opponent}, '
        f'{args.games} games', flush=True)

  started = time.time()
  def on_game(n, result, reason, board, as_white):
    print(f'  game {n + 1:3d}: rl as {"white" if as_white else "black"} '
          f'{("lost", "drew", "won")[result + 1]} by {reason} in {board.ply()} plies',
          flush=True)

  try:
    summary = play_match(player, opponent, args.games, args.opening_plies,
                         args.max_plies, seed=args.seed, on_game=on_game)
  finally:
    opponent.close()
  line = describe(summary)
  if hasattr(opponent, 'elo'):
    line += f'   => about {elo_estimate(summary, opponent.elo):.0f} Elo (vs Stockfish at {opponent.elo})'
  print(f'{line}   ({time.time() - started:.0f}s)')

if __name__ == '__main__':
  main()
