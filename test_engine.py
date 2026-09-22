"""Sanity checks for the evaluation and the search.

Not a unit test suite so much as a set of traps for the bugs that used to make
the engine play badly at the start and the end of a game. Run it directly:

    python3 test_engine.py

It loads the real checkpoints, so it takes a few seconds to start.
"""

import random
import subprocess
import sys
import time

import chess

import run
from evaluation import eval_pos

failures = []

def check(name, ok, detail=''):
  print(('  pass  ' if ok else '  FAIL  ') + name + (('   ' + detail) if detail else ''))
  if not ok:
    failures.append(name)

def section(name):
  print('\n' + name)

def ev(fen, side=1):
  return eval_pos(run.encode(fen), side)

def best(fen, depth=4, budget=20.0):
  board = chess.Board(fen)
  return run.find_best_move(board, limit=depth, budget=budget)[0]


section('encoding')
random.seed(11)
board = chess.Board()
mismatch = 0
for _ in range(400):
  moves = list(board.legal_moves)
  if not moves:
    board = chess.Board()
    continue
  board.push(random.choice(moves))
  if run.encode_board(board) != run.encode(board.fen()):
    mismatch += 1
check('encode_board matches encode(fen)', mismatch == 0, f'({mismatch} mismatches)')

section('evaluation')
check('rook is worth more than bishop',
      ev('4k3/8/8/8/8/8/8/R3K1b1') > 100)
check('a pawn on the 7th beats a pawn on the 2nd',
      ev('7k/P7/8/8/8/8/8/7K') > ev('7k/8/8/8/8/8/P7/7K') + 100)
check('K+N vs K is a draw', ev('7k/8/8/8/8/8/8/6NK') == 0)
check('castling is worth something',
      ev('rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - -')
      > ev('rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQ -'))
check('development is worth something',
      ev('rnbqkbnr/pppppppp/8/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq -') > 30)
check('mop-up drives the bare king to the edge',
      ev('7k/8/6K1/8/8/8/8/6Q1') > ev('8/8/8/3k4/8/8/K7/6Q1'))

# Includes lopsided material so the mop-up term is exercised, not just the tables.
antisym = 0
for fen in ['7k/8/6K1/8/8/8/8/6Q1 w - -', '8/8/8/3k4/8/8/K7/6Q1 w - -',
            '8/8/4k3/8/8/2K5/8/4R3 w - -', '8/5k2/8/8/8/1K6/8/5R2 b - -']:
  if eval_pos(run.encode(fen), 1) != -eval_pos(run.encode(chess.Board(fen).mirror().fen()), 1):
    antisym += 1
b = chess.Board()
for _ in range(200):
  moves = list(b.legal_moves)
  if not moves:
    b = chess.Board()
    continue
  b.push(random.choice(moves))
  if eval_pos(run.encode_board(b), 1) != -eval_pos(run.encode_board(b.mirror()), 1):
    antisym += 1
check('evaluation is antisymmetric under colour mirror', antisym == 0,
      f'({antisym} mismatches)')

section('search')
move = best('8/P6k/8/8/8/8/8/K7 w - - 0 1', depth=3)
check('promotes a free pawn', move is not None and move.promotion is not None,
      f'(played {move})')

move = best('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1', depth=3)
check('finds mate in one', move == chess.Move.from_uci('a1a8'), f'(played {move})')

# White is a queen up; Qc7 would be stalemate and must not be chosen.
move = best('k7/8/2Q5/8/8/8/8/7K w - - 0 1', depth=3)
after = chess.Board('k7/8/2Q5/8/8/8/8/7K w - - 0 1')
after.push(move)
check('does not stalemate a won position', not after.is_stalemate(), f'(played {move})')

board = chess.Board('k7/8/2Q5/8/8/8/8/7K w - - 0 1')
_, score = run.search(board, 0, 2, {}, time.time() + 30, run.threading.Event())
check('a won position scores as won', score > 500, f'(score {score})')

board = chess.Board('k7/2Q5/8/8/8/8/8/7K b - - 0 1')
_, score = run.search(board, 0, 2, {}, time.time() + 30, run.threading.Event())
check('stalemate scores 0', score == 0, f'(score {score})')

board = chess.Board('6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1')
board.push_uci('g8h8')
board.push_uci('a1a8')
_, score = run.search(board, 0, 2, {}, time.time() + 30, run.threading.Event())
check('being mated scores as mated', score < -(run.MATE - 1000), f'(score {score})')

section('repetition')
board = chess.Board('8/8/8/4k3/8/4K3/4P3/8 w - - 6 1')
seen = run.history_keys(board)
check('history includes the position on the board',
      seen.get(chess.polyglot.zobrist_hash(board)) == 1)

board = chess.Board()
for uci in ['e2e4', 'e7e5', 'g1f3', 'g8f6', 'f3g1', 'f6g8']:
  board.push_uci(uci)
seen = run.history_keys(board)
repeated = max(seen.values())
check('a repeated position is counted twice', repeated == 2, f'(max count {repeated})')

board = chess.Board()
for uci in ['e2e4', 'e7e5', 'g1f3', 'g8f6', 'f3g1']:
  board.push_uci(uci)
seen = run.history_keys(board)
board.push_uci('f6g8')  # back to a position already seen
score = run.child_score(board, 1, 3, seen, time.time() + 30, run.threading.Event())
check('a move that repeats scores as a draw', score == 0, f'(score {score})')

section('time management')
board = chess.Board()
board.turn = chess.WHITE
check('uses the clock when given one',
      run.time_budget(board, {'wtime': 60000, 'binc': 0, 'winc': 0}) is not None)
check('movetime is honoured',
      abs(run.time_budget(board, {'movetime': 1000}) - (1.0 - run.move_overhead)) < 1e-9)
check('an untimed go is still capped',
      run.time_budget(board, {}) == run.default_budget)
check('a nearly flagged clock still returns something positive',
      run.time_budget(board, {'wtime': 50}) > 0)

started = time.time()
run.find_best_move(chess.Board(), limit=8, budget=2.0)
elapsed = time.time() - started
check('respects the time budget', elapsed < 8.0, f'({elapsed:.1f}s for a 2.0s budget)')

section('uci')
def talk(lines):
  p = subprocess.run([sys.executable, 'run.py'], input=''.join(l + '\n' for l in lines),
                     capture_output=True, text=True, timeout=300)
  return p.stdout, p.stderr

out, err = talk([
  'uci', 'isready', 'ucinewgame',
  'position startpos',
  'go depth 2',
  'position fen 8/8/8/4k3/8/4K3/4P3/8 w - - 0 1',
  'go depth 2',
  'position startpos moves e2e4 e7e5',
  'go movetime 500',
  'go',
  'setoption name Depth value 3',
  'quit',
])
check('answers uci', 'uciok' in out)
check('answers isready', 'readyok' in out)
check('returns a move for every go', out.count('bestmove') == 4,
      f'({out.count("bestmove")} bestmove lines)')
check('parses a fen position', 'bad fen' not in out, err.strip()[-200:])
check('setoption takes effect', 'set to 3' in out)
check('nothing written to stderr', not err.strip(), err.strip()[-300:])

# The old command loop recursed once per command and blew the stack in long games.
lines = ['uci']
game = ['e2e4', 'e7e5', 'g1f3', 'g8f6', 'f3g1', 'f6g8']
for i in range(1, 400):
  lines.append('position startpos moves ' + ' '.join(game[:i % 6 + 1]))
lines.append('quit')
out, err = talk(lines)
check('survives a long stream of commands', not err.strip(), err.strip()[-300:])

print('\n' + ('all checks passed' if not failures
              else f'{len(failures)} failed: {failures}'))
sys.exit(1 if failures else 0)
