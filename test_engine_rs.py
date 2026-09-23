"""Checks for the Rust build of the imitation engine (engine/), against run.py.

    cargo build --release --manifest-path engine/Cargo.toml
    python3 test_engine_rs.py

The first two sections need the torch checkpoints and the .safetensors exports
to be the same weights (rerun save.py after retraining): they compare the Rust
nets' raw logits and the shortlist candidates() builds with run.py's, position
by position. The rest drives the binary over UCI the way test_engine.py drives
run.py, so it takes about a minute.
"""

import os
import random
import subprocess
import sys
import time

import chess
import torch

import run

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engine', 'target', 'release', 'nightybot')
if not os.path.exists(BIN):
  print(f'{BIN} is missing; build it with: cargo build --release --manifest-path engine/Cargo.toml')
  sys.exit(1)

failures = []

def check(name, ok, detail=''):
  print(('  pass  ' if ok else '  FAIL  ') + name + (('   ' + detail) if detail else ''))
  if not ok:
    failures.append(name)

def section(name):
  print('\n' + name)

def rs(*args):
  p = subprocess.run([BIN, *args], capture_output=True, text=True, timeout=120)
  if p.returncode:
    raise RuntimeError(p.stderr)
  return p.stdout

def talk(lines, timeout=300):
  p = subprocess.run([BIN], input=''.join(l + '\n' for l in lines),
                     capture_output=True, text=True, timeout=timeout)
  return p.stdout, p.stderr

def bestmove(out):
  for line in out.splitlines():
    if line.startswith('bestmove'):
      return chess.Move.from_uci(line.split()[1])
  return None

def random_positions(n, seed):
  rng = random.Random(seed)
  board = chess.Board()
  out = []
  while len(out) < n:
    moves = list(board.legal_moves)
    if not moves or board.fullmove_number > 60:
      board = chess.Board()
      continue
    board.push(rng.choice(moves))
    if list(board.legal_moves):
      out.append(board.copy())
  return out


section('nets')
worst_from = worst_to = 0.0
for board in random_positions(30, 3):
  lines = rs('logits', board.fen()).splitlines()
  rs_from = [float(v) for v in lines[0].split()[1:]]
  rs_to = [[float(v) for v in l.split()[2:]] for l in lines[1:]]
  enc = torch.tensor([run.encode_board(board)])
  with torch.no_grad():
    py_from = run.fmodel(enc).flatten().tolist()
    py_to = run.model(enc, torch.arange(64)).tolist()
  worst_from = max(worst_from, max(abs(a - b) for a, b in zip(rs_from, py_from)))
  worst_to = max(worst_to, max(abs(a - b) for ra, pa in zip(rs_to, py_to) for a, b in zip(ra, pa)))
check('from-model logits match torch', worst_from < 1e-3, f'(max diff {worst_from:.2e})')
check('to-model logits match torch', worst_to < 1e-3, f'(max diff {worst_to:.2e})')

section('candidates')
positions = random_positions(200, 5)
same = 0
examples = []
for board in positions:
  py = [m.uci() for m in run.candidates(board, run.encode_board(board), list(board.legal_moves))]
  rust = rs('candidates', board.fen()).split()
  if py == rust:
    same += 1
  elif len(examples) < 2:
    examples.append((board.fen(), py, rust))
# Ties and near-ties in the logits may order two moves differently; anything
# more than a handful is a real divergence.
check('shortlists agree with run.py', same >= len(positions) - 4, f'({same}/{len(positions)} identical)')
for fen, py, rust in examples:
  print(f'        {fen}\n          run.py {py}\n          rust   {rust}')

section('search')
# Alpha-beta inside the root subtrees is the one thing the port adds; it has
# to return run.py's values and move exactly, not just close ones.
agree = 0
pairs = [('r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3', 3),
         ('rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 1 5', 3),
         ('8/5k2/8/8/8/1K6/8/4R3 w - - 0 1', 4)]
for fen, depth in pairs:
  move, score = run.find_best_move(chess.Board(fen), limit=depth, budget=120)
  out, _ = talk([f'position fen {fen}', f'go depth {depth}', 'quit'])
  last = [l.split() for l in out.splitlines() if l.startswith('info depth')][-1]
  rs_score = int(last[last.index('score') + 2])
  if (move, score) == (bestmove(out), rs_score):
    agree += 1
  else:
    print(f'        {fen}: run.py {move} {score}, rust {bestmove(out)} {rs_score}')
check('same move and score as run.py at equal depth', agree == len(pairs), f'({agree}/{len(pairs)})')

out, _ = talk(['position fen 8/P6k/8/8/8/8/8/K7 w - - 0 1', 'go depth 3', 'quit'])
move = bestmove(out)
check('promotes a free pawn', move is not None and move.promotion is not None, f'(played {move})')

out, _ = talk(['position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'go depth 3', 'quit'])
check('finds mate in one', bestmove(out) == chess.Move.from_uci('a1a8'), f'(played {bestmove(out)})')
check('reports it as mate', 'score mate 1' in out)

# White is a queen up; Qc7 would be stalemate and must not be chosen.
# (test_engine.py's version of this position has the queen on c6, giving check
# with White to move; python-chess does not mind, shakmaty refuses it.)
out, _ = talk(['position fen k7/8/1Q6/8/8/8/8/7K w - - 0 1', 'go depth 3', 'quit'])
after = chess.Board('k7/8/1Q6/8/8/8/8/7K w - - 0 1')
move = bestmove(out)
legal = move in after.legal_moves
if legal:
  after.push(move)
check('does not stalemate a won position', legal and not after.is_stalemate(), f'(played {move})')

# Black's king has walked into the corner; Ra1-a8 is mate in one for White,
# which Black should see coming and avoid at depth 2.
out, _ = talk(['position fen 6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1', 'go depth 2', 'quit'])
check('avoids walking into mate', bestmove(out) != chess.Move.from_uci('g8h8'), f'(played {bestmove(out)})')

# White is better but the only non-losing continuation repeats; the search has
# to count the game history, not only its own line, to see the draw claim.
out, _ = talk(['position startpos moves e2e4 e7e5 g1f3 g8f6 f3g1 f6g8 g1f3 g8f6',
               'go depth 2', 'quit'])
check('a game with history still returns a move', bestmove(out) is not None)

section('time management')
started = time.time()
out, _ = talk(['position startpos', 'setoption name Depth value 12', 'go movetime 2000', 'quit'])
elapsed = time.time() - started
check('respects movetime', bestmove(out) is not None and elapsed < 5.0, f'({elapsed:.1f}s for movetime 2000)')

started = time.time()
out, _ = talk(['position startpos', 'go wtime 50 btime 50', 'quit'])
elapsed = time.time() - started
check('a nearly flagged clock still moves quickly', bestmove(out) is not None and elapsed < 2.0,
      f'({elapsed:.2f}s)')

# stop has to interrupt a running search, not wait for it.
p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdin.write('position startpos\nsetoption name Depth value 30\ngo infinite\n')
p.stdin.flush()
time.sleep(1.0)
started = time.time()
p.stdin.write('stop\n')
p.stdin.flush()
line = ''
while not line.startswith('bestmove') and time.time() - started < 10:
  line = p.stdout.readline()
elapsed = time.time() - started
p.stdin.write('quit\n')
p.stdin.flush()
p.wait(timeout=10)
check('stop interrupts go infinite', line.startswith('bestmove') and elapsed < 1.0, f'({elapsed:.2f}s to answer)')

section('uci')
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
check('returns a move for every go', out.count('bestmove') == 4, f'({out.count("bestmove")} bestmove lines)')
check('parses a fen position', 'bad fen' not in out, err.strip()[-200:])
check('setoption takes effect', 'set to 3' in out)
check('nothing written to stderr', not err.strip(), err.strip()[-300:])

out, err = talk(['position startpos', 'go infinite', 'stop', 'stop', 'quit'])
check('go infinite answers once, on stop', out.count('bestmove') == 1, f'({out.count("bestmove")} bestmove lines)')

lines = ['uci']
game = ['e2e4', 'e7e5', 'g1f3', 'g8f6', 'f3g1', 'f6g8']
for i in range(1, 400):
  lines.append('position startpos moves ' + ' '.join(game[:i % 6 + 1]))
lines.append('quit')
out, err = talk(lines)
check('survives a long stream of commands', not err.strip(), err.strip()[-300:])

section('speed (informational)')
fen = 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3'
count = [0]
orig = run.search
def counting(*a, **k):
  count[0] += 1
  return orig(*a, **k)
run.search = counting
board = chess.Board(fen)
started = time.time()
run.find_best_move(board, limit=3, budget=120)
py_dt = time.time() - started
run.search = orig
print(f'  run.py     depth 3: {count[0]} nodes in {py_dt:.2f}s -> {count[0] / py_dt:.0f} nodes/s')
for line in rs('bench', '4').splitlines():
  if line.startswith('total'):
    print('  nightybot  depth 4: ' + line[len('total '):])

print('\n' + ('all checks passed' if not failures else f'{len(failures)} failed: {failures}'))
sys.exit(1 if failures else 0)
