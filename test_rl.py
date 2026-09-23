"""Sanity checks for the reinforcement-learning engine: the encodings, the
search, a training step, a short real training run, and the UCI loop.

    python3 test_rl.py

It does not need a trained model: it trains a tiny one for a few seconds in a
temporary directory and plays with that. Takes about a minute.
"""

import os
import random
import subprocess
import sys
import tempfile

import chess
import numpy as np
import torch

import rl_backend
import rl_eval
import train_rl
from rl_mcts import MCTS, Tree, best_child, game_over
from rl_model import (N_MOVES, OUR_CASTLING_ROOK, THEIR_CASTLING_ROOK, EP_TOKEN,
                      Evaluator, RLNet, encode, load_net, move_index)

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
failures = []

def check(name, ok, detail=''):
  print(('  pass  ' if ok else '  FAIL  ') + name + (('   ' + detail) if detail else ''))
  if not ok:
    failures.append(name)

def section(name):
  print('\n' + name)

def random_positions(n, seed):
  rng = random.Random(seed)
  board = chess.Board()
  for _ in range(n):
    if board.is_game_over():
      board = chess.Board()
    yield board
    board.push(rng.choice(list(board.legal_moves)))

torch.manual_seed(0)  # the untrained net's random priors decide what a small search finds
net = RLNet().eval()
mcts = MCTS(Evaluator(net, 'cpu'), rng=np.random.default_rng(0))

def search(fen_or_board, sims, leaves=1):
  board = fen_or_board if isinstance(fen_or_board, chess.Board) else chess.Board(fen_or_board)
  tree = Tree(board)
  tree.reset_search(sims)
  mcts.run([tree], leaves)
  return tree


section('encoding')
mirror = index = 0
for board in random_positions(2000, 7):
  if encode(board) != encode(board.mirror()):
    mirror += 1
  legal = list(board.legal_moves)
  idx = [move_index(m, board.turn) for m in legal]
  if len(set(idx)) != len(idx) or min(idx) < 0 or max(idx) >= N_MOVES:
    index += 1
  flipped = board.mirror()
  for m in legal:
    mm = chess.Move(m.from_square ^ 56, m.to_square ^ 56, m.promotion)
    if move_index(m, board.turn) != move_index(mm, flipped.turn):
      mirror += 1
check('both colours see the same thing (encode and move_index are mirror-invariant)',
      mirror == 0, f'({mirror} mismatches)')
check('move indices are unique and in range', index == 0, f'({index} bad positions)')

board = chess.Board('1n1n4/2P5/8/8/8/8/8/K6k w - - 0 1')
idx = [move_index(m, board.turn) for m in board.legal_moves]
check('every promotion, under- or not, has its own index', len(set(idx)) == len(idx))

tokens, _ = encode(chess.Board())
check('castling rooks are marked',
      [tokens[s] for s in (chess.A1, chess.H1, chess.A8, chess.H8)]
      == [OUR_CASTLING_ROOK] * 2 + [THEIR_CASTLING_ROOK] * 2)
board = chess.Board()
for uci in ['e2e4', 'd7d5', 'e4e5', 'f7f5']:
  board.push_uci(uci)
check('a legal en passant square is marked', encode(board)[0][chess.F6] == EP_TOKEN)
board = chess.Board()
for uci in ['e2e4', 'd7d5', 'e4e5', 'a7a6', 'a2a3', 'f7f5']:
  board.push_uci(uci)
board.push_uci('a3a4')  # black to move: no en passant for black here
check('an unusable en passant square is not', EP_TOKEN not in encode(board)[0])


section('search')
tree = search('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 200)
move = tree.root.moves[best_child(tree.root)]
check('an untrained net still finds mate in one by searching',
      move == chess.Move.from_uci('a1a8'), f'(played {move})')

def consistent(node):
  if node is None or node.N is None:
    return True
  return (not node.pending
          and abs(node.visits - 1 - node.N.sum()) < 1e-3
          and bool(np.all(np.abs(node.W) <= node.N + 1e-3))
          and all(consistent(c) for c in node.children))

tree = search(chess.Board(), 400, leaves=8)
check('virtual loss is fully taken back out', consistent(tree.root))
check('batched search does exactly the simulations asked for',
      tree.sims == 400 and tree.root.visits == 400, f'(sims {tree.sims}, visits {tree.root.visits})')

trees = [Tree(b.copy()) for b in random_positions(12, 3)]
for n, t in enumerate(trees):
  t.reset_search(20 + n * 10)
mcts.run(trees)
check('lockstep search gives each tree its own budget',
      all(t.sims == 20 + n * 10 for n, t in enumerate(trees)))

board = chess.Board()
for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
  board.push_uci(uci)
tree = search(board, 50)
check('a repeated position at the root is still searched',
      tree.root.terminal is None and tree.root.N is not None and tree.root.N.sum() > 0)
board.push_uci('g1f3')
tree = search(board, 300)
reply = tree.root.children[tree.root.moves.index(chess.Move.from_uci('g8f6'))]
check('a repetition inside the tree is scored as a draw',
      reply is not None and reply.terminal == 0.0)

tree = search(chess.Board(), 100)
kept = tree.root.moves[best_child(tree.root)]
visits = int(tree.root.N[best_child(tree.root)])
tree.advance(kept)
check('advancing keeps the searched subtree', tree.root.visits == visits,
      f'({tree.root.visits} vs {visits})')

board = chess.Board()
for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
  board.push_uci(uci)
check('threefold repetition ends a real game', game_over(board) == (0, 'repetition'))

backends = ['python'] + (['rust'] if rl_backend.rust_available() else [])
if len(backends) == 1:
  print('  (nighty_rs is not built, so the Rust backend is not tested)')

if rl_backend.rust_available():
  section('rust')
  import nighty_rs
  mismatches = 0
  for board in random_positions(3000, 13):
    fen = board.fen()
    tokens, halfmove = nighty_rs.encode_fen(fen)
    if list(tokens) != encode(board)[0] or abs(halfmove - encode(board)[1]) > 1e-6:
      mismatches += 1
    rust = {(f, t, p): i for (f, t, p), i in nighty_rs.legal_moves_fen(fen)}
    python = {(m.from_square, m.to_square, m.promotion or 0): move_index(m, board.turn)
              for m in board.legal_moves}
    if rust != python:
      mismatches += 1
  check('Rust and Python agree on every token and move index', mismatches == 0,
        f'({mismatches} positions differ)')

for name in backends:
  section(f'search through the {name} backend')
  backend = rl_backend.make_backend(name, net, 'cpu', seed=3, threads=2)
  def searched(board, sims, leaves=1):
    tree = backend.tree(board.copy())
    tree.reset_search(sims)
    backend.run([tree], leaves)
    return tree
  tree = searched(chess.Board('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1'), 200)
  check('finds mate in one', tree.moves()[tree.best()] == chess.Move.from_uci('a1a8')
        and tree.mate_in() == 1, f'({tree.moves()[tree.best()]}, mate_in {tree.mate_in()})')
  tree = searched(chess.Board('7k/8/8/5K2/8/8/8/6Q1 w - - 0 1'), 3000, leaves=8)
  check('reports a forced mate it has seen', tree.mate_in() is not None and tree.mate_in() > 0,
        f'(mate_in {tree.mate_in()})')
  check('does the simulations asked for', tree.sims == 3000 and abs(tree.visits().sum() - 2999) < 1,
        f'(sims {tree.sims}, visits {tree.visits().sum()})')
  board = chess.Board()
  for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
    board.push_uci(uci)
  tree = searched(board, 50)
  check('a repeated root is still searched', tree.expanded() and tree.visits().sum() > 0)
  check('the game is not over at a twofold repetition', tree.game_over() is None)
  for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
    board.push_uci(uci)
  check('and is over at threefold', backend.tree(board.copy()).game_over() == (0, 'repetition'))
  tree = searched(chess.Board(), 100)
  best = tree.best()
  kept = int(tree.visits()[best])
  tree.advance(tree.moves()[best])
  tree.reset_search(50)
  backend.run([tree])
  check('advancing keeps the searched subtree', tree.visits().sum() >= kept - 1 + 49,
        f'({tree.visits().sum()} visits after keeping {kept})')
  tokens, halfmove, indices = tree.encoding()
  check('the root encoding matches rl_model', list(tokens) == encode(tree.board)[0]
        and list(indices) == [move_index(m, tree.board.turn) for m in tree.moves()])
  trees = [backend.tree(b.copy()) for b in random_positions(12, 3)]
  for n, t in enumerate(trees):
    t.reset_search(20 + n * 10)
  backend.run(trees)
  check('many trees get their own budgets', all(t.sims == 20 + n * 10 for n, t in enumerate(trees)))
  for t in trees:
    t.close()


section('learning')
args = train_rl.parse_args(['--sims', '16', '--fast-sims', '4', '--full-prob', '1',
                            '--max-plies', '40', '--resign', '-1', '--backend', backends[-1]])
rng = np.random.default_rng(1)
game_backend = rl_backend.make_backend(backends[-1], net, 'cpu', seed=1, threads=2)
class Flag:
  value = 1
games = [train_rl.SelfPlayGame(args, rng, Flag(), game_backend) for _ in range(4)]
record = None
while record is None:
  for game in games:
    if game.search_done():
      record = record or game.play_move()
  game_backend.step([g.tree for g in games])
n = record['positions']
check('a finished game becomes training rows', n > 0 and record['tokens'].shape == (n, 64))
probs = record['probs'].astype(np.float32)
check('policy targets are distributions over legal moves only',
      np.allclose(probs.sum(1), 1, atol=1e-2) and np.all(probs[record['legal'] < 0] == 0))
check('value targets agree with the result',
      set(record['wdl']) <= ({1} if record['result'] == 0 else {0, 2}))

# An opponent pool: a snapshot of the net on disk, plus random and greedy.
pool_dir = tempfile.mkdtemp(prefix='nighty_rl_pool_')
os.makedirs(os.path.join(pool_dir, 'rl_snapshots'))
from rl_model import save_net
save_net(net, os.path.join(pool_dir, 'rl_snapshots', 'step_0000001.pt'))
pool_args = train_rl.parse_args(['--sims', '8', '--fast-sims', '4', '--max-plies', '30', '--resign', '-1',
                                 '--backend', backends[-1], '--out-dir', pool_dir,
                                 '--opponents', 'self=1,snapshot=1,random=1,greedy=1',
                                 '--opponents-final', 'self=1', '--opponents-steps', '10'])
pool = train_rl.OpponentPool(pool_args, 'cpu', rng, os.path.join(pool_dir, 'rl_snapshots'))
check('the pool loads the snapshot', len(pool.snapshots) == 1)
mix = train_rl.mix_at(pool_args, 5)
check('the opponent mix slides towards --opponents-final',
      abs(mix['self'] - 0.625) < 1e-6 and abs(mix['greedy'] - 0.125) < 1e-6, f'({mix})')
pool_games = [train_rl.SelfPlayGame(pool_args, rng, Flag(), game_backend, pool, 0) for _ in range(12)]
seen = {}
while len(seen) < 4 and sum(1 for g in pool_games if g is not None) and len(seen) < 4:
  everything = [game_backend] + pool.backends()
  done = {id(b): b.done_ids() for b in everything}
  for j, g in enumerate(pool_games):
    if g is not None and g.ready(done):
      rec = g.play_move()
      if rec is not None:
        seen.setdefault(rec['opponent'], rec)
        pool_games[j] = train_rl.SelfPlayGame(pool_args, rng, Flag(), game_backend, pool, 0)
  for b in everything:
    b.step(None)
check('games finish against every kind of opponent', set(seen) == set(train_rl.OPPONENT_KINDS),
      f'({sorted(seen)})')
check('only the current net\'s own moves become rows',
      all(r['positions'] <= (r['plies'] + 1) // 2 + 1 for k, r in seen.items() if k != 'self'))
check('pool games report our score', all(seen[k]['ours'] in (-1, 0, 1) for k in seen if k != 'self')
      and seen['self']['ours'] is None)

replay = train_rl.Replay(50)
for _ in range(4):
  replay.add(record)
check('the replay buffer wraps around', replay.size == min(50, 4 * n) and replay.pos == (4 * n) % 50)

learner = RLNet(width=64, blocks=1)
opt = torch.optim.AdamW(learner.parameters(), lr=3e-3)
batch = train_rl.to_device(replay.sample(32, rng), 'cpu')
first = None
for _ in range(60):
  policy, value, _ = train_rl.losses(learner, batch)
  first = first if first is not None else (policy + value).item()
  opt.zero_grad()
  (policy + value).backward()
  opt.step()
last = (policy + value).item()
check('a fixed batch can be overfit', np.isfinite(last) and last < first * 0.7,
      f'(loss {first:.3f} -> {last:.3f})')


section('training run')
tmp = tempfile.mkdtemp(prefix='nighty_rl_test_')
base = [sys.executable, os.path.join(HERE, 'train_rl.py'), '--out-dir', tmp,
        '--actors', '2', '--games-per-actor', '8', '--sims', '16', '--fast-sims', '4',
        '--max-plies', '60', '--min-buffer', '100', '--batch-size', '32',
        '--eval-every', '0', '--log-every', '5', '--seed', '1', '--actor-threads', '2']
for name in backends:
  cmd = base + ['--backend', name]
  p = subprocess.run(cmd + ['--max-steps', '5', '--fresh'], capture_output=True, text=True, timeout=600)
  check(f'train_rl.py runs with the {name} backend and exits cleanly', p.returncode == 0,
        p.stderr.strip()[-400:])
weights = os.path.join(tmp, 'nighty_rl.pt')
check('it writes the weights and a checkpoint',
      os.path.exists(weights) and os.path.exists(os.path.join(tmp, 'rl_checkpoint.pt')))
p = subprocess.run(cmd + ['--max-steps', '3', '--snapshot-every', '2',
                          '--opponents', 'self=0.4,snapshot=0.3,random=0.2,greedy=0.1'],
                   capture_output=True, text=True, timeout=600)
resumed = p.returncode == 0 and 'resuming' in p.stdout and 'at step 8' in p.stdout
check('a second run resumes from the checkpoint', resumed,
      '' if resumed else (p.stdout + p.stderr).strip()[-300:])
trained = load_net(weights).eval()


section('arena')
for name in backends:
  player = rl_eval.MCTSPlayer(trained, 'cpu', sims=8, backend=name, threads=2)
  summary = rl_eval.play_match(player, rl_eval.RandomPlayer(0), games=2, max_plies=40)
  check(f'a match runs and is scored ({name})', summary['games'] == 2
        and summary['wins'] + summary['draws'] + summary['losses'] == 2)
board = chess.Board('4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1')
check('the greedy yardstick takes a hanging queen',
      rl_eval.GreedyPlayer(1).choose([board])[0] == chess.Move.from_uci('d1d5'))


section('uci')
def talk(lines, name):
  p = subprocess.run([sys.executable, os.path.join(HERE, 'run_rl.py'), '--weights', weights,
                      '--backend', name, '--threads', '2'],
                     input=''.join(l + '\n' for l in lines),
                     capture_output=True, text=True, timeout=300)
  return p.stdout, p.stderr

for name in backends:
  out, err = talk([
    'uci', 'isready', 'ucinewgame',
    'position startpos', 'go nodes 50',
    'position startpos moves e2e4 e7e5', 'go movetime 300',
    'position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'go nodes 10',
    'position fen 8/8/8/4k3/8/4K3/4P3/8 w - - 0 1', 'go wtime 2000 btime 2000',
    'position fen 7k/8/8/5K2/8/8/8/6Q1 w - - 0 1', 'go nodes 4000',
    'setoption name Batch value 4',
    'quit',
  ], name)
  moves = [line.split()[1] for line in out.splitlines() if line.startswith('bestmove')]
  check(f'answers uci and isready ({name})', 'uciok' in out and 'readyok' in out)
  check(f'returns a move for every go ({name})', len(moves) == 5, f'({moves})')
  check(f'a queued quit does not cut a search short ({name})', 'nodes 50 ' in out)
  check(f'plays the mate in one ({name})', len(moves) > 2 and moves[2] == 'a1a8', f'({moves})')
  check(f'setoption takes effect ({name})', 'batch set to 4' in out)
  check(f'nothing written to stderr ({name})', not err.strip(), err.strip()[-300:])

p = subprocess.Popen([sys.executable, os.path.join(HERE, 'run_rl.py'), '--weights', weights],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
p.stdin.write('position startpos\ngo infinite\n')
p.stdin.flush()
line = p.stdout.readline()  # the first info line: it is searching
p.stdin.write('stop\n')
p.stdin.flush()
while line and not line.startswith('bestmove'):
  line = p.stdout.readline()
p.stdin.write('quit\n')
p.stdin.flush()
p.wait(timeout=30)
check('go infinite runs until stop, then answers', line.startswith('bestmove'), repr(line))

print('\n' + ('all checks passed' if not failures
              else f'{len(failures)} failed: {failures}'))
sys.exit(1 if failures else 0)
