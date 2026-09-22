import re
import sys
import time
import threading
import torch
import chess
import chess.polyglot
from torch import nn
from evaluation import eval_pos

device = "cuda" if torch.cuda.is_available() else "cpu"

# The nets are tiny and every forward pass here is a batch of at most a handful
# of rows, so intra-op parallelism buys nothing. The useful parallelism is one
# thread per root move, below.
torch.set_num_threads(1)

def encode(fen):
  fen = fen.split(" ")[0].replace("/", "")
  replaces = ".prnbqkPRNBQK"
  for n in range(1, 9):
    fen = fen.replace(str(n), "." * n)
  return list(map(replaces.index, fen))

def sqr_to_num(sqr):
  sqr = sqr.lower()
  indices = 'abcdefgh'
  return indices.index(sqr[0]) + 8 * (int(sqr[1]) - 1)

def num_to_sqr(num):
  indices = 'abcdefgh'
  return indices[num % 8] + str(num // 8 + 1)

# encode() goes through board.fen(), which builds and re-parses a string at every
# node. This is the same encoding straight off the board object -- squares are
# python-chess numbered (a1 = 0) and `sq ^ 56` mirrors them into encode()'s FEN
# reading order (a8 = 0). test_engine.py checks the two agree.
piece_code = {
  (chess.PAWN, chess.BLACK): 1, (chess.ROOK, chess.BLACK): 2,
  (chess.KNIGHT, chess.BLACK): 3, (chess.BISHOP, chess.BLACK): 4,
  (chess.QUEEN, chess.BLACK): 5, (chess.KING, chess.BLACK): 6,
  (chess.PAWN, chess.WHITE): 7, (chess.ROOK, chess.WHITE): 8,
  (chess.KNIGHT, chess.WHITE): 9, (chess.BISHOP, chess.WHITE): 10,
  (chess.QUEEN, chess.WHITE): 11, (chess.KING, chess.WHITE): 12,
}

def encode_board(board):
  out = [0] * 64
  for sq, piece in board.piece_map().items():
    out[sq ^ 56] = piece_code[(piece.piece_type, piece.color)]
  return out

class ChessModel(torch.nn.Module):
  def __init__(self, input_dimension=1088, feature=620):
    super().__init__()

    self.em_board = nn.Embedding(13, 16)
    self.em_piece = nn.Embedding(64, 64)
    self.f1 = nn.Linear(input_dimension, feature)
    self.f2 = nn.Linear(feature, feature)
    self.f3 = nn.Linear(feature, feature)
    self.f4 = nn.Linear(feature, feature)
    self.f5 = nn.Linear(feature, 64)
    self.gelu = nn.GELU()
    self.layer_norm = nn.LayerNorm(feature)

  def forward(self, board_in, piece_in):
    data = torch.concat((
      self.em_board(board_in.repeat(len(piece_in), 1)).flatten(1),
      self.em_piece(piece_in)
    ), 1)
    data = self.layer_norm(self.gelu(self.f1(data)))
    data = self.layer_norm(self.gelu(self.f2(data))) + data
    data = self.layer_norm(self.gelu(self.f3(data))) + data
    data = self.layer_norm(self.gelu(self.f4(data))) + data

    return self.f5(data)

class ChessFromModel(torch.nn.Module):
  def __init__(self, input_dimension=1024, feature=620):
    super().__init__()

    self.em_board = nn.Embedding(13, 16)
    self.f1 = nn.Linear(input_dimension, feature)
    self.f2 = nn.Linear(feature, feature)
    self.f3 = nn.Linear(feature, feature)
    self.f4 = nn.Linear(feature, feature)
    self.f5 = nn.Linear(feature, 128)
    self.gelu = nn.GELU()
    self.layer_norm = nn.LayerNorm(feature)

  def forward(self, board_in):
    data = self.em_board(board_in).flatten(1)
    data = self.layer_norm(self.gelu(self.f1(data)))
    data = self.layer_norm(self.gelu(self.f2(data))) + data
    data = self.layer_norm(self.gelu(self.f3(data))) + data
    data = self.layer_norm(self.gelu(self.f4(data))) + data

    return self.f5(data).reshape(2, 64)

# Changed from 7,2,2 since this seemed to work better
search_depth = 5
search_moves = 3
search_pieces = 4
move_overhead = 0.1
default_budget = 30.0

MATE = 1000000
infinity = 10000000

def load_model(path):
  # The checkpoints are whole pickled modules, so unpickling looks the classes up
  # in __main__ -- which works while run.py *is* __main__ and breaks the moment
  # anything imports it. Publishing them there first keeps both cases working,
  # and the forward that ends up running is still this file's (the one that
  # repeats the board across candidate source squares), not train5.py's.
  main = sys.modules['__main__']
  for cls in (ChessModel, ChessFromModel):
    if not hasattr(main, cls.__name__):
      setattr(main, cls.__name__, cls)
  # They were saved from a CUDA box, so they need map_location on a CPU-only
  # machine, and weights_only=False on torch 2.6+ where that flag defaults to True.
  try:
    return torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    return torch.load(path, map_location=device)

# One copy of each net, shared by every search thread. Both forwards are pure
# functions of their input and the weights -- no buffers are mutated -- so this
# is safe, and it drops startup and memory from twelve copies to one.
model = load_model('chess5.model').to(device).eval()
fmodel = load_model('chess_from.model').to(device).eval()

def best_moves(fen, side, movables, model, fmodel):
  from_out = fmodel(fen).tolist()[side]
  best_from = list(filter(
    lambda x: x in movables,
    sorted(
      list(range(64)),
      key=lambda i: from_out[i],
      reverse=True
    )
  ))
  to_out = model(
    fen,
    torch.tensor(best_from).to(device)
  ).tolist()
  best_to = list(map(
    lambda x: sorted(
      list(range(64)),
      key=lambda i: x[i],
      reverse=True
    ),
    to_out
  ))
  return (best_from, best_to)

def beam_size(board):
  """How wide to look. Endgames have few legal moves, so the same budget buys a
  shortlist that covers nearly all of them -- which is what stops the engine
  from walking past a promotion it never generated."""
  pieces = bin(board.occupied).count('1')
  if pieces <= 8:
    return (search_pieces + 2, search_moves + 2)
  if pieces <= 14:
    return (search_pieces + 1, search_moves + 1)
  if board.fullmove_number <= 10:
    # Nothing narrows the opening down for us, so give the nets a little more
    # room before the shortlist hardens into a single plan.
    return (search_pieces + 1, search_moves)
  return (search_pieces, search_moves)

def candidates(board, encoded, legals):
  """The nets' shortlist for this position, as legal moves, best first."""
  n_pieces, n_moves = beam_size(board)

  by_source = {}
  for m in legals:
    by_source.setdefault(m.from_square, []).append(m)

  side = 0 if board.turn == chess.WHITE else 1
  with torch.no_grad():
    board_t = torch.tensor([encoded]).to(device)
    ranked_from, ranked_to = best_moves(
      board_t, side, by_source.keys(), model, fmodel
    )
  ranked_from = ranked_from[:n_pieces]

  picked = []
  for n, src in enumerate(ranked_from):
    taken = 0
    for dest in ranked_to[n]:
      match = None
      for m in by_source[src]:
        # A promotion needs its piece filled in -- chess.Move(from, to) on its
        # own is not a legal move, which is why the engine could never queen a
        # pawn. python-chess lists the queen promotion first.
        if m.to_square == dest:
          match = m
          break
      if match is None:
        continue
      picked.append(match)
      # Underpromotion is almost always wrong, but a knight that arrives with
      # check is the one case worth a node.
      if match.promotion == chess.QUEEN:
        knight = chess.Move(match.from_square, dest, promotion=chess.KNIGHT)
        if board.gives_check(knight):
          picked.append(knight)
      taken += 1
      if taken >= n_moves:
        break
  return picked

def forced_mate(board, legals):
  """Mate in one, if there is one.

  The nets' shortlist is the engine's whole move-selection story, but it is a
  shortlist: in the endgame the mating move is regularly not in it, and an
  engine that walks past mate in one cannot convert anything. Only a check can
  be mate, so this costs a push per legal move and nothing else, next to a net
  forward pass that costs far more. It only ever adds a forced mate -- every
  other move still comes from the models."""
  for move in legals:
    if not board.gives_check(move):
      continue
    board.push(move)
    try:
      if board.is_checkmate():
        return move
    finally:
      board.pop()
  return None

def leaf_eval(board):
  # eval_pos takes encode()'s colour indices, where 1 is white and 0 is black,
  # and reports from that side's point of view. Scoring from the side to move
  # is what makes the negations below a plain negamax.
  return eval_pos(encode_board(board), 1 if board.turn == chess.WHITE else 0)

class Timeout(Exception):
  pass

def rep_key(board):
  # A repetition needs at least four plies with no capture and no pawn move, so
  # below that there is nothing to hash.
  if board.halfmove_clock < 4:
    return None
  return chess.polyglot.zobrist_hash(board)

def search(board, ply, limit, seen, deadline, stop):
  """Negamax over the nets' shortlist. Returns (move, score), score from the
  point of view of the side to move."""
  if stop.is_set() or time.time() > deadline:
    raise Timeout()

  legals = list(board.legal_moves)
  if not legals:
    # Mate scores shrink with distance from the root, so the engine takes the
    # quickest mate and the slowest loss. Scoring every mate the same is how a
    # won endgame turns into shuffling.
    return (None, -(MATE - ply) if board.is_checkmate() else 0)

  if board.is_insufficient_material() or board.halfmove_clock >= 100:
    return (None, 0)

  if ply >= limit:
    return (None, leaf_eval(board))

  mate = forced_mate(board, legals)
  if mate is not None:
    return (mate, MATE - (ply + 1))

  moves = candidates(board, encode_board(board), legals)
  if not moves:
    return (None, leaf_eval(board))

  best_move = None
  best_score = -infinity
  for move in moves:
    board.push(move)
    try:
      score = child_score(board, ply + 1, limit, seen, deadline, stop)
    finally:
      board.pop()
    if score > best_score:
      best_score = score
      best_move = move
  return (best_move, best_score)

def child_score(board, ply, limit, seen, deadline, stop):
  """Score of a position a move has just been pushed onto, from the point of
  view of the side that made the move."""
  key = rep_key(board)
  if key is not None and seen.get(key):
    # Already on the board once, so claiming it is a draw. Scoring it as one
    # keeps the engine from repeating away a win, and lets it hunt for the
    # repetition when it is worse.
    return 0
  if key is not None:
    seen[key] = seen.get(key, 0) + 1
  try:
    return -search(board, ply, limit, seen, deadline, stop)[1]
  finally:
    if key is not None:
      seen[key] -= 1
      if not seen[key]:
        del seen[key]

def history_keys(board):
  """Positions already reached in this game, so the search can see a repetition
  coming instead of walking into it.

  Every position gets hashed here, not just the ones past rep_key's halfmove
  cutoff: the *first* time a position appears its clock may be anything, and it
  is the second appearance that has to find it already in the map. (Repetitions
  that happen entirely inside one search line are only caught from the cutoff
  onwards, which at these depths is nearly all of them.)"""
  seen = {}
  replay = board.copy()
  stack = []
  while replay.move_stack:
    stack.append(replay.pop())

  def note():
    key = chess.polyglot.zobrist_hash(replay)
    seen[key] = seen.get(key, 0) + 1

  note()
  for move in reversed(stack):
    replay.push(move)
    note()
  return seen

def root_worker(board, limit, seen, deadline, stop, results, idx):
  try:
    results[idx] = child_score(board, 1, limit, seen, deadline, stop)
  except Timeout:
    results[idx] = None
  except Exception as e:
    send(f'info string search thread failed: {e}')
    results[idx] = None

def root_search(board, roots, limit, base, deadline, stop):
  """One iteration of the deepening loop. Returns (move, score), or None if the
  clock ran out before every root move had an answer."""
  results = [None] * len(roots)
  threads = []
  for i, move in enumerate(roots):
    child = board.copy()
    child.push(move)
    t = threading.Thread(
      target=root_worker,
      args=(child, limit, dict(base), deadline, stop, results, i)
    )
    t.start()
    threads.append(t)
  for t in threads:
    t.join()

  best = None
  for move, score in zip(roots, results):
    if score is None:
      # This root move never finished, so the iteration is not comparable.
      return None
    if best is None or score > best[1]:
      best = (move, score)
  return best

def find_best_move(board, limit=None, budget=None):
  """Iterative deepening. Each completed depth replaces the previous answer, so
  running out of time costs accuracy rather than the move."""
  limit = search_depth if limit is None else limit
  deadline = time.time() + budget if budget else time.time() + 86400
  stop = threading.Event()
  started = time.time()

  legals = list(board.legal_moves)
  if not legals:
    return None

  # The root builds its own move list rather than going through search(), so it
  # needs the same mate-in-one sweep.
  mate = forced_mate(board, legals)
  if mate is not None:
    send(f'info depth 1 score mate 1 pv {mate.uci()}')
    return (mate, MATE - 1)

  roots = candidates(board, encode_board(board), legals) or legals
  base = history_keys(board)

  # Something legal to play even if the very first iteration is cut short.
  best = (roots[0], 0)
  for depth in range(1, limit + 1):
    result = root_search(board, roots, depth, base, deadline, stop)
    if result is None:
      break
    best = result
    elapsed = time.time() - started
    send(f'info depth {depth} score {score_string(best[1])} '
         f'time {int(elapsed * 1000)} pv {best[0].uci()}')
    if abs(best[1]) > MATE - 1000:
      break
    # No point starting a depth we have no chance of finishing.
    if budget and elapsed > budget * 0.45:
      break
  return best

def score_string(score):
  if score > MATE - 1000:
    return f'mate {(MATE - score + 1) // 2}'
  if score < -(MATE - 1000):
    return f'mate {-((MATE + score + 1) // 2)}'
  return f'cp {score}'

def time_budget(board, params):
  """Seconds to spend on this move."""
  if 'movetime' in params:
    return max(0.02, params['movetime'] / 1000.0 - move_overhead)

  clock = 'wtime' if board.turn == chess.WHITE else 'btime'
  bonus = 'winc' if board.turn == chess.WHITE else 'binc'
  if clock not in params:
    # No clock to reason about. Still cap it: a wide endgame shortlist at the
    # full depth limit can otherwise run for minutes.
    return default_budget

  left = params[clock] / 1000.0
  inc = params.get(bonus, 0) / 1000.0
  togo = params.get('movestogo', 0)
  share = max(1, min(togo, 30)) if togo else 28

  budget = left / share + inc * 0.75
  return max(0.02, min(budget, left * 0.4) - move_overhead)

def send(line):
  # stdout is a pipe under lichess-bot, so it is block buffered unless we flush.
  sys.stdout.write(line + '\n')
  sys.stdout.flush()

board = chess.Board()
last_move = None

int_options = {
  'depth': 'search_depth',
  'analyzemoves': 'search_moves',
  'analyzepieces': 'search_pieces',
}

def set_option(cmd):
  global search_depth, search_moves, search_pieces, move_overhead
  m = re.match(r'^setoption\s+name\s+(.+?)\s+value\s+(.+)$', cmd, re.IGNORECASE)
  if not m:
    return
  name = m.group(1).strip().lower()
  value = m.group(2).strip()
  try:
    if name == 'move overhead':
      move_overhead = max(0.0, int(value) / 1000.0)
    elif name in int_options:
      globals()[int_options[name]] = max(1, int(value))
    else:
      return
  except ValueError:
    return
  send(f'info string {name} set to {value}')

def set_position(cmd):
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
  known = ('wtime', 'btime', 'winc', 'binc', 'movestogo', 'movetime', 'depth')
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

def do_go(cmd):
  global last_move
  args = cmd.split()
  params = parse_go(args)
  infinite = 'infinite' in args

  budget = None if infinite else time_budget(board, params)
  limit = params.get('depth', search_depth)

  result = find_best_move(board, limit=limit, budget=budget)
  if result is None or result[0] is None:
    send('info string no legal moves')
    send('bestmove (none)')
    return
  last_move = result[0]
  if not infinite:
    send(f'bestmove {last_move.uci()}')

def run():
  while True:
    try:
      cmd = input().strip()
    except EOFError:
      return
    if not cmd:
      continue

    if cmd == 'uci':
      send('id name NightyBot')
      send('id author Nighty')
      send(f'option name Depth type spin default {search_depth} min 1 max 32')
      send(f'option name AnalyzeMoves type spin default {search_moves} min 1 max 24')
      send(f'option name AnalyzePieces type spin default {search_pieces} min 1 max 16')
      send(f'option name Move Overhead type spin default {int(move_overhead * 1000)} min 0 max 5000')
      send('uciok')
    elif cmd == 'isready':
      send('readyok')
    elif cmd == 'ucinewgame':
      board.reset()
    elif cmd.startswith('position'):
      set_position(cmd)
    elif cmd.startswith('go'):
      do_go(cmd)
    elif cmd == 'stop':
      if last_move:
        send(f'bestmove {last_move.uci()}')
    elif cmd.startswith('setoption'):
      set_option(cmd)
    elif cmd == 'quit':
      return
    else:
      send('info string Unrecognized Command')

if __name__ == '__main__':
  run()
