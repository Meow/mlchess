import os
import torch
import numpy as np
import chess
import re
from safetensors.torch import load_file, save_file
from evaluation import eval_pos

device = "cuda" if torch.cuda.is_available() else "cpu"

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

model = torch.load('chess5.model').to(device)
fmodel = torch.load('chess_from.model').to(device)
search_depth = 7
search_moves = 2
search_pieces = 2

def best_moves(fen, side, movables):
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

def ensure_legal(out, legals, idx=0, depth=1):
  ok = 0
  i = 0
  move = None
  while ok < depth:
    move = chess.Move(out[0][idx], out[1][idx][i])
    if move in legals:
      ok += 1
    if i == 63:
      return None
    i += 1
  return move

def flip_idx(idx):
  if idx == 0:
    return 1
  return 0

def is_enemy(depth):
  return depth % 2 == 0

nodes = 0

def find_best_move(board, depth=1, prev_highest=-99999999):
  global nodes

  if depth == 1:
    nodes = 0
  elif depth > search_depth:
    return None

  encoded = encode(board.fen())
  legals = board.legal_moves
  movable_pieces = list(set([m.from_square for m in legals]))
  side = 0 if board.turn == chess.WHITE else 1
  if len(movable_pieces) < 1:
    return None

  score = eval_pos(encoded, side)
  out = best_moves(torch.tensor([encoded]).to(device), side, movable_pieces)

  best = (0, 0)
  best_branch = (0, 0)
  highest = prev_highest

  if depth <= search_depth:
    for i in range(len(out[0])):
      if i >= search_pieces:
        break
      for i2 in range(search_moves):
        nodes += 1
        move = ensure_legal(out, legals, i, i2 + 1)
        if move == None:
          continue
        board.push(move)
        this_eval = eval_pos(encode(board.fen()), side)
        if this_eval > score:
          best = (i, i2 + 1)
          score = this_eval
        res = find_best_move(board, depth + 1, highest)
        if res:
          if depth == search_depth:
            if res[1] > highest:
              highest = res[1]
              print(f'info string Hit deepest branch, eval: {highest}')
          elif depth == 1:
            print(f'info string Propagating highest to root branch')
            if res[2] > highest:
              highest = res[2]
              best_branch = (i, i2 + 1)
              print(f'info string Best branch is now {best_branch}')
          else:
            if res[2] > highest:
              highest = res[2]
            elif highest == -99999999:
              highest = res[1]
        elif highest == -99999999:
          highest = prev_highest
        board.pop()
  
  picked_move = None

  if depth == 1:
    print(f'info string Highest Eval: {highest}')
    picked_move = ensure_legal(out, legals, best_branch[0], best_branch[1])
    if picked_move == None:
      print("info string Warning: Playing suboptimal move!")
      picked_move = ensure_legal(out, legals, best[0], best[1])
      if picked_move == None:
        print("info string Warning: Playing first possible move.")
        picked_move = ensure_legal(out, legals, 0, 1)
  else:
    picked_move = ensure_legal(out, legals, best[0], best[1])

  if not picked_move:
    return None

  board.push(picked_move)
  encoded = encode(board.fen())
  board.pop()

  # if depth == 1:
  #   print("")

  return (picked_move, eval_pos(encoded, side if is_enemy(depth) else flip_idx(side)), highest)

board = chess.Board()
opponent_move = None

def run():
  cmd = input()
  if cmd == 'uci':
    print('id name NightyBot\nid author Nighty')
    print('option name Depth type spin default 6 min 1 max 32')
    print('option name AnalyzeMoves type spin default 2 min 1 max 24')
    print('option name AnalyzePieces type spin default 3 min 1 max 16')
    print('option name Move Overhead type spin default 100 min 1 max 1000')
    print('option name Threads type spin default 1 min 1 max 64')
    print('option name Hash type spin default 4096 min 1 max 999999999')
    print('option name SyzygyPath type string default')
    print('option name UCI_ShowWDL type check default true')
    print('uciok')
  elif cmd == 'isready':
    print('readyok')
  elif cmd == 'ucinewgame':
    board.reset()
  elif cmd.startswith('position'):
    m = re.findall('^position\s+(.+)\s+moves\s+(.+)$', cmd)
    if len(m) != 1:
      return run() 
    if m[0][0] == 'startpos':
      board.reset()
    else:
      board.set_fen(m[0][0])
    for x in m[0][1].split():
      board.push_uci(x)
  elif cmd.startswith('go'):
    cmd = cmd.split()
    opponent_move = find_best_move(board)
    if cmd[1] != 'infinite':
      print(f'bestmove {opponent_move[0].uci()}')
  elif cmd == 'stop':
    print(f'bestmove {opponent_move[0].uci()}')
  elif cmd.startswith('setoption'):
    search_depth = 6
    # placeholder
  elif cmd == 'quit':
    exit()
  else:
    print('info string Unrecognized Command')
  run()

run()

# res = model(
#   torch.tensor([encode('2r2k2/4ppb1/3p4/3Pn2p/P3PN2/1r3P2/1PR5/1KBq4 w - - 0 33')]),
#   torch.tensor([sqr_to_num('c2')])
# ).flatten()
# decodings = [num_to_sqr(n) for n in range(64)]
# res_list = sorted(
#   list(range(64)),
#   key=lambda i: res[i],
#   reverse=True
# )
# print(list(map(lambda i: decodings[i], res_list)))
