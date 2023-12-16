import os
import torch
import numpy as np
import chess.pgn
import random
from safetensors.torch import load_file, save_file
from torch import nn
# import wandb

# wandb.init(project='mlchess')

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'using {device} for training')

print("lewding data...")
pgn = open("/home/luna/Downloads/lichess_db_standard_rated_2023-09.pgn")


def encode(fen):
  fen = fen.split(" ")[0].replace("/", "")
  for n in range(1, 9):
    fen = fen.replace(str(n), "." * n)
  return list(map(".prnbqkPRNBQK".index, fen))

def random_variation(game):
  headers = game.headers
  if (int(headers['WhiteElo']) + int(headers['BlackElo'])) / 2.0 < 1600.0:
    return None
  winner = 1 if headers['Result'] == '1-0' else 0
  game_list = list(game.mainline())
  total_variations = len(game_list)
  if total_variations < 6:
    return None
  variation = random.randint(0, total_variations - 1)
  pick = game_list[variation]
  while variation % 2 != winner or pick.parent == None:
    variation = random.randint(0, total_variations - 1)
    pick = game_list[variation]
  return pick

def variation_to_board(variation):
  return encode(variation.board().fen())


class ChessModel(torch.nn.Module):
  def __init__(self, input_dimension=1088, feature=512):
    super().__init__()

    self.em_board = nn.Embedding(13, 16)
    self.em_piece = nn.Embedding(64, 64)
    self.f1 = nn.Linear(input_dimension, feature)
    self.f2 = nn.Linear(feature, feature)
    self.f3 = nn.Linear(feature, feature)
    self.f4 = nn.Linear(feature, 64)
    self.gelu = nn.GELU()
    self.layer_norm = nn.LayerNorm(feature)

  def forward(self, board_in, piece_in):
    data = torch.concat((
      self.em_board(board_in).flatten(1),
      self.em_piece(piece_in)
    ), 1)
    data = self.layer_norm(self.gelu(self.f1(data)))
    data = self.layer_norm(self.gelu(self.f2(data))) + data
    data = self.layer_norm(self.gelu(self.f3(data))) + data

    return self.f4(data)

print("intitializing training...")

model = torch.load('chess.model') if os.path.exists('chess.model') else ChessModel()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss = torch.nn.CrossEntropyLoss()
v_random_variation = np.vectorize(random_variation)

def generate_data(n = 16):
  boards = []
  froms = []
  tos = []

  for v in (item for item in [random_variation(chess.pgn.read_game(pgn)) for _ in range(1, n)] if item):
    boards.append(variation_to_board(v.parent))
    froms.append(v.move.from_square)
    tos.append(v.move.to_square)

  return (boards, froms, tos)

# def generate_data_old(n = 8):
#   variations = list(
#     filter(
#       lambda v: v != None,
#       v_random_variation([chess.pgn.read_game(pgn) for _ in range(1, n)])
#     )
#   )
#   return (
#     list(map(lambda x: variation_to_board(x.parent), variations)),
#     list(map(lambda x: x.move.from_square, variations)),
#     list(map(lambda x: x.move.to_square, variations))
#   )

def train():
  loss_log = []
  step = 0
  while True:
    data = generate_data()
    while len(data[0]) == 0:
      data = generate_data()
    output = model(
      torch.tensor(data[0]).to(device),
      torch.tensor(data[1]).to(device)
    )
    ideal_move = torch.tensor(data[2]).to(device)
    l = loss(output, ideal_move)
    loss_log.append(l.detach().to('cpu'))
    # wandb.log({'loss': l.item(), 'epoch': step})
    l.backward()
    if step % 500 == 0:
      print(step // 500, np.mean(loss_log), np.std(loss_log), max(loss_log), min(loss_log))
      # wandb.log({
      #   'mean': np.mean(loss_log).item(),
      #   'std': np.std(loss_log).item(),
      #   'max': max(loss_log),
      #   'min': min(loss_log)
      # })
      loss_log = []
      optimizer.step()
      optimizer.zero_grad()

    if step > 100 and step % 10000 == 0:
      print("autosaving model...")
      torch.save(model, 'chess.model')
    step += 1

print("beginning training...")

train()
