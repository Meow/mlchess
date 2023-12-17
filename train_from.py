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
pgn = open("/home/luna/Downloads/lichess_db_standard_rated_2023-10.pgn")


def encode(fen):
  fen = fen.split(" ")[0].replace("/", "")
  for n in range(1, 9):
    fen = fen.replace(str(n), "." * n)
  return list(map(".prnbqkPRNBQK".index, fen))

def random_variation(game):
  headers = game.headers
  if (int(headers['WhiteElo']) + int(headers['BlackElo'])) / 2.0 < 1900.0:
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

    return self.f5(data)

print("intitializing training...")

model = torch.load('chess_from.model') if os.path.exists('chess_from.model') else ChessFromModel()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss = torch.nn.CrossEntropyLoss(reduction='none')
v_random_variation = np.vectorize(random_variation)

# this is actually faster than using list functions like map, filter, etc
def generate_data(n = 16):
  boards = []
  froms = []
  plys = []

  for v in (item for item in [random_variation(chess.pgn.read_game(pgn)) for _ in range(1, n)] if item):
    boards.append(variation_to_board(v.parent))
    froms.append(v.move.from_square)
    plys.append(0 if v.parent.turn() == chess.WHITE else 1)

  return (boards, froms, plys)

def train():
  loss_log = []
  step = 0
  while True:
    data = generate_data()
    while len(data[0]) == 0:
      data = generate_data()
    output = model(torch.tensor(data[0]).to(device))
    player = torch.tensor(data[2]).to(device)
    output_white = output[:, :64]
    output_black = output[:, 64:]
    ideal_piece = torch.tensor(data[1]).to(device)
    l = (
      (player == 0) * loss(output_white, ideal_piece) +
      (player == 1) * loss(output_black, ideal_piece)
    ).mean()
    loss_log.append(l.detach().to('cpu'))
    #wandb.log({'loss': l.item(), 'epoch': step})
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
      torch.save(model, 'chess_from.model')
    step += 1

print("beginning training...")

train()
