import os
import torch
import numpy as np
import chess
from safetensors.torch import load_file, save_file

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

model = torch.load('chess_from.model').to('cpu')
out = model(
  torch.tensor([encode('r4rk1/1bpq1p1p/3p1p2/pp6/8/3QnN1P/PP3PP1/RB4K1 w - - 0 20')])
).tolist()
whites = sorted(
  list(range(64)),
  key=lambda i: out[0][i],
  reverse=True
)
blacks = sorted(
  list(range(64)),
  key=lambda i: out[1][i],
  reverse=True
)

print(list(map(lambda x: num_to_sqr(x), whites[:5])))
print(list(map(lambda x: num_to_sqr(x), blacks[:5])))

# res_list = sorted(
#   list(range(64)),
#   key=lambda i: res[i],
#   reverse=True
# )
# print(list(map(lambda i: decodings[i], res_list)))
