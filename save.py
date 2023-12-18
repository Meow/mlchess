import torch
from safetensors.torch import save_file
from evaluation import eval_pos

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

model = torch.load('chess5.model').to('cpu')
fmodel = torch.load('chess_from.model').to('cpu')

print('exporting chess5.model')

weights = {}

for name, layer in model._modules.items():
    if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm)):
        weights[f'{name}_weight'] = layer.weight
        weights[f'{name}_bias'] = layer.bias
    elif isinstance(layer, torch.nn.Embedding):
        weights[name] = layer.weight

save_file(weights, 'chess.safetensors')

print('exporting chess_from.model')

weights = {}

for name, layer in fmodel._modules.items():
    if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm)):
        weights[f'{name}_weight'] = layer.weight
        weights[f'{name}_bias'] = layer.bias
    elif isinstance(layer, torch.nn.Embedding):
        weights[name] = layer.weight

save_file(weights, 'chess_from.safetensors')
