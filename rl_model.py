"""The reinforcement-learning model: one net that looks at a position and
returns both a policy over moves and a win/draw/loss estimate. train_rl.py
trains it from self-play, run_rl.py plays with it.

It shares nothing with chess5.model / chess_from.model. Those are imitation
learned and saved as whole pickled modules (see CLAUDE.md); this one is saved
as a plain state_dict plus its constructor arguments, so the class below is the
only definition there is and loading does not care what __main__ is.
"""

import os

import chess
import torch
from torch import nn

# --- position encoding --------------------------------------------------------
#
# Everything is seen from the side to move. When Black is to move the board is
# flipped top to bottom and the colours are swapped, so the net only ever plays
# "White" and does not have to learn every pattern twice. Squares are
# python-chess numbered (a1 = 0) -- not encode()'s FEN order from run.py -- and
# `^ 56` is the flip.
#
# One token per square:
#   0       empty
#   1-6     our pawn, knight, bishop, rook, queen, king (chess.PAWN .. chess.KING)
#   7-12    their pawn .. king
#   13      the en passant target square, when the capture is actually legal
#   14, 15  our / their rook that still has its castling right
# plus one scalar, the fifty-move counter scaled to [0, 1].

N_TOKENS = 16
EP_TOKEN = 13
OUR_CASTLING_ROOK = 14
THEIR_CASTLING_ROOK = 15

def encode(board):
  """(tokens, halfmove) for one position, from the side to move's view."""
  us = board.turn
  flip = 0 if us == chess.WHITE else 56
  tokens = [0] * 64
  for color, base in ((us, 0), (not us, 6)):
    for piece in chess.PIECE_TYPES:
      for sq in chess.scan_forward(board.pieces_mask(piece, color)):
        tokens[sq ^ flip] = base + piece
  for sq in chess.scan_forward(board.clean_castling_rights()):
    ours = board.occupied_co[us] & chess.BB_SQUARES[sq]
    tokens[sq ^ flip] = OUR_CASTLING_ROOK if ours else THEIR_CASTLING_ROOK
  if board.ep_square is not None and board.has_legal_en_passant():
    tokens[board.ep_square ^ flip] = EP_TOKEN
  return tokens, min(board.halfmove_clock, 100) / 100.0

# --- move encoding ------------------------------------------------------------
#
# Also from the side to move's view. A move is from * 64 + to, which covers
# queen promotions as well; the 72 underpromotions (8 files x 3 directions x
# knight/bishop/rook) get slots of their own after that.

N_MOVES = 64 * 64 + 8 * 3 * 3
UNDERPROMOTION = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}

def move_index(move, turn):
  flip = 0 if turn == chess.WHITE else 56
  src = move.from_square ^ flip
  dst = move.to_square ^ flip
  under = UNDERPROMOTION.get(move.promotion)
  if under is None:
    return src * 64 + dst
  file = src & 7
  return 4096 + (file * 3 + (dst & 7) - file + 1) * 3 + under

# --- the net ------------------------------------------------------------------

class Block(nn.Module):
  def __init__(self, width):
    super().__init__()
    self.norm = nn.LayerNorm(width)
    self.f1 = nn.Linear(width, width)
    self.f2 = nn.Linear(width, width)
    self.gelu = nn.GELU()

  def forward(self, x):
    return x + self.f2(self.gelu(self.f1(self.norm(x))))

class RLNet(nn.Module):
  """Same shape of idea as the imitation nets -- embed every square, flatten,
  residual MLP -- with two heads on top: policy logits over N_MOVES, and
  win/draw/loss logits for the side to move, in that order."""

  def __init__(self, width=512, blocks=4, embedding=16):
    super().__init__()
    self.config = {'width': width, 'blocks': blocks, 'embedding': embedding}
    self.em_board = nn.Embedding(N_TOKENS, embedding)
    self.f_in = nn.Linear(64 * embedding + 1, width)
    self.blocks = nn.ModuleList(Block(width) for _ in range(blocks))
    self.norm = nn.LayerNorm(width)
    self.policy = nn.Linear(width, N_MOVES)
    self.value = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 3))
    self.gelu = nn.GELU()
    # An untrained net calls every position even. The policy head is left
    # randomly initialised on purpose: exactly uniform priors tie every move
    # at every node, argmax then always descends into the first legal move,
    # and self-play games shuffle one piece back and forth into a threefold
    # repetition. Slightly uneven random priors make the first games the
    # random walk they ought to be.
    nn.init.zeros_(self.value[-1].weight)
    nn.init.zeros_(self.value[-1].bias)

  def forward(self, tokens, halfmove):
    x = torch.cat((self.em_board(tokens).flatten(1), halfmove.unsqueeze(1)), 1)
    x = self.gelu(self.f_in(x))
    for block in self.blocks:
      x = block(x)
    x = self.norm(x)
    return self.policy(x), self.value(x)

def expected_score(wdl_logits):
  """Win minus loss probability, in [-1, 1]: the value the search backs up."""
  p = torch.softmax(wdl_logits.float(), 1)
  return p[:, 0] - p[:, 2]

class Evaluator:
  """numpy in, numpy out -- what the search calls to score a batch of leaves."""

  def __init__(self, net, device):
    self.net = net
    self.device = device

  def __call__(self, tokens, halfmove, indices):
    """indices is (batch, M): each row's legal move indices, padded with 0.
    Returns those moves' logits, (batch, M), and each position's value."""
    return self.fetch(self.submit(tokens, halfmove, indices))

  # GPU work is asynchronous until something reads the result, so a caller
  # with two batches can submit one, prepare the other, and only then fetch.
  @torch.inference_mode()
  def submit(self, tokens, halfmove, indices):
    logits, wdl = self.net(
      torch.from_numpy(tokens).to(self.device),
      torch.from_numpy(halfmove).to(self.device)
    )
    # Pull out the legal moves' logits before leaving the device: it is ~30
    # numbers per row, against 4168 for the whole head.
    picked = logits.gather(1, torch.from_numpy(indices).to(self.device))
    return picked.float(), expected_score(wdl)

  @torch.inference_mode()
  def fetch(self, handle):
    picked, value = handle
    return picked.cpu().numpy(), value.cpu().numpy()

# --- devices and files --------------------------------------------------------

def pick_device(name='auto'):
  if name != 'auto':
    return name
  if torch.cuda.is_available():
    return 'cuda'
  if torch.backends.mps.is_available():
    return 'mps'
  return 'cpu'

def torch_load(path, device='cpu'):
  # Plain tensors and numbers only, so weights_only is safe, and map_location
  # lets a file written on a CUDA box load anywhere.
  try:
    return torch.load(path, map_location=device, weights_only=True)
  except TypeError:  # torch < 1.13 has no weights_only
    return torch.load(path, map_location=device)

def cpu_state(module):
  return {k: v.detach().cpu() for k, v in module.state_dict().items()}

def save_net(net, path):
  # Write then rename, so an engine or evaluator reading the file never sees
  # half of it.
  tmp = path + '.tmp'
  torch.save({'config': net.config, 'state_dict': cpu_state(net)}, tmp)
  os.replace(tmp, path)

def load_net(path, device='cpu'):
  data = torch_load(path, device)
  net = RLNet(**data['config'])
  net.load_state_dict(data['state_dict'])
  return net.to(device)
