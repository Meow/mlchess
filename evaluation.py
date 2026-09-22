# Largely stolen and adapted from
# https://github.com/algerbrex/blunder/blob/a5e1f1bfe1958f87023e62d9e04a6e54fc7df5b2/engine/evaluation.go
# This uses the MIT License
# Go read it here: https://github.com/algerbrex/blunder/blob/a5e1f1bfe1958f87023e62d9e04a6e54fc7df5b2/LICENSE

phase_vals = [
  0, 1, 1, 2, 4, 0
]
total_phase = 24
midgame_piece_pos = [
  [ # pawns
    100, 100, 100, 100, 100, 100, 100, 100,
    176, 214, 147, 194, 189, 214, 132, 77,
    82, 88, 106, 113, 150, 146, 110, 73,
    67, 93, 83, 95, 97, 92, 99, 63,
    55, 74, 80, 89, 94, 86, 90, 55,
    55, 70, 68, 69, 76, 81, 101, 66,
    52, 84, 66, 60, 69, 99, 117, 60,
    100, 100, 100, 100, 100, 100, 100, 100,
  ],
  [ # knights
    116, 228, 271, 270, 338, 213, 278, 191,
    225, 247, 353, 331, 321, 360, 300, 281,
    258, 354, 343, 362, 389, 428, 375, 347,
    300, 332, 325, 360, 349, 379, 339, 333,
    298, 322, 325, 321, 337, 332, 332, 303,
    287, 297, 316, 319, 327, 320, 327, 294,
    276, 259, 300, 304, 308, 322, 296, 292,
    208, 290, 257, 274, 296, 284, 293, 284,
  ],
  [ # bishops
    292, 338, 254, 283, 299, 294, 337, 323,
    316, 342, 319, 319, 360, 385, 343, 295,
    342, 377, 373, 374, 368, 392, 385, 363,
    332, 338, 356, 384, 370, 380, 337, 341,
    327, 354, 353, 366, 373, 346, 345, 341,
    335, 350, 351, 347, 352, 361, 350, 344,
    333, 354, 354, 339, 344, 353, 367, 333,
    309, 341, 342, 325, 334, 332, 302, 313,
  ],
  [ # rooks
    493, 511, 487, 515, 514, 483, 485, 495,
    493, 498, 529, 534, 546, 544, 483, 508,
    465, 490, 499, 497, 483, 519, 531, 480,
    448, 464, 476, 495, 484, 506, 467, 455,
    442, 451, 468, 470, 476, 472, 498, 454,
    441, 461, 468, 465, 478, 481, 478, 452,
    443, 472, 467, 476, 483, 500, 487, 423,
    459, 463, 470, 479, 480, 480, 446, 458,
  ],
  [ # queen
    865, 902, 922, 911, 964, 948, 933, 928,
    886, 865, 903, 921, 888, 951, 923, 940,
    902, 901, 907, 919, 936, 978, 965, 966,
    881, 885, 897, 894, 898, 929, 906, 915,
    907, 884, 899, 896, 904, 906, 912, 911,
    895, 916, 900, 902, 904, 912, 924, 917,
    874, 899, 918, 908, 915, 924, 911, 906,
    906, 899, 906, 918, 898, 890, 878, 858,
  ],
  [ # king
    -11, 70, 55, 31, -37, -16, 22, 22,
    37, 24, 25, 36, 16, 8, -12, -31,
    33, 26, 42, 11, 11, 40, 35, -2,
    0, -9, 1, -21, -20, -22, -15, -60,
    -25, 16, -27, -67, -81, -58, -40, -62,
    7, -2, -37, -77, -79, -60, -23, -26,
    12, 15, -13, -72, -56, -28, 15, 17,
    -6, 44, 29, -58, 8, -25, 34, 28
  ]
]
endgame_piece_pos = [
  [ # pawns
    100, 100, 100, 100, 100, 100, 100, 100,
    277, 270, 252, 229, 240, 233, 264, 285,
    190, 197, 182, 168, 155, 150, 180, 181,
    128, 117, 108, 102, 93, 100, 110, 110,
    107, 101, 89, 85, 86, 83, 92, 91,
    96, 96, 85, 92, 88, 83, 85, 82,
    107, 99, 97, 97, 100, 89, 89, 84,
    100, 100, 100, 100, 100, 100, 100, 100,
  ],
  [ # knights
    229, 236, 269, 250, 257, 249, 219, 188,
    252, 274, 263, 281, 273, 258, 260, 229,
    253, 264, 290, 289, 278, 275, 263, 243,
    267, 280, 299, 301, 299, 293, 285, 264,
    263, 273, 293, 301, 296, 293, 284, 261,
    258, 276, 278, 290, 287, 274, 260, 255,
    241, 259, 270, 277, 276, 262, 260, 237,
    253, 233, 258, 264, 261, 260, 234, 215,
  ],
  [ # bishops
    288, 278, 287, 292, 293, 290, 287, 277,
    289, 294, 301, 288, 296, 289, 294, 281,
    292, 289, 296, 292, 296, 300, 296, 293,
    293, 302, 305, 305, 306, 302, 296, 297,
    289, 293, 304, 308, 298, 301, 291, 288,
    285, 294, 304, 303, 306, 294, 290, 280,
    285, 284, 291, 299, 300, 290, 284, 271,
    277, 292, 286, 295, 294, 288, 290, 285,
  ],
  [ # rooks
    506, 500, 508, 502, 504, 507, 505, 503,
    505, 506, 502, 502, 491, 497, 506, 501,
    504, 503, 499, 500, 500, 495, 496, 496,
    503, 502, 510, 500, 502, 504, 500, 505,
    505, 509, 509, 506, 504, 503, 496, 495,
    500, 503, 500, 505, 498, 498, 499, 489,
    496, 495, 502, 505, 498, 498, 491, 499,
    492, 497, 498, 496, 493, 493, 497, 480,
  ],
  [ # queen
    918, 937, 943, 945, 934, 926, 924, 942,
    907, 945, 946, 951, 982, 933, 928, 912,
    896, 921, 926, 967, 963, 937, 924, 915,
    926, 944, 939, 962, 983, 957, 981, 950,
    893, 949, 942, 970, 952, 956, 953, 936,
    911, 892, 933, 928, 934, 942, 934, 924,
    907, 898, 883, 903, 903, 893, 886, 888,
    886, 887, 890, 872, 916, 890, 906, 879,
  ],
  [ # king
    -74, -43, -23, -25, -11, 10, 1, -12,
    -18, 6, 4, 9, 7, 26, 14, 8,
    -3, 6, 10, 6, 8, 24, 27, 3,
    -16, 8, 13, 20, 14, 19, 10, -3,
    -25, -14, 13, 20, 24, 15, 1, -15,
    -27, -10, 9, 20, 23, 14, 2, -12,
    -32, -17, 4, 14, 15, 5, -10, -22,
    -55, -40, -23, -6, -20, -8, -28, -47,
  ],
]
flip_list = [
  [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
  ],
  [
    56, 57, 58, 59, 60, 61, 62, 63,
    48, 49, 50, 51, 52, 53, 54, 55,
    40, 41, 42, 43, 44, 45, 46, 47,
    32, 33, 34, 35, 36, 37, 38, 39,
    24, 25, 26, 27, 28, 29, 30, 31,
    16, 17, 18, 19, 20, 21, 22, 23,
    8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7,
  ],
]

# --- how the encoded board lines up with the tables above ------------------
#
# `board` is what encode() produces: 64 ints in FEN reading order, so index 0
# is a8 and index 63 is h1, with 1-6 = black p,r,n,b,q,k and 7-12 = white.
# Two things about that do not line up with blunder's tables, and both used to
# be indexed straight through:
#
#  * The tables are in blunder's order (pawn, knight, bishop, rook, queen,
#    king), but the encoding alphabet is ".prnbqk". Reading table[piece - 1]
#    scored every rook off the knight table and every bishop off the rook
#    table, so the engine thought a bishop outweighed a rook.
#  * The tables are written from White's side with a8 first, which is already
#    the orientation encode() hands us. White therefore reads them straight and
#    only Black's squares need mirroring -- blunder mirrors White instead
#    because its own squares are a1-first. Mirroring the wrong colour turned
#    every table upside down for both sides: a pawn one square from queening
#    scored less than one still on its starting square, and the midgame king
#    table pulled the king up the board instead of into the corner.

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)

# ".prnbqk" order -> table order
piece_to_table = [PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING]

# indexed by colour: 0 = black (mirrored), 1 = white (as written)
pov_list = [flip_list[1], flip_list[0]]

material_vals = [100, 320, 330, 500, 900, 0]

# --- extra terms the piece-square tables do not cover ----------------------
# Indexed by relative rank: 0 is the pawn's own back rank, 6 is one step from
# queening. Endgame numbers are deliberately large -- a protected passer is
# usually the whole point of a pawn endgame.
passed_mg = [0, 2, 6, 14, 28, 52, 88, 0]
passed_eg = [0, 10, 22, 42, 78, 130, 200, 0]

doubled_mg, doubled_eg = -9, -22
isolated_mg, isolated_eg = -14, -12
bishop_pair_mg, bishop_pair_eg = 24, 44
rook_open_mg, rook_open_eg = 24, 10
rook_semi_mg, rook_semi_eg = 11, 5
rook_seventh_mg, rook_seventh_eg = 16, 24
shield_missing_mg = -16
king_open_file_mg = -18
undeveloped_mg = -13
early_queen_mg = -28
blocked_center_pawn_mg = -16

# home squares, as encode() indexes them (0 = a8)
home_minor = [
  {KNIGHT: (1, 6), BISHOP: (2, 5)},      # black b8/g8, c8/f8
  {KNIGHT: (57, 62), BISHOP: (58, 61)},  # white b1/g1, c1/f1
]
home_queen = [3, 59]                      # d8, d1
home_center_pawns = [(11, 12), (51, 52)]  # d7/e7, d2/e2
forward_step = [8, -8]                    # one rank towards the enemy

center_distance = [
  6, 5, 4, 3, 3, 4, 5, 6,
  5, 4, 3, 2, 2, 3, 4, 5,
  4, 3, 2, 1, 1, 2, 3, 4,
  3, 2, 1, 0, 0, 1, 2, 3,
  3, 2, 1, 0, 0, 1, 2, 3,
  4, 3, 2, 1, 1, 2, 3, 4,
  5, 4, 3, 2, 2, 3, 4, 5,
  6, 5, 4, 3, 3, 4, 5, 6,
]

def sq_file(i):
  return i & 7

def sq_rank(i):
  # 0 = rank 1, 7 = rank 8
  return 7 - (i >> 3)

def rel_rank(color, i):
  # 0 = the colour's own back rank, 7 = the rank it promotes on
  r = sq_rank(i)
  return r if color == 1 else 7 - r

def manhattan(a, b):
  return abs(sq_file(a) - sq_file(b)) + abs(sq_rank(a) - sq_rank(b))

def eval_pos(board, side = 0):
  other_side = 1 if side == 0 else 0

  mg = [0, 0]
  eg = [0, 0]
  material = [0, 0]
  phase = total_phase

  pawn_ranks = [[[] for _ in range(8)], [[] for _ in range(8)]]
  pawns = [[], []]
  rooks = [[], []]
  bishops = [0, 0]
  minors = [0, 0]
  at_home = [0, 0]
  queens = [0, 0]
  queen_out = [0, 0]
  kings = [None, None]

  for i in range(64):
    piece = board[i]
    if piece == 0:
      continue
    color = 0 if piece < 7 else 1
    kind = piece_to_table[(piece - 1) if piece < 7 else (piece - 7)]
    sq = pov_list[color][i]

    mg[color] += midgame_piece_pos[kind][sq]
    eg[color] += endgame_piece_pos[kind][sq]
    material[color] += material_vals[kind]
    phase -= phase_vals[kind]

    if kind == PAWN:
      pawns[color].append(i)
      pawn_ranks[color][sq_file(i)].append(sq_rank(i))
    elif kind == KING:
      kings[color] = i
    elif kind == ROOK:
      rooks[color].append(i)
    elif kind == QUEEN:
      queens[color] += 1
      if i != home_queen[color]:
        queen_out[color] += 1
    else:
      minors[color] += 1
      if kind == BISHOP:
        bishops[color] += 1
      if i in home_minor[color][kind]:
        at_home[color] += 1

  # Bare kings, or a lone minor that cannot mate, is a dead draw however nice
  # the piece-square tables think the squares are.
  if (not pawns[0] and not pawns[1]
      and not rooks[0] and not rooks[1]
      and not queens[0] and not queens[1]
      and material[0] <= 330 and material[1] <= 330):
    return 0

  for color in (0, 1):
    enemy = 1 if color == 0 else 0
    ahead = (lambda er, r: er > r) if color == 1 else (lambda er, r: er < r)

    for i in pawns[color]:
      f = sq_file(i)
      r = sq_rank(i)
      rel = rel_rank(color, i)

      if len(pawn_ranks[color][f]) > 1:
        mg[color] += doubled_mg
        eg[color] += doubled_eg

      if not ((f > 0 and pawn_ranks[color][f - 1])
              or (f < 7 and pawn_ranks[color][f + 1])):
        mg[color] += isolated_mg
        eg[color] += isolated_eg

      passed = True
      for ff in (f - 1, f, f + 1):
        if ff < 0 or ff > 7:
          continue
        for er in pawn_ranks[enemy][ff]:
          if ahead(er, r):
            passed = False
            break
        if not passed:
          break
      # a pawn stuck behind one of its own is not going anywhere either
      if passed:
        for orr in pawn_ranks[color][f]:
          if ahead(orr, r):
            passed = False
            break

      if passed:
        mg[color] += passed_mg[rel]
        eg[color] += passed_eg[rel]
        # Whose king gets to the queening square first usually decides it.
        if rel >= 3 and kings[color] is not None and kings[enemy] is not None:
          promo = sq_file(i) if color == 1 else 56 + sq_file(i)
          eg[color] += 7 * manhattan(kings[enemy], promo)
          eg[color] -= 5 * manhattan(kings[color], promo)

    if bishops[color] >= 2:
      mg[color] += bishop_pair_mg
      eg[color] += bishop_pair_eg

    for i in rooks[color]:
      f = sq_file(i)
      if not pawn_ranks[color][f]:
        if not pawn_ranks[enemy][f]:
          mg[color] += rook_open_mg
          eg[color] += rook_open_eg
        else:
          mg[color] += rook_semi_mg
          eg[color] += rook_semi_eg
      if rel_rank(color, i) == 6:
        mg[color] += rook_seventh_mg
        eg[color] += rook_seventh_eg

    # --- opening and king safety, midgame only ---------------------------
    at_home_count = at_home[color]
    mg[color] += undeveloped_mg * at_home_count

    # A queen that comes out before the minors do just gets chased around.
    if queen_out[color] and at_home_count >= 2:
      mg[color] += early_queen_mg

    # The classic "bishop parked in front of its own d/e pawn".
    step = forward_step[color]
    own_pawn = 7 if color == 1 else 1
    for p in home_center_pawns[color]:
      if board[p] == own_pawn and board[p + step] != 0:
        mg[color] += blocked_center_pawn_mg

    k = kings[color]
    if k is not None and rel_rank(color, k) <= 1:
      kf = sq_file(k)
      kr = sq_rank(k)
      shield = 0
      for ff in (kf - 1, kf, kf + 1):
        if ff < 0 or ff > 7:
          continue
        covered = False
        for pr in pawn_ranks[color][ff]:
          if 1 <= (pr - kr if color == 1 else kr - pr) <= 2:
            covered = True
            break
        if covered:
          shield += 1
        elif not pawn_ranks[color][ff]:
          mg[color] += king_open_file_mg
      mg[color] += shield_missing_mg * (3 - shield)

  # --- mop-up: with a decisive edge and no enemy pawns, walk their king to
  # the edge and bring yours in, otherwise a won K+Q/K+R ending just shuffles.
  mop_up = 0
  for color in (0, 1):
    enemy = 1 if color == 0 else 0
    if (material[color] - material[enemy] >= 450 and not pawns[enemy]
        and kings[color] is not None and kings[enemy] is not None):
      bonus = (47 * center_distance[kings[enemy]]
               + 16 * (14 - manhattan(kings[color], kings[enemy]))) // 10
      mop_up += bonus if color == 1 else -bonus

  mg_total = mg[side] - mg[other_side]
  eg_total = eg[side] - eg[other_side]
  if side == 1:
    eg_total += mop_up
  else:
    eg_total -= mop_up

  phase = (phase * 256 + total_phase // 2) // total_phase

  # Truncate towards zero rather than flooring, so that eval(pos) == -eval(mirrored pos)
  # exactly. Flooring biases every negative score down by one centipawn, which is
  # harmless on its own but makes the search's negamax negations slightly lopsided.
  total = mg_total * (256 - phase) + eg_total * phase
  return total // 256 if total >= 0 else -((-total) // 256)
