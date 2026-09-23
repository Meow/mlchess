//! A small classical engine for the opponent pool: evaluation.py's
//! piece-square evaluation, ported term for term, under a shallow alpha-beta.
//! No quiescence search and no lookahead to speak of -- it is there to be a
//! different kind of opponent (materialist, blind past its horizon), not a
//! strong one. The tables come from evaluation.py via gen_tables.py, and
//! test_rl.py checks `evaluate` against `eval_pos` on random positions.
//!
//! The evaluation works on the same board layout as the Python: 64 codes in
//! FEN reading order (index 0 = a8), 1-6 black p r n b q k, 7-12 white.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Move, Position, Role};

use crate::tables::{CENTER_DISTANCE, ENDGAME, MATERIAL, MIDGAME, PASSED_EG, PASSED_MG, PHASE_VALS};

const PAWN: usize = 0;
const KNIGHT: usize = 1;
const BISHOP: usize = 2;
const ROOK: usize = 3;
const QUEEN: usize = 4;
const KING: usize = 5;
const TOTAL_PHASE: i32 = 24;
// ".prnbqk" order -> table order
const PIECE_TO_TABLE: [usize; 6] = [PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING];

const DOUBLED: (i32, i32) = (-9, -22);
const ISOLATED: (i32, i32) = (-14, -12);
const BISHOP_PAIR: (i32, i32) = (24, 44);
const ROOK_OPEN: (i32, i32) = (24, 10);
const ROOK_SEMI: (i32, i32) = (11, 5);
const ROOK_SEVENTH: (i32, i32) = (16, 24);
const SHIELD_MISSING_MG: i32 = -16;
const KING_OPEN_FILE_MG: i32 = -18;
const UNDEVELOPED_MG: i32 = -13;
const EARLY_QUEEN_MG: i32 = -28;
const BLOCKED_CENTER_PAWN_MG: i32 = -16;

// home squares, FEN order (0 = a8); [black, white]
const HOME_KNIGHT: [(usize, usize); 2] = [(1, 6), (57, 62)];
const HOME_BISHOP: [(usize, usize); 2] = [(2, 5), (58, 61)];
const HOME_QUEEN: [usize; 2] = [3, 59];
const HOME_CENTER_PAWNS: [(usize, usize); 2] = [(11, 12), (51, 52)];
const FORWARD_STEP: [i32; 2] = [8, -8];

pub const MATE: i32 = 1_000_000;

fn sq_file(i: usize) -> i32 {
    (i & 7) as i32
}

fn sq_rank(i: usize) -> i32 {
    7 - (i >> 3) as i32
}

fn rel_rank(color: usize, i: usize) -> usize {
    let r = sq_rank(i);
    (if color == 1 { r } else { 7 - r }) as usize
}

fn manhattan(a: usize, b: usize) -> i32 {
    (sq_file(a) - sq_file(b)).abs() + (sq_rank(a) - sq_rank(b)).abs()
}

/// evaluation.py's board: FEN order, ".prnbqkPRNBQK" codes.
pub fn fen_order(pos: &Chess) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (sq, piece) in pos.board().clone() {
        let code = match piece.role {
            Role::Pawn => 1,
            Role::Rook => 2,
            Role::Knight => 3,
            Role::Bishop => 4,
            Role::Queen => 5,
            Role::King => 6,
        };
        out[usize::from(sq) ^ 56] = code + if piece.color == Color::White { 6 } else { 0 };
    }
    out
}

/// eval_pos(board, side): centipawns for `side` (1 = white, 0 = black).
pub fn evaluate(board: &[u8; 64], side: usize) -> i32 {
    let other = 1 - side;
    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut material = [0i32; 2];
    let mut phase = TOTAL_PHASE;
    // ranks of pawns per colour per file, in the order they were met
    let mut pawn_ranks: [[Vec<i32>; 8]; 2] = Default::default();
    let mut pawns: [Vec<usize>; 2] = Default::default();
    let mut rooks: [Vec<usize>; 2] = Default::default();
    let mut bishops = [0i32; 2];
    let mut at_home = [0i32; 2];
    let mut queens = [0i32; 2];
    let mut queen_out = [0i32; 2];
    let mut kings: [Option<usize>; 2] = [None, None];

    for i in 0..64 {
        let piece = board[i] as usize;
        if piece == 0 {
            continue;
        }
        let color = if piece < 7 { 0 } else { 1 };
        let kind = PIECE_TO_TABLE[if piece < 7 { piece - 1 } else { piece - 7 }];
        // white reads the tables as written, black mirrored
        let sq = if color == 1 { i } else { i ^ 56 };
        mg[color] += MIDGAME[kind][sq];
        eg[color] += ENDGAME[kind][sq];
        material[color] += MATERIAL[kind];
        phase -= PHASE_VALS[kind];
        match kind {
            PAWN => {
                pawns[color].push(i);
                pawn_ranks[color][i & 7].push(sq_rank(i));
            }
            KING => kings[color] = Some(i),
            ROOK => rooks[color].push(i),
            QUEEN => {
                queens[color] += 1;
                if i != HOME_QUEEN[color] {
                    queen_out[color] += 1;
                }
            }
            _ => {
                if kind == BISHOP {
                    bishops[color] += 1;
                }
                let home = if kind == KNIGHT { HOME_KNIGHT[color] } else { HOME_BISHOP[color] };
                if i == home.0 || i == home.1 {
                    at_home[color] += 1;
                }
            }
        }
    }

    // Bare kings, or a lone minor that cannot mate, is a dead draw.
    if pawns[0].is_empty() && pawns[1].is_empty()
        && rooks[0].is_empty() && rooks[1].is_empty()
        && queens[0] == 0 && queens[1] == 0
        && material[0] <= 330 && material[1] <= 330
    {
        return 0;
    }

    for color in 0..2 {
        let enemy = 1 - color;
        let ahead = |er: i32, r: i32| if color == 1 { er > r } else { er < r };

        for &i in &pawns[color] {
            let f = (i & 7) as usize;
            let r = sq_rank(i);
            let rel = rel_rank(color, i);
            if pawn_ranks[color][f].len() > 1 {
                mg[color] += DOUBLED.0;
                eg[color] += DOUBLED.1;
            }
            let neighbour = (f > 0 && !pawn_ranks[color][f - 1].is_empty())
                || (f < 7 && !pawn_ranks[color][f + 1].is_empty());
            if !neighbour {
                mg[color] += ISOLATED.0;
                eg[color] += ISOLATED.1;
            }
            let mut passed = true;
            'files: for ff in [f as i32 - 1, f as i32, f as i32 + 1] {
                if !(0..8).contains(&ff) {
                    continue;
                }
                for &er in &pawn_ranks[enemy][ff as usize] {
                    if ahead(er, r) {
                        passed = false;
                        break 'files;
                    }
                }
            }
            if passed {
                for &orr in &pawn_ranks[color][f] {
                    if ahead(orr, r) {
                        passed = false;
                        break;
                    }
                }
            }
            if passed {
                mg[color] += PASSED_MG[rel];
                eg[color] += PASSED_EG[rel];
                if rel >= 3 {
                    if let (Some(k), Some(ek)) = (kings[color], kings[enemy]) {
                        let promo = if color == 1 { f } else { 56 + f };
                        eg[color] += 7 * manhattan(ek, promo);
                        eg[color] -= 5 * manhattan(k, promo);
                    }
                }
            }
        }

        if bishops[color] >= 2 {
            mg[color] += BISHOP_PAIR.0;
            eg[color] += BISHOP_PAIR.1;
        }

        for &i in &rooks[color] {
            let f = i & 7;
            if pawn_ranks[color][f].is_empty() {
                if pawn_ranks[enemy][f].is_empty() {
                    mg[color] += ROOK_OPEN.0;
                    eg[color] += ROOK_OPEN.1;
                } else {
                    mg[color] += ROOK_SEMI.0;
                    eg[color] += ROOK_SEMI.1;
                }
            }
            if rel_rank(color, i) == 6 {
                mg[color] += ROOK_SEVENTH.0;
                eg[color] += ROOK_SEVENTH.1;
            }
        }

        mg[color] += UNDEVELOPED_MG * at_home[color];
        if queen_out[color] > 0 && at_home[color] >= 2 {
            mg[color] += EARLY_QUEEN_MG;
        }
        let step = FORWARD_STEP[color];
        let own_pawn = if color == 1 { 7 } else { 1 };
        for &p in &[HOME_CENTER_PAWNS[color].0, HOME_CENTER_PAWNS[color].1] {
            let ahead_sq = p as i32 + step;
            if board[p] == own_pawn && board[ahead_sq as usize] != 0 {
                mg[color] += BLOCKED_CENTER_PAWN_MG;
            }
        }

        if let Some(k) = kings[color] {
            if rel_rank(color, k) <= 1 {
                let kf = sq_file(k);
                let kr = sq_rank(k);
                let mut shield = 0;
                for ff in [kf - 1, kf, kf + 1] {
                    if !(0..8).contains(&ff) {
                        continue;
                    }
                    let ranks = &pawn_ranks[color][ff as usize];
                    let covered = ranks.iter().any(|&pr| {
                        let d = if color == 1 { pr - kr } else { kr - pr };
                        (1..=2).contains(&d)
                    });
                    if covered {
                        shield += 1;
                    } else if ranks.is_empty() {
                        mg[color] += KING_OPEN_FILE_MG;
                    }
                }
                mg[color] += SHIELD_MISSING_MG * (3 - shield);
            }
        }
    }

    // Mop-up: with a decisive edge and no enemy pawns, drive their king to
    // the edge and bring ours in.
    let mut mop_up = 0;
    for color in 0..2 {
        let enemy = 1 - color;
        if material[color] - material[enemy] >= 450 && pawns[enemy].is_empty() {
            if let (Some(k), Some(ek)) = (kings[color], kings[enemy]) {
                let bonus = (47 * CENTER_DISTANCE[ek] + 16 * (14 - manhattan(k, ek))) / 10;
                mop_up += if color == 1 { bonus } else { -bonus };
            }
        }
    }

    let mg_total = mg[side] - mg[other];
    let mut eg_total = eg[side] - eg[other];
    if side == 1 {
        eg_total += mop_up;
    } else {
        eg_total -= mop_up;
    }
    // Floor division like Python's //, which matters once promotions push
    // the phase below zero.
    let phase = (phase * 256 + TOTAL_PHASE / 2).div_euclid(TOTAL_PHASE);
    let total = mg_total * (256 - phase) + eg_total * phase;
    // Rust's / truncates towards zero, which is what the Python does by hand.
    total / 256
}

fn hash_of(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

fn victim_value(m: &Move) -> i32 {
    match m {
        Move::EnPassant { .. } => 100,
        _ => match m.capture() {
            Some(role) => MATERIAL[PIECE_TO_TABLE[match role {
                Role::Pawn => 0,
                Role::Rook => 1,
                Role::Knight => 2,
                Role::Bishop => 3,
                Role::Queen => 4,
                Role::King => 5,
            }]],
            None => 0,
        },
    }
}

fn negamax(pos: &Chess, depth: u32, mut alpha: i32, beta: i32, ply: i32, hashes: &mut Vec<u64>) -> i32 {
    let mut legal: Vec<Move> = pos.legal_moves().into_iter().collect();
    if legal.is_empty() {
        return if pos.is_check() { -(MATE - ply) } else { 0 };
    }
    let halfmoves = pos.halfmoves() as usize;
    if halfmoves >= 100 || pos.is_insufficient_material() {
        return 0;
    }
    if halfmoves >= 4 {
        let h = hash_of(pos);
        if hashes.iter().rev().skip(1).take(halfmoves).any(|&x| x == h) {
            return 0;
        }
    }
    if depth == 0 {
        let side = if pos.turn() == Color::White { 1 } else { 0 };
        return evaluate(&fen_order(pos), side);
    }
    legal.sort_by_key(|m| -victim_value(m));
    let mut best = -MATE * 2;
    for m in legal {
        let mut child = pos.clone();
        child.play_unchecked(&m);
        hashes.push(hash_of(&child));
        let score = -negamax(&child, depth - 1, -beta, -alpha, ply + 1, hashes);
        hashes.pop();
        if score > best {
            best = score;
        }
        if score >= beta {
            return score;
        }
        if score > alpha {
            alpha = score;
        }
    }
    best
}

/// The engine's move: a full-width search `depth` plies deep, ties at the
/// root broken at random. `history` is the game's recent position hashes,
/// current position last, for repetition detection.
pub fn choose(pos: &Chess, history: &[u64], depth: u32, seed: u64) -> Option<Move> {
    let legal: Vec<Move> = pos.legal_moves().into_iter().collect();
    if legal.is_empty() {
        return None;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut hashes: Vec<u64> = history.to_vec();
    let mut best: Vec<Move> = Vec::new();
    let mut best_score = -MATE * 2;
    for m in legal {
        let mut child = pos.clone();
        child.play_unchecked(&m);
        hashes.push(hash_of(&child));
        let score = -negamax(&child, depth.saturating_sub(1), -MATE * 2, MATE * 2, 1, &mut hashes);
        hashes.pop();
        if score > best_score {
            best_score = score;
            best.clear();
        }
        if score == best_score {
            best.push(m);
        }
    }
    Some(best[rng.gen_range(0..best.len())].clone())
}
