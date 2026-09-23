//! run.py's search: the nets' shortlist at every node, negamax over it, a
//! mate-in-one sweep, repetition and fifty-move handling, iterative deepening
//! with one thread per root move. Scores are from the point of view of the
//! side to move, as in the Python. The one addition is alpha-beta pruning
//! inside each root subtree, which returns the same values and picks the same
//! move (root children are searched with a full window, and ties go to the
//! first candidate either way) for a fraction of the nodes.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, File, Move, Position, Role, Square};

use crate::classical::{self, fen_order};
use crate::net::{FromNet, ToNet};

pub use crate::classical::MATE;
const INFINITY: i32 = 10_000_000;

// run.py stops at 5. This build reaches that in well under a second, so the
// ceiling is higher and the clock decides; with a few seconds a move it
// usually completes 6.
pub const DEFAULT_DEPTH: u32 = 7;
pub const DEFAULT_MOVES: usize = 3;
pub const DEFAULT_PIECES: usize = 4;
pub const DEFAULT_OVERHEAD: f64 = 0.1;
pub const DEFAULT_BUDGET: f64 = 30.0;

#[derive(Clone, Copy)]
pub struct Params {
    pub depth: u32,
    pub moves: usize,
    pub pieces: usize,
    pub move_overhead: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            depth: DEFAULT_DEPTH,
            moves: DEFAULT_MOVES,
            pieces: DEFAULT_PIECES,
            move_overhead: DEFAULT_OVERHEAD,
        }
    }
}

pub struct Engine {
    pub from: FromNet,
    pub to: ToNet,
}

pub fn hash_of(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// (from, to) the way python-chess and UCI write it: castling as the king's
/// two-square step, which is the square the to-model was trained on.
fn plain(m: &Move) -> (usize, usize) {
    let (from, to) = match *m {
        Move::Castle { king, rook } => {
            let file = if rook.file() > king.file() { File::G } else { File::C };
            (king, Square::from_coords(file, king.rank()))
        }
        _ => (m.from().expect("drops do not occur in chess"), m.to()),
    };
    (usize::from(from), usize::from(to))
}

fn leaf_eval(pos: &Chess) -> i32 {
    // evaluate() takes encode()'s colour indices, 1 white and 0 black, and
    // reports from that side's point of view.
    let side = if pos.turn() == Color::White { 1 } else { 0 };
    classical::evaluate(&fen_order(pos), side)
}

/// How wide to look. Endgames have few legal moves, so the same budget buys a
/// shortlist that covers nearly all of them -- which is what stops the engine
/// from walking past a promotion it never generated.
fn beam_size(pos: &Chess, p: &Params) -> (usize, usize) {
    let pieces = pos.board().occupied().count();
    if pieces <= 8 {
        return (p.pieces + 2, p.moves + 2);
    }
    if pieces <= 14 {
        return (p.pieces + 1, p.moves + 1);
    }
    if pos.fullmoves().get() <= 10 {
        // Nothing narrows the opening down for us, so give the nets a little
        // more room before the shortlist hardens into a single plan.
        return (p.pieces + 1, p.moves);
    }
    (p.pieces, p.moves)
}

fn ranked_desc(scores: &[f32], squares: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut v: Vec<usize> = squares.collect();
    // Stable, like Python's sorted(): ties keep ascending square order.
    v.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// The nets' shortlist for this position, as legal moves, best first.
pub fn candidates(engine: &Engine, pos: &Chess, legals: &[Move], p: &Params) -> Vec<Move> {
    let (n_pieces, n_moves) = beam_size(pos, p);
    let board = fen_order(pos);
    let side = if pos.turn() == Color::White { 0 } else { 1 };

    let mut movable = [false; 64];
    for m in legals {
        movable[plain(m).0] = true;
    }
    let from_logits = engine.from.forward(&board);
    let mut sources = ranked_desc(&from_logits[side * 64..(side + 1) * 64], (0..64).filter(|&s| movable[s]));
    sources.truncate(n_pieces);

    let to_logits = engine.to.forward(&board, &sources);
    let mut picked = Vec::new();
    for (n, &src) in sources.iter().enumerate() {
        let row = &to_logits[n * 64..(n + 1) * 64];
        let mut taken = 0;
        for dest in ranked_desc(row, 0..64) {
            // A promotion needs its piece filled in; python-chess lists the
            // queen first, and that is the one the Python picks.
            let Some(m) = legals
                .iter()
                .filter(|m| plain(m) == (src, dest))
                .min_by_key(|m| match m.promotion() {
                    None | Some(Role::Queen) => 0,
                    Some(Role::Rook) => 1,
                    Some(Role::Bishop) => 2,
                    _ => 3,
                })
            else {
                continue;
            };
            picked.push(m.clone());
            // Underpromotion is almost always wrong, but a knight that arrives
            // with check is the one case worth a node.
            if m.promotion() == Some(Role::Queen) {
                if let Some(knight) = legals
                    .iter()
                    .find(|k| plain(k) == (src, dest) && k.promotion() == Some(Role::Knight))
                {
                    let mut after = pos.clone();
                    after.play_unchecked(knight);
                    if after.is_check() {
                        picked.push(knight.clone());
                    }
                }
            }
            taken += 1;
            if taken >= n_moves {
                break;
            }
        }
    }
    picked
}

/// Mate in one, if there is one. The shortlist regularly misses the mating
/// move in the endgame, and an engine that walks past mate in one cannot
/// convert anything. This only ever adds a forced mate; every other move still
/// comes from the models.
pub fn forced_mate(pos: &Chess, legals: &[Move]) -> Option<Move> {
    for m in legals {
        let mut after = pos.clone();
        after.play_unchecked(m);
        if after.is_checkmate() {
            return Some(m.clone());
        }
    }
    None
}

pub struct Timeout;

struct Ctx<'a> {
    engine: &'a Engine,
    params: Params,
    deadline: Option<Instant>,
    stop: &'a AtomicBool,
    /// Positions already reached in this game.
    history: &'a HashSet<u64>,
    /// Positions on the line from the root to here, past the halfmove cutoff.
    line: Vec<u64>,
    nodes: u64,
}

impl Ctx<'_> {
    fn search(&mut self, pos: &Chess, ply: i32, limit: i32, mut alpha: i32, beta: i32) -> Result<i32, Timeout> {
        if self.stop.load(Ordering::Relaxed) || self.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(Timeout);
        }
        self.nodes += 1;

        let legals = pos.legal_moves();
        if legals.is_empty() {
            // Mate scores shrink with distance from the root, so the engine
            // takes the quickest mate and the slowest loss.
            return Ok(if pos.is_check() { -(MATE - ply) } else { 0 });
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
            return Ok(0);
        }
        if ply >= limit {
            return Ok(leaf_eval(pos));
        }
        if forced_mate(pos, &legals).is_some() {
            return Ok(MATE - (ply + 1));
        }
        let moves = candidates(self.engine, pos, &legals, &self.params);
        if moves.is_empty() {
            return Ok(leaf_eval(pos));
        }

        let mut best = -INFINITY;
        for m in &moves {
            let mut child = pos.clone();
            child.play_unchecked(m);
            let score = self.child_score(&child, ply + 1, limit, -beta, -alpha)?;
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        Ok(best)
    }

    /// Score of a position a move has just been played into, from the point of
    /// view of the side that made the move.
    fn child_score(&mut self, pos: &Chess, ply: i32, limit: i32, alpha: i32, beta: i32) -> Result<i32, Timeout> {
        // A repetition needs at least four plies with no capture and no pawn
        // move, so below that there is nothing to hash.
        let key = (pos.halfmoves() >= 4).then(|| hash_of(pos));
        if let Some(h) = key {
            // Already on the board once, so claiming it is a draw. Scoring it
            // as one keeps the engine from repeating away a win, and lets it
            // hunt for the repetition when it is worse.
            if self.history.contains(&h) || self.line.contains(&h) {
                return Ok(0);
            }
            self.line.push(h);
        }
        let result = self.search(pos, ply, limit, alpha, beta).map(|s| -s);
        if key.is_some() {
            self.line.pop();
        }
        result
    }
}

/// One iteration of the deepening loop: every root move in its own thread.
/// None if the clock ran out before every root move had an answer.
fn root_search(
    engine: &Engine,
    pos: &Chess,
    roots: &[Move],
    limit: u32,
    params: Params,
    history: &HashSet<u64>,
    deadline: Option<Instant>,
    stop: &AtomicBool,
) -> Option<(Move, i32, u64)> {
    let results: Vec<Result<(i32, u64), Timeout>> = std::thread::scope(|s| {
        let handles: Vec<_> = roots
            .iter()
            .map(|m| {
                s.spawn(move || {
                    let mut child = pos.clone();
                    child.play_unchecked(m);
                    let mut ctx = Ctx {
                        engine,
                        params,
                        deadline,
                        stop,
                        history,
                        line: Vec::new(),
                        nodes: 0,
                    };
                    ctx.child_score(&child, 1, limit as i32, -INFINITY, INFINITY)
                        .map(|score| (score, ctx.nodes))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("search thread panicked")).collect()
    });

    let mut best: Option<(Move, i32)> = None;
    let mut nodes = 0;
    for (m, r) in roots.iter().zip(results) {
        // One root move never finished, so the iteration is not comparable.
        let (score, n) = r.ok()?;
        nodes += n;
        if best.as_ref().is_none_or(|b| score > b.1) {
            best = Some((m.clone(), score));
        }
    }
    best.map(|(m, s)| (m, s, nodes))
}

pub struct Answer {
    pub best: Move,
    pub score: i32,
}

/// Iterative deepening. Each completed depth replaces the previous answer, so
/// running out of time costs accuracy rather than the move. `info` gets one
/// line per completed depth.
pub fn find_best_move(
    engine: &Engine,
    pos: &Chess,
    history: &HashSet<u64>,
    params: Params,
    limit: u32,
    budget: Option<f64>,
    stop: &AtomicBool,
    info: &mut dyn FnMut(String),
) -> Option<Answer> {
    let started = Instant::now();
    let deadline = budget.map(|b| started + Duration::from_secs_f64(b));

    let legals: Vec<Move> = pos.legal_moves().into_iter().collect();
    if legals.is_empty() {
        return None;
    }

    // The root builds its own move list rather than going through search(),
    // so it needs the same mate-in-one sweep.
    if let Some(m) = forced_mate(pos, &legals) {
        info(format!("info depth 1 score mate 1 pv {}", uci(&m)));
        return Some(Answer { best: m, score: MATE - 1 });
    }

    let mut roots = candidates(engine, pos, &legals, &params);
    if roots.is_empty() {
        roots = legals;
    }

    // Something legal to play even if the very first iteration is cut short.
    let mut best = Answer { best: roots[0].clone(), score: 0 };
    let mut total_nodes = 0;
    for depth in 1..=limit {
        let Some((m, score, nodes)) = root_search(engine, pos, &roots, depth, params, history, deadline, stop)
        else {
            break;
        };
        best = Answer { best: m, score };
        total_nodes += nodes;
        let elapsed = started.elapsed().as_secs_f64();
        info(format!(
            "info depth {depth} score {} time {} nodes {total_nodes} nps {} pv {}",
            score_string(score),
            (elapsed * 1000.0) as u64,
            (total_nodes as f64 / elapsed.max(1e-6)) as u64,
            uci(&best.best)
        ));
        if score.abs() > MATE - 1000 {
            break;
        }
        // No point starting a depth we have no chance of finishing.
        if budget.is_some_and(|b| elapsed > b * 0.45) {
            break;
        }
    }
    Some(best)
}

pub fn uci(m: &Move) -> String {
    m.to_uci(shakmaty::CastlingMode::Standard).to_string()
}

pub fn score_string(score: i32) -> String {
    if score > MATE - 1000 {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score < -(MATE - 1000) {
        format!("mate -{}", (MATE + score + 1) / 2)
    } else {
        format!("cp {score}")
    }
}

/// The `go` parameters that matter for the clock, in milliseconds.
#[derive(Default, Clone, Copy)]
pub struct GoParams {
    pub wtime: Option<i64>,
    pub btime: Option<i64>,
    pub winc: Option<i64>,
    pub binc: Option<i64>,
    pub movestogo: Option<i64>,
    pub movetime: Option<i64>,
    pub depth: Option<i64>,
    pub infinite: bool,
}

/// Seconds to spend on this move.
pub fn time_budget(turn: Color, g: &GoParams, move_overhead: f64) -> f64 {
    if let Some(mt) = g.movetime {
        return (mt as f64 / 1000.0 - move_overhead).max(0.02);
    }
    let (clock, bonus) = if turn == Color::White { (g.wtime, g.winc) } else { (g.btime, g.binc) };
    let Some(left) = clock else {
        // No clock to reason about. Still cap it: a wide endgame shortlist at
        // the full depth limit can otherwise run for minutes.
        return DEFAULT_BUDGET;
    };
    let left = left as f64 / 1000.0;
    let inc = bonus.unwrap_or(0) as f64 / 1000.0;
    let share = match g.movestogo {
        Some(togo) if togo > 0 => togo.clamp(1, 30) as f64,
        _ => 28.0,
    };
    let budget = left / share + inc * 0.75;
    (budget.min(left * 0.4) - move_overhead).max(0.02)
}
