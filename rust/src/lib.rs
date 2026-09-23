//! The per-simulation half of rl_mcts.py, in Rust: move generation, the
//! position and move encodings, PUCT selection with virtual loss, backup,
//! terminal detection and root noise. One `Searcher` holds any number of
//! trees and walks them all in parallel with rayon; Python only sees whole
//! batches -- `collect()` hands back every new leaf as arrays for the net,
//! `apply()` takes the net's answers -- and the moves and visit counts of a
//! root when it wants to play a move.
//!
//! Conventions are exactly rl_model.py's and rl_mcts.py's: everything is from
//! the side to move's view (flipped with `^ 56` when Black is to move), a
//! move is from * 64 + to with the 72 underpromotions after 4096, values are
//! in [-1, 1] for the side to move at the node that holds them, and one
//! repetition inside the tree is already a draw but the root never is.
//! test_rl.py checks the encodings against the Python ones square by square.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Dirichlet, Distribution};
use rayon::prelude::*;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, File, Move, Position, Role, Square};

const EP_TOKEN: i64 = 13;
const OUR_CASTLING_ROOK: i64 = 14;
const THEIR_CASTLING_ROOK: i64 = 15;
const NONE: u32 = u32::MAX;

// --- encodings ---------------------------------------------------------------

fn encode(pos: &Chess) -> ([i64; 64], f32) {
    let us = pos.turn();
    let flip = if us == Color::White { 0 } else { 56 };
    let mut tokens = [0i64; 64];
    let board = pos.board();
    for (sq, piece) in board.clone() {
        let base = if piece.color == us { 0 } else { 6 };
        tokens[usize::from(sq) ^ flip] = base + piece.role as i64;
    }
    let ours = board.by_color(us);
    for sq in pos.castles().castling_rights() {
        tokens[usize::from(sq) ^ flip] =
            if ours.contains(sq) { OUR_CASTLING_ROOK } else { THEIR_CASTLING_ROOK };
    }
    if let Some(sq) = pos.ep_square(EnPassantMode::Legal) {
        tokens[usize::from(sq) ^ flip] = EP_TOKEN;
    }
    (tokens, pos.halfmoves().min(100) as f32 / 100.0)
}

/// (from, to, promotion) the way python-chess writes it: castling as the
/// king's two-square step, promotion as python-chess's piece number (0 = none).
fn plain(m: &Move) -> (u8, u8, u8) {
    let (from, to) = match *m {
        Move::Castle { king, rook } => {
            let file = if rook.file() > king.file() { File::G } else { File::C };
            (king, Square::from_coords(file, king.rank()))
        }
        _ => (m.from().expect("drops do not occur in chess"), m.to()),
    };
    (u8::from(from), u8::from(to), m.promotion().map_or(0, |r| r as u8))
}

fn move_index(m: &Move, turn: Color) -> i64 {
    let flip = if turn == Color::White { 0 } else { 56 };
    let (from, to, _) = plain(m);
    let src = usize::from(from) ^ flip;
    let dst = usize::from(to) ^ flip;
    let under = match m.promotion() {
        Some(Role::Knight) => 0,
        Some(Role::Bishop) => 1,
        Some(Role::Rook) => 2,
        _ => return (src * 64 + dst) as i64,
    };
    let file = src & 7;
    (4096 + (file * 3 + (dst & 7) + 1 - file) * 3 + under) as i64
}

fn hash_of(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

// --- the tree ----------------------------------------------------------------

#[derive(Default)]
struct Node {
    moves: Vec<Move>,
    prior: Vec<f32>,
    n: Vec<f32>,
    w: Vec<f32>,
    child: Vec<u32>,
    visits: f32,
    value_sum: f32,
    terminal: Option<f32>,
    expanded: bool,
    pending: bool,
}

/// A leaf that has been sent to the net and is waiting for its answer.
struct Pending {
    path: Vec<(u32, usize)>,
    node: u32,
    legal: Vec<Move>,
    indices: Vec<i64>,
    tokens: [i64; 64],
    halfmove: f32,
}

struct Tree {
    nodes: Vec<Node>,
    root: u32,
    pos: Chess,
    /// Plies played before `history[0]`, so ply() is the game's, not the
    /// tree's: a tree may be started from the last irreversible move only.
    ply_base: usize,
    /// Zobrist hashes of every position the game has been through, the
    /// current one last, so the search can see a repetition of one of them.
    history: Vec<u64>,
    target: u32,
    sims: u32,
    noise: Option<(f32, f32)>,
    root_prior: Option<Vec<f32>>,
    pending: Vec<Pending>,
    rng: StdRng,
}

enum Walk {
    Done,
    Collided,
    Leaf(Pending),
}

impl Tree {
    fn new(pos: Chess, history: Vec<u64>, seed: u64) -> Tree {
        let ply_base = 2 * (pos.fullmoves().get() as usize - 1) + (pos.turn() == Color::Black) as usize
            + 1 - history.len();
        Tree {
            nodes: vec![Node::default()],
            root: 0,
            pos,
            ply_base,
            history,
            target: 0,
            sims: 0,
            noise: None,
            root_prior: None,
            pending: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn node(&self, i: u32) -> &Node {
        &self.nodes[i as usize]
    }

    fn node_mut(&mut self, i: u32) -> &mut Node {
        &mut self.nodes[i as usize]
    }

    /// Exact value for the side to move if the game is over here, else None.
    fn terminal_value(&self, pos: &Chess, no_moves: bool, path_hashes: &[u64]) -> Option<f32> {
        if no_moves {
            return Some(if pos.is_check() { -1.0 } else { 0.0 });
        }
        let halfmoves = pos.halfmoves() as usize;
        if halfmoves >= 100 || pos.is_insufficient_material() {
            return Some(0.0);
        }
        // A repetition needs four reversible plies, and can only match a
        // position from within the last `halfmoves` of them.
        if halfmoves >= 4 {
            let hash = hash_of(pos);
            let in_path = path_hashes.iter().rev().take(halfmoves).any(|&h| h == hash);
            let back = halfmoves.saturating_sub(path_hashes.len());
            let start = self.history.len().saturating_sub(back);
            if in_path || self.history[start..].contains(&hash) {
                return Some(0.0);
            }
        }
        None
    }

    fn select(&self, i: u32, c_puct: f32, fpu_reduction: f32) -> usize {
        let node = self.node(i);
        let prior: &[f32] = match (&self.root_prior, i == self.root) {
            (Some(p), true) => p,
            _ => &node.prior,
        };
        let fpu = node.value_sum / node.visits - fpu_reduction;
        let scale = c_puct * node.visits.sqrt();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for k in 0..node.moves.len() {
            let n = node.n[k];
            let q = if n > 0.0 { node.w[k] / n } else { fpu };
            let score = q + scale * prior[k] / (1.0 + n);
            if score > best_score {
                best_score = score;
                best = k;
            }
        }
        best
    }

    fn descend(&mut self, c_puct: f32, fpu_reduction: f32) -> Walk {
        let mut i = self.root;
        let mut pos = self.pos.clone();
        let mut path: Vec<(u32, usize)> = Vec::new();
        let mut hashes: Vec<u64> = Vec::new();
        loop {
            if let Some(v) = self.node(i).terminal {
                self.backup(&path, i, v);
                return Walk::Done;
            }
            if !self.node(i).expanded {
                if self.node(i).pending {
                    self.undo(&path);
                    return Walk::Collided;
                }
                let legal: Vec<Move> = pos.legal_moves().into_iter().collect();
                // The root is the real game, which only a lack of moves ends.
                let value = if path.is_empty() && !legal.is_empty() {
                    None
                } else {
                    self.terminal_value(&pos, legal.is_empty(), &hashes)
                };
                if let Some(v) = value {
                    self.node_mut(i).terminal = Some(v);
                    self.backup(&path, i, v);
                    return Walk::Done;
                }
                self.node_mut(i).pending = true;
                let (tokens, halfmove) = encode(&pos);
                let turn = pos.turn();
                let indices = legal.iter().map(|m| move_index(m, turn)).collect();
                return Walk::Leaf(Pending { path, node: i, legal, indices, tokens, halfmove });
            }
            let k = self.select(i, c_puct, fpu_reduction);
            let node = self.node_mut(i);
            // Virtual loss: count the visit now and score it as a loss until
            // the real value comes back, so other descents look elsewhere.
            node.n[k] += 1.0;
            node.w[k] -= 1.0;
            node.visits += 1.0;
            let m = node.moves[k].clone();
            path.push((i, k));
            pos.play_unchecked(&m);
            hashes.push(hash_of(&pos));
            let mut child = node.child[k];
            if child == NONE {
                child = self.nodes.len() as u32;
                self.nodes.push(Node::default());
                self.node_mut(i).child[k] = child;
            }
            i = child;
        }
    }

    fn expand(&mut self, leaf: Pending, logits: &[f32], value: f32) {
        let n = leaf.legal.len();
        let top = logits[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut prior: Vec<f32> = logits[..n].iter().map(|&l| (l - top).exp()).collect();
        let total: f32 = prior.iter().sum();
        for p in prior.iter_mut() {
            *p /= total;
        }
        let node = self.node_mut(leaf.node);
        node.moves = leaf.legal;
        node.prior = prior;
        node.n = vec![0.0; n];
        node.w = vec![0.0; n];
        node.child = vec![NONE; n];
        node.expanded = true;
        node.pending = false;
        self.backup(&leaf.path, leaf.node, value);
    }

    fn backup(&mut self, path: &[(u32, usize)], leaf: u32, mut value: f32) {
        let node = self.node_mut(leaf);
        node.visits += 1.0;
        node.value_sum += value;
        for &(i, k) in path.iter().rev() {
            value = -value;
            let node = self.node_mut(i);
            node.w[k] += 1.0 + value; // the 1 takes the virtual loss back out
            node.value_sum += value;
        }
    }

    fn undo(&mut self, path: &[(u32, usize)]) {
        for &(i, k) in path {
            let node = self.node_mut(i);
            node.n[k] -= 1.0;
            node.w[k] += 1.0;
            node.visits -= 1.0;
        }
    }

    fn add_noise(&mut self) {
        let (alpha, fraction) = match self.noise {
            Some(x) => x,
            None => return,
        };
        let prior = self.node(self.root).prior.clone();
        if prior.len() < 2 {
            return;
        }
        let noise = Dirichlet::new(&vec![alpha; prior.len()])
            .expect("alpha is positive")
            .sample(&mut self.rng);
        self.root_prior = Some(
            prior.iter().zip(noise).map(|(p, e)| (1.0 - fraction) * p + fraction * e as f32).collect(),
        );
    }

    /// Play a move, keeping whatever was searched below it. The old tree is
    /// dropped whole -- a game's worth of dead subtrees would otherwise pile up.
    fn advance(&mut self, m: &Move) {
        let root = self.node(self.root);
        let kept = root
            .moves
            .iter()
            .position(|x| x == m)
            .map(|k| root.child[k])
            .filter(|&c| c != NONE)
            .filter(|&c| {
                let child = self.node(c);
                // A child the tree scored as a repetition draw is a live
                // position once it is the real one: search it afresh.
                !child.pending && child.terminal.is_none()
            });
        let mut nodes = Vec::new();
        match kept {
            Some(c) => {
                self.copy_subtree(c, &mut nodes);
            }
            None => nodes.push(Node::default()),
        };
        self.nodes = nodes;
        self.root = 0;
        self.pos.play_unchecked(m);
        self.history.push(hash_of(&self.pos));
        self.root_prior = None;
        self.sims = 0;
    }

    fn copy_subtree(&self, from: u32, into: &mut Vec<Node>) -> u32 {
        let at = into.len() as u32;
        let src = self.node(from);
        into.push(Node {
            moves: src.moves.clone(),
            prior: src.prior.clone(),
            n: src.n.clone(),
            w: src.w.clone(),
            child: vec![NONE; src.child.len()],
            visits: src.visits,
            value_sum: src.value_sum,
            terminal: src.terminal,
            expanded: src.expanded,
            pending: false,
        });
        for k in 0..src.child.len() {
            if src.child[k] != NONE {
                let c = self.copy_subtree(src.child[k], into);
                into[at as usize].child[k] = c;
            }
        }
        at
    }

    fn best_child(&self, random_ties: bool, rng: &mut StdRng) -> usize {
        let root = self.node(self.root);
        let top_n = root.n.iter().cloned().fold(0.0, f32::max);
        let mut ties: Vec<usize> = if top_n <= 0.0 {
            let top_p = root.prior.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            (0..root.moves.len()).filter(|&k| root.prior[k] == top_p).collect()
        } else {
            (0..root.moves.len()).filter(|&k| root.n[k] == top_n).collect()
        };
        if ties.len() > 1 && top_n > 0.0 {
            let q = |k: usize| root.w[k] / root.n[k];
            let top_q = ties.iter().map(|&k| q(k)).fold(f32::NEG_INFINITY, f32::max);
            ties.retain(|&k| q(k) == top_q);
        }
        if ties.len() == 1 || !random_ties {
            ties[0]
        } else {
            ties[rng.gen_range(0..ties.len())]
        }
    }

    fn pv(&self, limit: usize) -> (Vec<Move>, u32) {
        let mut line = Vec::new();
        let mut i = self.root;
        loop {
            let node = self.node(i);
            if !node.expanded || line.len() >= limit || node.n.iter().all(|&n| n <= 0.0) {
                return (line, i);
            }
            let mut rng = StdRng::seed_from_u64(0); // ties resolved deterministically
            let k = self.best_child_at(i, &mut rng);
            line.push(node.moves[k].clone());
            if node.child[k] == NONE {
                return (line, NONE);
            }
            i = node.child[k];
        }
    }

    fn best_child_at(&self, i: u32, _rng: &mut StdRng) -> usize {
        let node = self.node(i);
        let mut best = 0;
        for k in 1..node.moves.len() {
            let better = node.n[k] > node.n[best]
                || (node.n[k] == node.n[best] && node.n[k] > 0.0
                    && node.w[k] / node.n[k] > node.w[best] / node.n[best]);
            if better {
                best = k;
            }
        }
        best
    }
}

// --- the Python side ---------------------------------------------------------

#[pyclass]
struct Searcher {
    trees: Vec<Option<Tree>>,
    c_puct: f32,
    fpu_reduction: f32,
    rng: StdRng,
    /// Own thread pool, or None for rayon's global one (see set_threads),
    /// which several searchers in one process share instead of each
    /// spinning up a full set of threads and fighting over the cores.
    pool: Option<rayon::ThreadPool>,
    /// Trees with leaves in the batch handed out by the last collect(), in
    /// batch order, so apply() can hand the rows back.
    batched: Vec<usize>,
    sims: u64,
    evals: u64,
}

fn moves_out(moves: &[Move]) -> Vec<(u8, u8, u8)> {
    moves.iter().map(plain).collect()
}

#[pymethods]
impl Searcher {
    #[new]
    #[pyo3(signature = (c_puct=1.75, fpu_reduction=0.25, seed=None, threads=0))]
    fn new(c_puct: f32, fpu_reduction: f32, seed: Option<u64>, threads: usize) -> PyResult<Self> {
        let pool = if threads == 0 {
            None
        } else {
            Some(rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyValueError::new_err(e.to_string()))?)
        };
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Ok(Searcher { trees: Vec::new(), c_puct, fpu_reduction, rng, pool, batched: Vec::new(), sims: 0, evals: 0 })
    }

    /// A new tree at the position reached by playing `moves` (UCI) from `fen`.
    /// Returns its id.
    fn new_tree(&mut self, fen: &str, moves: Vec<String>) -> PyResult<usize> {
        let setup: Fen = fen.parse().map_err(|e: shakmaty::fen::ParseFenError| PyValueError::new_err(e.to_string()))?;
        let mut pos: Chess = setup
            .into_position(CastlingMode::Standard)
            .map_err(|e: shakmaty::PositionError<Chess>| PyValueError::new_err(e.to_string()))?;
        let mut history = vec![hash_of(&pos)];
        for uci in moves {
            let m = parse_move(&pos, &uci)?;
            pos.play_unchecked(&m);
            history.push(hash_of(&pos));
        }
        let tree = Tree::new(pos, history, self.rng.gen());
        if let Some(slot) = self.trees.iter().position(|t| t.is_none()) {
            self.trees[slot] = Some(tree);
            return Ok(slot);
        }
        self.trees.push(Some(tree));
        Ok(self.trees.len() - 1)
    }

    fn set_c_puct(&mut self, value: f32) {
        self.c_puct = value;
    }

    fn drop_tree(&mut self, id: usize) -> PyResult<()> {
        self.tree(id)?;
        self.trees[id] = None;
        Ok(())
    }

    #[pyo3(signature = (id, target, noise_alpha=0.0, noise_fraction=0.0))]
    fn reset_search(&mut self, id: usize, target: u32, noise_alpha: f32, noise_fraction: f32) -> PyResult<()> {
        let tree = self.tree_mut(id)?;
        tree.target = target;
        tree.sims = 0;
        tree.noise = if noise_alpha > 0.0 && noise_fraction > 0.0 { Some((noise_alpha, noise_fraction)) } else { None };
        tree.root_prior = None;
        Ok(())
    }

    fn advance(&mut self, id: usize, uci: &str) -> PyResult<()> {
        let tree = self.tree_mut(id)?;
        let m = parse_move(&tree.pos, uci)?;
        tree.advance(&m);
        Ok(())
    }

    /// Descend every unfinished tree up to `leaves_per_tree` times and return
    /// the new leaves for the net: tokens (B, 64), halfmove (B,), and each
    /// row's legal-move indices padded with 0 (B, M). B is 0 once nothing is
    /// left to search.
    fn collect<'py>(&mut self, py: Python<'py>, leaves_per_tree: u32)
        -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<i64>>)> {
        let c_puct = self.c_puct;
        let fpu = self.fpu_reduction;
        let trees = &mut self.trees;
        let pool = &self.pool;
        let sims: u64 = py.allow_threads(|| {
            install(pool, || {
                trees
                    .par_iter_mut()
                    .map(|slot| {
                        let tree = match slot {
                            Some(t) => t,
                            None => return 0,
                        };
                        let want = leaves_per_tree.min(tree.target.saturating_sub(tree.sims));
                        if want == 0 {
                            return 0;
                        }
                        if tree.noise.is_some() && tree.root_prior.is_none() && tree.node(tree.root).expanded {
                            tree.add_noise();
                        }
                        let mut done = 0;
                        for _ in 0..want {
                            match tree.descend(c_puct, fpu) {
                                Walk::Collided => break,
                                Walk::Done => {}
                                Walk::Leaf(leaf) => tree.pending.push(leaf),
                            }
                            tree.sims += 1;
                            done += 1;
                        }
                        done
                    })
                    .sum()
            })
        });
        self.sims += sims;

        self.batched = (0..self.trees.len())
            .filter(|&i| self.trees[i].as_ref().map_or(false, |t| !t.pending.is_empty()))
            .collect();
        let rows: usize = self.batched.iter().map(|&i| self.trees[i].as_ref().unwrap().pending.len()).sum();
        let width = self
            .batched
            .iter()
            .flat_map(|&i| self.trees[i].as_ref().unwrap().pending.iter().map(|p| p.indices.len()))
            .max()
            .unwrap_or(0);
        let mut tokens = Vec::with_capacity(rows * 64);
        let mut halfmove = Vec::with_capacity(rows);
        let mut indices = vec![0i64; rows * width];
        let mut r = 0;
        for &i in &self.batched {
            for leaf in &self.trees[i].as_ref().unwrap().pending {
                tokens.extend_from_slice(&leaf.tokens);
                halfmove.push(leaf.halfmove);
                indices[r * width..r * width + leaf.indices.len()].copy_from_slice(&leaf.indices);
                r += 1;
            }
        }
        self.evals += rows as u64;
        Ok((
            PyArray1::from_vec(py, tokens).reshape([rows, 64])?,
            PyArray1::from_vec(py, halfmove),
            PyArray1::from_vec(py, indices).reshape([rows, width])?,
        ))
    }

    /// The net's answers for the last collect(): the legal moves' logits
    /// (B, M) and a value per row. Expands and backs up every leaf.
    fn apply(&mut self, py: Python<'_>, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let logits = logits.as_array();
        let values = values.as_array();
        let rows: usize = self.batched.iter().map(|&i| self.trees[i].as_ref().unwrap().pending.len()).sum();
        if logits.nrows() != rows || values.len() != rows {
            return Err(PyValueError::new_err(format!("expected {} rows, got {}", rows, logits.nrows())));
        }
        let width = logits.ncols();
        let logits: Vec<f32> = logits.iter().cloned().collect();
        let values: Vec<f32> = values.iter().cloned().collect();
        // Each tree's rows are consecutive, so hand every tree its slice and
        // let them expand in parallel.
        let mut offsets = Vec::with_capacity(self.batched.len());
        let mut at = 0;
        for &i in &self.batched {
            offsets.push(at);
            at += self.trees[i].as_ref().unwrap().pending.len();
        }
        let batched = std::mem::take(&mut self.batched);
        let mut work: Vec<(&mut Tree, usize)> = Vec::with_capacity(batched.len());
        {
            // Distinct ids, so the mutable borrows do not overlap.
            let mut slots: Vec<Option<&mut Tree>> = self.trees.iter_mut().map(|t| t.as_mut()).collect();
            for (&i, &off) in batched.iter().zip(&offsets) {
                work.push((slots[i].take().unwrap(), off));
            }
        }
        let pool = &self.pool;
        py.allow_threads(|| {
            install(pool, || {
                work.into_par_iter().for_each(|(tree, off)| {
                    let pending = std::mem::take(&mut tree.pending);
                    for (j, leaf) in pending.into_iter().enumerate() {
                        let r = off + j;
                        tree.expand(leaf, &logits[r * width..(r + 1) * width], values[r]);
                    }
                });
            })
        });
        Ok(())
    }

    /// Ids of trees whose search has reached its target.
    fn done(&self) -> Vec<usize> {
        (0..self.trees.len())
            .filter(|&i| self.trees[i].as_ref().map_or(false, |t| t.target > 0 && t.sims >= t.target))
            .collect()
    }

    fn busy(&self) -> bool {
        self.trees.iter().flatten().any(|t| t.sims < t.target)
    }

    fn sims(&self, id: usize) -> PyResult<u32> {
        Ok(self.tree(id)?.sims)
    }

    fn total_sims(&self) -> u64 {
        self.sims
    }

    fn total_evals(&self) -> u64 {
        self.evals
    }

    /// The root's moves as (from, to, promotion) triples, with their visit
    /// counts, values for the side choosing them, and priors.
    fn root<'py>(&self, py: Python<'py>, id: usize)
        -> PyResult<(Vec<(u8, u8, u8)>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let tree = self.tree(id)?;
        let root = tree.node(tree.root);
        if !root.expanded {
            return Err(PyValueError::new_err("root not expanded yet"));
        }
        let q: Vec<f32> = root.n.iter().zip(&root.w).map(|(&n, &w)| if n > 0.0 { w / n } else { 0.0 }).collect();
        Ok((
            moves_out(&root.moves),
            root.n.clone().into_pyarray(py),
            q.into_pyarray(py),
            root.prior.clone().into_pyarray(py),
        ))
    }

    fn expanded(&self, id: usize) -> PyResult<bool> {
        let tree = self.tree(id)?;
        Ok(tree.node(tree.root).expanded)
    }

    #[pyo3(signature = (id, random_ties=false))]
    fn best_child(&mut self, id: usize, random_ties: bool) -> PyResult<usize> {
        let rng = &mut self.rng;
        let tree = self.trees.get(id).and_then(|t| t.as_ref()).ok_or_else(|| PyValueError::new_err("no such tree"))?;
        Ok(tree.best_child(random_ties, rng))
    }

    /// A move drawn in proportion to visits ** (1 / temperature).
    fn sample_child(&mut self, id: usize, temperature: f32) -> PyResult<usize> {
        if temperature <= 0.0 {
            return self.best_child(id, true);
        }
        let tree = self.trees.get(id).and_then(|t| t.as_ref()).ok_or_else(|| PyValueError::new_err("no such tree"))?;
        let root = tree.node(tree.root);
        let weights: Vec<f64> = root.n.iter().map(|&n| (n as f64).powf(1.0 / temperature as f64)).collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Ok(tree.best_child(true, &mut self.rng));
        }
        let mut pick = self.rng.gen::<f64>() * total;
        for (k, w) in weights.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                return Ok(k);
            }
        }
        Ok(weights.len() - 1)
    }

    /// The line of most-visited moves.
    #[pyo3(signature = (id, limit=24))]
    fn pv(&self, id: usize, limit: usize) -> PyResult<Vec<(u8, u8, u8)>> {
        Ok(moves_out(&self.tree(id)?.pv(limit).0))
    }

    /// Moves to mate if the principal variation ends in one, signed for the
    /// side to move at the root (negative: it is the one being mated).
    fn mate_in(&self, id: usize) -> PyResult<Option<i32>> {
        let tree = self.tree(id)?;
        let (line, end) = tree.pv(64);
        if end == NONE || tree.node(end).terminal != Some(-1.0) {
            return Ok(None);
        }
        let plies = line.len() as i32;
        Ok(Some(if plies % 2 == 1 { (plies + 1) / 2 } else { -(plies / 2) }))
    }

    /// The root position the way the net sees it, plus the root moves'
    /// indices: a training row without a python-chess round trip.
    fn root_encoding<'py>(&self, py: Python<'py>, id: usize)
        -> PyResult<(Bound<'py, PyArray1<i64>>, f32, Bound<'py, PyArray1<i64>>)> {
        let tree = self.tree(id)?;
        let (tokens, halfmove) = encode(&tree.pos);
        let turn = tree.pos.turn();
        let root = tree.node(tree.root);
        let indices: Vec<i64> = root.moves.iter().map(|m| move_index(m, turn)).collect();
        Ok((tokens.to_vec().into_pyarray(py), halfmove, indices.into_pyarray(py)))
    }

    /// (White's score, reason) if the game at the root is over, else None.
    /// Every draw is claimed as soon as it can be, like rl_mcts.game_over().
    fn game_over(&self, id: usize) -> PyResult<Option<(i8, String)>> {
        let tree = self.tree(id)?;
        let pos = &tree.pos;
        if pos.legal_moves().is_empty() {
            if pos.is_check() {
                return Ok(Some((if pos.turn() == Color::White { -1 } else { 1 }, "mate".into())));
            }
            return Ok(Some((0, "stalemate".into())));
        }
        if pos.is_insufficient_material() {
            return Ok(Some((0, "material".into())));
        }
        let halfmoves = pos.halfmoves() as usize;
        if halfmoves >= 100 {
            return Ok(Some((0, "fifty".into())));
        }
        if halfmoves >= 8 {
            let hash = hash_of(pos);
            let start = tree.history.len().saturating_sub(halfmoves + 1);
            if tree.history[start..].iter().filter(|&&h| h == hash).count() >= 3 {
                return Ok(Some((0, "repetition".into())));
            }
        }
        Ok(None)
    }

    fn ply(&self, id: usize) -> PyResult<usize> {
        let tree = self.tree(id)?;
        Ok(tree.ply_base + tree.history.len() - 1)
    }

    fn white_to_move(&self, id: usize) -> PyResult<bool> {
        Ok(self.tree(id)?.pos.turn() == Color::White)
    }

    fn fen(&self, id: usize) -> PyResult<String> {
        let tree = self.tree(id)?;
        Ok(Fen::from_position(tree.pos.clone(), EnPassantMode::Legal).to_string())
    }

    fn node_count(&self, id: usize) -> PyResult<usize> {
        Ok(self.tree(id)?.nodes.len())
    }
}

impl Searcher {
    fn tree(&self, id: usize) -> PyResult<&Tree> {
        self.trees.get(id).and_then(|t| t.as_ref()).ok_or_else(|| PyValueError::new_err("no such tree"))
    }

    fn tree_mut(&mut self, id: usize) -> PyResult<&mut Tree> {
        self.trees.get_mut(id).and_then(|t| t.as_mut()).ok_or_else(|| PyValueError::new_err("no such tree"))
    }
}

fn install<R: Send>(pool: &Option<rayon::ThreadPool>, work: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(p) => p.install(work),
        None => work(),
    }
}

/// Size rayon's global pool, which every Searcher made with threads=0 uses.
/// Only the first call in a process counts; later ones are ignored.
#[pyfunction]
fn set_threads(threads: usize) -> bool {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().is_ok()
}

fn parse_move(pos: &Chess, uci: &str) -> PyResult<Move> {
    let uci = UciMove::from_ascii(uci.as_bytes()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    uci.to_move(pos).map_err(|e| PyValueError::new_err(format!("{} is not legal here: {}", uci, e)))
}

/// rl_model.encode() for a FEN, for checking the two agree.
#[pyfunction]
fn encode_fen(fen: &str) -> PyResult<(Vec<i64>, f32)> {
    let pos = position(fen)?;
    let (tokens, halfmove) = encode(&pos);
    Ok((tokens.to_vec(), halfmove))
}

/// Every legal move of a FEN as ((from, to, promotion), move_index), for
/// checking against rl_model.move_index().
#[pyfunction]
fn legal_moves_fen(fen: &str) -> PyResult<Vec<((u8, u8, u8), i64)>> {
    let pos = position(fen)?;
    let turn = pos.turn();
    Ok(pos.legal_moves().iter().map(|m| (plain(m), move_index(m, turn))).collect())
}

fn position(fen: &str) -> PyResult<Chess> {
    let setup: Fen = fen.parse().map_err(|e: shakmaty::fen::ParseFenError| PyValueError::new_err(e.to_string()))?;
    setup
        .into_position(CastlingMode::Standard)
        .map_err(|e: shakmaty::PositionError<Chess>| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn nighty_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Searcher>()?;
    m.add_function(wrap_pyfunction!(encode_fen, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves_fen, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add("N_MOVES", 4096 + 72)?;
    Ok(())
}
