//! NightyBot, the imitation-learned engine, as one binary: run.py's UCI loop
//! over search.rs and net.rs, with evaluation.py's piece-square evaluation
//! taken from the RL crate's port (../rust/src/classical.rs) so there is one
//! Rust copy of it.
//!
//!     nightybot                  UCI on stdin; finds the .safetensors files in
//!                                the working directory, then next to the binary
//!                                and its parents
//!     nightybot --models DIR     look for them in DIR instead
//!     nightybot bench [DEPTH]    fixed-depth searches of a few positions, nodes/s
//!     nightybot logits FEN       both nets' raw outputs, for test_engine_rs.py
//!     nightybot candidates FEN   the shortlist candidates() builds, same purpose
//!
//! stdin is read on a thread, so `stop` interrupts a running search (run.py
//! could only answer it afterwards). `quit` and end of input interrupt only a
//! `go infinite`; a timed or depth-limited search is left to finish first.

#[allow(dead_code)]
#[path = "../../rust/src/classical.rs"]
mod classical;
mod net;
mod search;
#[path = "../../rust/src/tables.rs"]
mod tables;

use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Move, Position};

use search::{Engine, GoParams, Params};

const TO_FILE: &str = "chess.safetensors";
const FROM_FILE: &str = "chess_from.safetensors";

fn send(line: &str) {
    // stdout is a pipe under lichess-bot; flush every line.
    let out = io::stdout();
    let mut out = out.lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// The working directory first (run.sh cds to the checkout), then the binary's
/// directory and its parents, so target/release/nightybot finds the repo root.
fn find_models(dir: Option<PathBuf>) -> Result<PathBuf, String> {
    let mut roots: Vec<PathBuf> = Vec::new();
    match dir {
        Some(d) => roots.push(d),
        None => {
            roots.push(PathBuf::from("."));
            if let Ok(exe) = std::env::current_exe() {
                let mut p = exe.parent();
                for _ in 0..4 {
                    if let Some(d) = p {
                        roots.push(d.to_path_buf());
                        p = d.parent();
                    }
                }
            }
        }
    }
    roots
        .into_iter()
        .find(|r| r.join(TO_FILE).is_file() && r.join(FROM_FILE).is_file())
        .ok_or_else(|| format!("cannot find {TO_FILE} and {FROM_FILE} (run save.py; or pass --models DIR)"))
}

fn load_engine(dir: &Path) -> Result<Engine, String> {
    Ok(Engine {
        to: net::ToNet::load(&dir.join(TO_FILE))?,
        from: net::FromNet::load(&dir.join(FROM_FILE))?,
    })
}

/// Nearly as lenient as python-chess, which run.py used: odd material, stale
/// castling rights, an impossible check -- play on anyway. The one thing
/// shakmaty will not do is generate moves with the side *not* to move in
/// check (an "opposite check"), and that position is refused.
fn parse_fen(fen: &str) -> Result<Chess, String> {
    let setup: Fen = fen.parse().map_err(|e: shakmaty::fen::ParseFenError| e.to_string())?;
    setup
        .into_position(CastlingMode::Standard)
        .or_else(|e| e.ignore_invalid_castling_rights())
        .or_else(|e| e.ignore_invalid_ep_square())
        .or_else(|e| e.ignore_too_much_material())
        .or_else(|e| e.ignore_impossible_check())
        .map_err(|e: shakmaty::PositionError<Chess>| e.to_string())
}

fn parse_move(pos: &Chess, uci: &str) -> Result<Move, String> {
    let m = UciMove::from_ascii(uci.as_bytes()).map_err(|e| e.to_string())?;
    m.to_move(pos).map_err(|e| e.to_string())
}

/// The game so far: the position and every position hash on the way to it,
/// including the current one, for repetition detection.
struct Game {
    pos: Chess,
    history: HashSet<u64>,
}

impl Game {
    fn new(pos: Chess) -> Self {
        let mut history = HashSet::new();
        history.insert(search::hash_of(&pos));
        Game { pos, history }
    }

    fn play(&mut self, m: &Move) {
        self.pos.play_unchecked(m);
        self.history.insert(search::hash_of(&self.pos));
    }
}

fn set_position(args: &[&str]) -> Option<Game> {
    let mut i = 2;
    let start = match args.get(1) {
        Some(&"startpos") => Chess::default(),
        Some(&"fen") => {
            let mut fields = Vec::new();
            while i < args.len() && args[i] != "moves" {
                fields.push(args[i]);
                i += 1;
            }
            match parse_fen(&fields.join(" ")) {
                Ok(p) => p,
                Err(e) => {
                    send(&format!("info string bad fen: {e}"));
                    return None;
                }
            }
        }
        _ => return None,
    };
    let mut game = Game::new(start);
    if args.get(i) == Some(&"moves") {
        for uci in &args[i + 1..] {
            match parse_move(&game.pos, uci) {
                Ok(m) => game.play(&m),
                Err(e) => {
                    send(&format!("info string bad move {uci}: {e}"));
                    return None;
                }
            }
        }
    }
    Some(game)
}

fn parse_go(args: &[&str]) -> GoParams {
    let mut g = GoParams::default();
    let mut i = 1;
    while i < args.len() {
        let value = args.get(i + 1).and_then(|v| v.parse::<i64>().ok());
        let slot = match args[i] {
            "wtime" => Some(&mut g.wtime),
            "btime" => Some(&mut g.btime),
            "winc" => Some(&mut g.winc),
            "binc" => Some(&mut g.binc),
            "movestogo" => Some(&mut g.movestogo),
            "movetime" => Some(&mut g.movetime),
            "depth" => Some(&mut g.depth),
            "infinite" => {
                g.infinite = true;
                None
            }
            _ => None,
        };
        match slot {
            Some(slot) => {
                *slot = value;
                i += 2;
            }
            None => i += 1,
        }
    }
    g
}

fn set_option(params: &mut Params, cmd: &str) {
    let lower = cmd.to_ascii_lowercase();
    let Some(rest) = lower.strip_prefix("setoption") else { return };
    let rest = rest.trim_start();
    let Some(rest) = rest.strip_prefix("name") else { return };
    let Some((name, value)) = rest.split_once(" value ") else { return };
    let (name, value) = (name.trim(), value.trim());
    let Ok(v) = value.parse::<i64>() else { return };
    match name {
        "move overhead" => params.move_overhead = v.max(0) as f64 / 1000.0,
        "depth" => params.depth = v.max(1) as u32,
        "analyzemoves" => params.moves = v.max(1) as usize,
        "analyzepieces" => params.pieces = v.max(1) as usize,
        _ => return,
    }
    send(&format!("info string {name} set to {value}"));
}

/// A `go` in flight. The thread prints its own bestmove unless the go was
/// infinite, in which case `stop` does.
struct Running {
    handle: thread::JoinHandle<Option<Move>>,
    stop: Arc<AtomicBool>,
    infinite: bool,
}

struct Uci {
    engine: Arc<Engine>,
    params: Params,
    game: Game,
    last_move: Option<Move>,
    running: Option<Running>,
}

impl Uci {
    /// Wait for the search in flight, if any, remember its answer, and say
    /// whether that go was infinite.
    fn join(&mut self) -> Option<bool> {
        let r = self.running.take()?;
        if let Some(m) = r.handle.join().unwrap_or(None) {
            self.last_move = Some(m);
        }
        Some(r.infinite)
    }

    fn go(&mut self, cmd: &str) {
        let args: Vec<&str> = cmd.split_whitespace().collect();
        let g = parse_go(&args);
        let budget = (!g.infinite).then(|| search::time_budget(self.game.pos.turn(), &g, self.params.move_overhead));
        let limit = g.depth.map_or(self.params.depth, |d| d.max(1) as u32);
        let stop = Arc::new(AtomicBool::new(false));
        let engine = Arc::clone(&self.engine);
        let pos = self.game.pos.clone();
        let history = self.game.history.clone();
        let params = self.params;
        let infinite = g.infinite;
        let flag = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let answer = search::find_best_move(&engine, &pos, &history, params, limit, budget, &flag, &mut |line| {
                send(&line)
            });
            let Some(answer) = answer else {
                send("info string no legal moves");
                send("bestmove (none)");
                return None;
            };
            if !infinite {
                send(&format!("bestmove {}", search::uci(&answer.best)));
            }
            Some(answer.best)
        });
        self.running = Some(Running { handle, stop, infinite });
    }

    fn stop(&mut self) {
        let Some(r) = &self.running else { return };
        r.stop.store(true, Ordering::Relaxed);
        if self.join() == Some(true) {
            match &self.last_move {
                Some(m) => send(&format!("bestmove {}", search::uci(m))),
                None => send("bestmove (none)"),
            }
        }
    }

    /// What `quit` and end of input do to a search in flight: an infinite one
    /// is stopped and answered, any other is allowed to finish, so a piped
    /// `go` followed by `quit` still gets its full search (run_rl.py does the
    /// same).
    fn finish(&mut self) {
        if self.running.as_ref().is_some_and(|r| r.infinite) {
            self.stop();
        } else {
            self.join();
        }
    }

    fn command(&mut self, cmd: &str) -> bool {
        let first = cmd.split_whitespace().next().unwrap_or("");
        match first {
            "uci" => {
                send("id name NightyBot");
                send("id author Nighty");
                send(&format!("option name Depth type spin default {} min 1 max 32", self.params.depth));
                send(&format!("option name AnalyzeMoves type spin default {} min 1 max 24", self.params.moves));
                send(&format!("option name AnalyzePieces type spin default {} min 1 max 16", self.params.pieces));
                send(&format!(
                    "option name Move Overhead type spin default {} min 0 max 5000",
                    (self.params.move_overhead * 1000.0) as i64
                ));
                send("uciok");
            }
            "isready" => send("readyok"),
            "stop" => self.stop(),
            "quit" => {
                self.finish();
                return false;
            }
            _ => {
                // Everything else changes state the search is reading.
                self.join();
                match first {
                    "ucinewgame" => self.game = Game::new(Chess::default()),
                    "position" => {
                        let args: Vec<&str> = cmd.split_whitespace().collect();
                        if let Some(game) = set_position(&args) {
                            self.game = game;
                        }
                    }
                    "go" => self.go(cmd),
                    "setoption" => set_option(&mut self.params, cmd),
                    _ => send("info string Unrecognized Command"),
                }
            }
        }
        true
    }
}

fn run_uci(engine: Engine) {
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut uci = Uci {
        engine: Arc::new(engine),
        params: Params::default(),
        game: Game::new(Chess::default()),
        last_move: None,
        running: None,
    };
    while let Ok(line) = rx.recv() {
        let cmd = line.trim();
        if cmd.is_empty() {
            continue;
        }
        if !uci.command(cmd) {
            return;
        }
    }
    // stdin closed
    uci.finish();
}

fn bench(engine: &Engine, depth: u32) {
    let positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
        "r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 0 10",
        "8/5k2/8/8/8/1K6/8/4R3 w - - 0 1",
    ];
    let stop = AtomicBool::new(false);
    let mut total_nodes = 0u64;
    let started = Instant::now();
    for fen in positions {
        let pos = parse_fen(fen).expect("bench fen");
        let mut game = Game::new(pos);
        let mut nodes = 0;
        let t = Instant::now();
        let answer = search::find_best_move(
            &engine,
            &game.pos,
            &game.history,
            Params::default(),
            depth,
            None,
            &stop,
            &mut |line| {
                if let Some(n) = line.split(" nodes ").nth(1).and_then(|s| s.split(' ').next()) {
                    nodes = n.parse().unwrap_or(0);
                }
            },
        )
        .expect("bench position has moves");
        let dt = t.elapsed().as_secs_f64();
        total_nodes += nodes;
        println!(
            "{:<70} depth {depth}: {} score {} {nodes} nodes in {dt:.2}s -> {:.0} nodes/s",
            fen,
            search::uci(&answer.best),
            search::score_string(answer.score),
            nodes as f64 / dt
        );
        game.play(&answer.best);
    }
    let dt = started.elapsed().as_secs_f64();
    println!("total {total_nodes} nodes in {dt:.2}s -> {:.0} nodes/s", total_nodes as f64 / dt);
}

fn logits(engine: &Engine, fen: &str) {
    let pos = parse_fen(fen).expect("logits fen");
    let board = classical::fen_order(&pos);
    let from = engine.from.forward(&board);
    println!("from {}", from.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
    let sources: Vec<usize> = (0..64).collect();
    let to = engine.to.forward(&board, &sources);
    for (src, row) in to.chunks_exact(64).enumerate() {
        println!("to {src} {}", row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
    }
}

fn candidates(engine: &Engine, fen: &str) {
    let pos = parse_fen(fen).expect("candidates fen");
    let legals: Vec<Move> = pos.legal_moves().into_iter().collect();
    let picked = search::candidates(engine, &pos, &legals, &Params::default());
    println!("{}", picked.iter().map(search::uci).collect::<Vec<_>>().join(" "));
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut models = None;
    if let Some(i) = args.iter().position(|a| a == "--models") {
        if i + 1 >= args.len() {
            eprintln!("--models needs a directory");
            std::process::exit(2);
        }
        models = Some(PathBuf::from(&args[i + 1]));
        args.drain(i..i + 2);
    }
    let dir = match find_models(models) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let engine = match load_engine(&dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    match args.first().map(String::as_str) {
        Some("bench") => bench(&engine, args.get(1).and_then(|d| d.parse().ok()).unwrap_or(4)),
        Some("logits") => logits(&engine, &args[1..].join(" ")),
        Some("candidates") => candidates(&engine, &args[1..].join(" ")),
        Some(other) => {
            eprintln!("unknown command {other}");
            std::process::exit(2);
        }
        None => run_uci(engine),
    }
}
