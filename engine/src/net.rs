//! The two imitation nets from run.py, run on the CPU from the safetensors
//! files that save.py exports. Both are the same trunk -- one input layer,
//! three residual GELU + LayerNorm layers of 620, one output layer -- and
//! differ only in what goes in and comes out:
//!
//!  * `FromNet` (chess_from.safetensors): 64 squares x 16-dim embedding ->
//!    128 logits, (side, square): which piece to move, row 0 White, row 1 Black.
//!  * `ToNet` (chess.safetensors): the same 1024 plus a 64-dim embedding of
//!    the source square -> 64 logits: where that piece should go.
//!
//! Squares in the board input are FEN reading order (index 0 = a8), the way
//! run.py's encode() lays them out; squares in the outputs and in the source
//! embedding are python-chess numbered (a1 = 0). See CLAUDE.md.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Reads a safetensors file: 8 bytes of little-endian header length, a JSON
/// header, then the raw tensor bytes. Only F32 tensors, which is all save.py
/// writes.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, Tensor>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 8 {
        return Err(format!("{}: too short to be a safetensors file", path.display()));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = bytes
        .get(8..8 + n)
        .ok_or_else(|| format!("{}: header runs past the end of the file", path.display()))?;
    let header: serde_json::Value =
        serde_json::from_slice(header).map_err(|e| format!("{}: bad header: {e}", path.display()))?;
    let base = 8 + n;
    let mut out = HashMap::new();
    for (name, info) in header.as_object().ok_or("safetensors header is not an object")? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"].as_str().unwrap_or("?");
        if dtype != "F32" {
            return Err(format!("{name}: dtype {dtype}, only F32 is supported"));
        }
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .ok_or_else(|| format!("{name}: missing shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let offsets = info["data_offsets"]
            .as_array()
            .ok_or_else(|| format!("{name}: missing data_offsets"))?;
        let (a, b) = (
            offsets[0].as_u64().unwrap_or(0) as usize,
            offsets[1].as_u64().unwrap_or(0) as usize,
        );
        let raw = bytes
            .get(base + a..base + b)
            .ok_or_else(|| format!("{name}: data runs past the end of the file"))?;
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if data.len() != shape.iter().product::<usize>() {
            return Err(format!("{name}: {} values for shape {shape:?}", data.len()));
        }
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok(out)
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let t = tensors.remove(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if t.shape != shape {
        return Err(format!("{name}: shape {:?}, expected {shape:?}", t.shape));
    }
    Ok(t.data)
}

/// Sixteen partial sums so the compiler can vectorise the loop without being
/// allowed to reassociate a single accumulator.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (ac, ar) = a.as_chunks::<16>();
    let (bc, br) = b.as_chunks::<16>();
    let mut acc = [0f32; 16];
    for (x, y) in ac.iter().zip(bc) {
        for k in 0..16 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ar.iter().zip(br) {
        tail += x * y;
    }
    acc.iter().sum::<f32>() + tail
}

pub struct Linear {
    w: Vec<f32>, // (out, inp), row-major like torch
    b: Vec<f32>,
    pub inp: usize,
    pub out: usize,
}

impl Linear {
    fn load(t: &mut HashMap<String, Tensor>, name: &str, inp: usize, out: usize) -> Result<Self, String> {
        Ok(Linear {
            w: take(t, &format!("{name}_weight"), &[out, inp])?,
            b: take(t, &format!("{name}_bias"), &[out])?,
            inp,
            out,
        })
    }

    /// y = x W^T + b for `rows` rows of x, both contiguous. Each weight row is
    /// streamed once and used against every input row while it is hot; the
    /// input rows are small enough to stay in L1.
    fn forward(&self, x: &[f32], rows: usize, y: &mut [f32]) {
        debug_assert_eq!(x.len(), rows * self.inp);
        debug_assert_eq!(y.len(), rows * self.out);
        for (j, wj) in self.w.chunks_exact(self.inp).enumerate() {
            for r in 0..rows {
                y[r * self.out + j] = dot(wj, &x[r * self.inp..(r + 1) * self.inp]) + self.b[j];
            }
        }
    }
}

pub struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    const EPS: f32 = 1e-5; // torch's default

    fn load(t: &mut HashMap<String, Tensor>, name: &str, n: usize) -> Result<Self, String> {
        Ok(LayerNorm {
            w: take(t, &format!("{name}_weight"), &[n])?,
            b: take(t, &format!("{name}_bias"), &[n])?,
        })
    }

    fn forward(&self, x: &mut [f32]) {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + Self::EPS).sqrt();
        for ((v, w), b) in x.iter_mut().zip(&self.w).zip(&self.b) {
            *v = (*v - mean) * inv * w + b;
        }
    }
}

/// torch.nn.GELU(): the exact erf form, not the tanh approximation.
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// f1 -> three residual layers -> f5, one LayerNorm shared by all of them,
/// exactly as ChessModel.forward / ChessFromModel.forward in run.py.
pub struct Trunk {
    f1: Linear,
    f2: Linear,
    f3: Linear,
    f4: Linear,
    f5: Linear,
    norm: LayerNorm,
}

impl Trunk {
    fn load(t: &mut HashMap<String, Tensor>, inp: usize, feature: usize, out: usize) -> Result<Self, String> {
        Ok(Trunk {
            f1: Linear::load(t, "f1", inp, feature)?,
            f2: Linear::load(t, "f2", feature, feature)?,
            f3: Linear::load(t, "f3", feature, feature)?,
            f4: Linear::load(t, "f4", feature, feature)?,
            f5: Linear::load(t, "f5", feature, out)?,
            norm: LayerNorm::load(t, "layer_norm", feature)?,
        })
    }

    /// `rows` rows of f1.inp in, `rows` rows of f5.out back.
    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let f = self.f1.out;
        let mut h = vec![0f32; rows * f];
        self.f1.forward(x, rows, &mut h);
        for row in h.chunks_exact_mut(f) {
            row.iter_mut().for_each(|v| *v = gelu(*v));
            self.norm.forward(row);
        }
        let mut t = vec![0f32; rows * f];
        for layer in [&self.f2, &self.f3, &self.f4] {
            layer.forward(&h, rows, &mut t);
            for (hrow, trow) in h.chunks_exact_mut(f).zip(t.chunks_exact_mut(f)) {
                trow.iter_mut().for_each(|v| *v = gelu(*v));
                self.norm.forward(trow);
                for (a, b) in hrow.iter_mut().zip(trow.iter()) {
                    *a += b;
                }
            }
        }
        let mut out = vec![0f32; rows * self.f5.out];
        self.f5.forward(&h, rows, &mut out);
        out
    }
}

pub const FEATURE: usize = 620;
const EMBED: usize = 16;
const PIECE_EMBED: usize = 64;
const N_CODES: usize = 13; // ".prnbqkPRNBQK"

fn embed_board(em: &[f32], board: &[u8; 64], x: &mut [f32]) {
    for (sq, &code) in board.iter().enumerate() {
        let c = code as usize;
        x[sq * EMBED..(sq + 1) * EMBED].copy_from_slice(&em[c * EMBED..(c + 1) * EMBED]);
    }
}

pub struct FromNet {
    em_board: Vec<f32>,
    trunk: Trunk,
}

impl FromNet {
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut t = load_safetensors(path)?;
        Ok(FromNet {
            em_board: take(&mut t, "em_board", &[N_CODES, EMBED])?,
            trunk: Trunk::load(&mut t, 64 * EMBED, FEATURE, 128)?,
        })
    }

    /// 128 logits: [side * 64 + square], side 0 = White, 1 = Black.
    pub fn forward(&self, board: &[u8; 64]) -> Vec<f32> {
        let mut x = vec![0f32; 64 * EMBED];
        embed_board(&self.em_board, board, &mut x);
        self.trunk.forward(&x, 1)
    }
}

pub struct ToNet {
    em_board: Vec<f32>,
    em_piece: Vec<f32>,
    trunk: Trunk,
}

impl ToNet {
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut t = load_safetensors(path)?;
        Ok(ToNet {
            em_board: take(&mut t, "em_board", &[N_CODES, EMBED])?,
            em_piece: take(&mut t, "em_piece", &[64, PIECE_EMBED])?,
            trunk: Trunk::load(&mut t, 64 * EMBED + PIECE_EMBED, FEATURE, 64)?,
        })
    }

    /// One row of 64 destination logits per source square, in the order given.
    pub fn forward(&self, board: &[u8; 64], sources: &[usize]) -> Vec<f32> {
        let inp = 64 * EMBED + PIECE_EMBED;
        let mut x = vec![0f32; sources.len() * inp];
        for (r, &src) in sources.iter().enumerate() {
            let row = &mut x[r * inp..(r + 1) * inp];
            embed_board(&self.em_board, board, &mut row[..64 * EMBED]);
            row[64 * EMBED..].copy_from_slice(&self.em_piece[src * PIECE_EMBED..(src + 1) * PIECE_EMBED]);
        }
        self.trunk.forward(&x, sources.len())
    }
}
