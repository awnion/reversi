//! Inference for the AlphaZero-style network, tuned for the browser.
//!
//! The network is evaluated once per MCTS simulation on a single CPU thread
//! (WASM), so the forward pass is the bot's whole time budget. Compared with
//! the layer-by-layer [`crate::reference`] implementation this one:
//!
//! - keeps activations channel-last (`[square][channel]`), so the innermost
//!   loop runs over output channels of one contiguous weight row and
//!   auto-vectorises (SIMD128 on wasm32);
//! - folds every BatchNorm into the preceding convolution at load time;
//! - skips zero inputs, which are common after ReLU and in the input planes;
//! - works in fixed-size stack buffers, without allocating.

use crate::layers::softmax;
use crate::weights::WeightStore;

const BOARD: usize = 8;
const SQUARES: usize = BOARD * BOARD;
/// Input planes: my discs, opponent discs, legal moves, phase.
const IN_PLANES: usize = 4;
/// Width of the residual tower.
const CHANNELS: usize = 32;
const BLOCKS: usize = 4;
const POLICY_CHANNELS: usize = 2;
const VALUE_HIDDEN: usize = 64;
const BN_EPS: f32 = 1e-5;

fn tensor<'a>(
    store: &'a WeightStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<&'a [f32], String> {
    let (data, shape) = store.get(name).ok_or_else(|| format!("missing weight: '{name}'"))?;
    if shape != expected_shape {
        return Err(format!("tensor '{name}' has shape {shape:?}, expected {expected_shape:?}"));
    }
    Ok(data)
}

// ── Conv ─────────────────────────────────────────────────────────────────────

/// Convolution over the 8×8 board with "same" padding and BatchNorm folded in.
struct Conv {
    in_ch: usize,
    out_ch: usize,
    ksize: usize,
    /// `[kh][kw][in][out]`: one contiguous row of `out_ch` weights per input value.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv {
    /// Load `{conv}.weight`/`.bias` (MLX layout `[out, kH, kW, in]`) and fold
    /// the BatchNorm stored under `{bn}` into them.
    fn load(
        store: &WeightStore,
        conv: &str,
        bn: &str,
        in_ch: usize,
        out_ch: usize,
        ksize: usize,
    ) -> Result<Self, String> {
        let w = tensor(store, &format!("{conv}.weight"), &[out_ch, ksize, ksize, in_ch])?;
        let b = tensor(store, &format!("{conv}.bias"), &[out_ch])?;
        let gamma = tensor(store, &format!("{bn}.weight"), &[out_ch])?;
        let beta = tensor(store, &format!("{bn}.bias"), &[out_ch])?;
        let mean = tensor(store, &format!("{bn}.running_mean"), &[out_ch])?;
        let var = tensor(store, &format!("{bn}.running_var"), &[out_ch])?;

        // bn(conv(x)) = scale * (w·x + b - mean) + beta
        let scale: Vec<f32> =
            gamma.iter().zip(var).map(|(&g, &v)| g / (v + BN_EPS).sqrt()).collect();

        let mut weight = vec![0.0f32; w.len()];
        for (o, &scale) in scale.iter().enumerate() {
            for kh in 0..ksize {
                for kw in 0..ksize {
                    for i in 0..in_ch {
                        let src = ((o * ksize + kh) * ksize + kw) * in_ch + i;
                        let dst = ((kh * ksize + kw) * in_ch + i) * out_ch + o;
                        weight[dst] = w[src] * scale;
                    }
                }
            }
        }
        let bias = (0..out_ch).map(|o| (b[o] - mean[o]) * scale[o] + beta[o]).collect();

        Ok(Conv { in_ch, out_ch, ksize, weight, bias })
    }

    #[cfg(feature = "test-utils")]
    fn zeros(in_ch: usize, out_ch: usize, ksize: usize) -> Self {
        Conv {
            in_ch,
            out_ch,
            ksize,
            weight: vec![0.0; ksize * ksize * in_ch * out_ch],
            bias: vec![0.0; out_ch],
        }
    }

    /// `input`: `[square][in_ch]`, `output`: `[square][out_ch]`.
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let (in_ch, out_ch, ksize) = (self.in_ch, self.out_ch, self.ksize);
        debug_assert_eq!(input.len(), SQUARES * in_ch);
        debug_assert_eq!(output.len(), SQUARES * out_ch);
        let pad = ksize / 2;

        for (square, out) in output.chunks_exact_mut(out_ch).enumerate() {
            out.copy_from_slice(&self.bias);
            let (row, col) = (square / BOARD, square % BOARD);

            for kh in 0..ksize {
                // Source row `row + kh - pad`, skipped when off the board.
                let Some(src_row) = (row + kh).checked_sub(pad).filter(|&r| r < BOARD) else {
                    continue;
                };
                for kw in 0..ksize {
                    let Some(src_col) = (col + kw).checked_sub(pad).filter(|&c| c < BOARD) else {
                        continue;
                    };
                    let src = (src_row * BOARD + src_col) * in_ch;
                    let taps = &self.weight[(kh * ksize + kw) * in_ch * out_ch..][..in_ch * out_ch];

                    for (&x, weights) in
                        input[src..src + in_ch].iter().zip(taps.chunks_exact(out_ch))
                    {
                        if x == 0.0 {
                            continue;
                        }
                        for (o, &w) in out.iter_mut().zip(weights) {
                            *o += x * w;
                        }
                    }
                }
            }
        }
    }
}

// ── Dense ────────────────────────────────────────────────────────────────────

/// Fully-connected layer `y = W x + b`.
struct Dense {
    in_f: usize,
    out_f: usize,
    /// `[in][out]`: one contiguous row of `out_f` weights per input value.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Dense {
    /// Load `{name}.weight` (`[out, in]`) and `{name}.bias`.
    fn load(store: &WeightStore, name: &str, in_f: usize, out_f: usize) -> Result<Self, String> {
        let w = tensor(store, &format!("{name}.weight"), &[out_f, in_f])?;
        let b = tensor(store, &format!("{name}.bias"), &[out_f])?;
        let mut weight = vec![0.0f32; w.len()];
        for o in 0..out_f {
            for i in 0..in_f {
                weight[i * out_f + o] = w[o * in_f + i];
            }
        }
        Ok(Dense { in_f, out_f, weight, bias: b.to_vec() })
    }

    #[cfg(feature = "test-utils")]
    fn zeros(in_f: usize, out_f: usize) -> Self {
        Dense { in_f, out_f, weight: vec![0.0; in_f * out_f], bias: vec![0.0; out_f] }
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_f);
        debug_assert_eq!(output.len(), self.out_f);
        output.copy_from_slice(&self.bias);
        for (&x, weights) in input.iter().zip(self.weight.chunks_exact(self.out_f)) {
            if x == 0.0 {
                continue;
            }
            for (o, &w) in output.iter_mut().zip(weights) {
                *o += x * w;
            }
        }
    }
}

fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

// ── AlphaZeroNet ─────────────────────────────────────────────────────────────

/// AlphaZero-style network for 8×8 Reversi: a 3×3 stem, four residual blocks
/// of 32 channels, and policy/value heads.
pub struct AlphaZeroNet {
    stem: Conv,
    /// `(conv1, conv2)` of each residual block.
    blocks: Vec<(Conv, Conv)>,
    policy_conv: Conv,
    policy_fc: Dense,
    value_conv: Conv,
    value_fc1: Dense,
    value_fc2: Dense,
}

impl AlphaZeroNet {
    /// Load the network from the `.bin` file written by `python/train/export.py`.
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        let store = WeightStore::load(bytes)?;
        Self::from_store(&store)
    }

    /// Build the network from an already-parsed `WeightStore` holding the MLX
    /// parameter tree (MLX names, channel-last convolution weights).
    pub fn from_store(store: &WeightStore) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let conv = |n: u8| {
                Conv::load(
                    store,
                    &format!("blocks.{i}.conv{n}"),
                    &format!("blocks.{i}.bn{n}"),
                    CHANNELS,
                    CHANNELS,
                    3,
                )
            };
            blocks.push((conv(1)?, conv(2)?));
        }

        Ok(AlphaZeroNet {
            stem: Conv::load(store, "stem.layers.0", "stem.layers.1", IN_PLANES, CHANNELS, 3)?,
            blocks,
            policy_conv: Conv::load(
                store,
                "policy_conv",
                "policy_bn",
                CHANNELS,
                POLICY_CHANNELS,
                1,
            )?,
            // The training model flattens the policy head as (H, W, C), which
            // is exactly this module's activation layout.
            policy_fc: Dense::load(store, "policy_fc", SQUARES * POLICY_CHANNELS, SQUARES)?,
            value_conv: Conv::load(store, "value_conv", "value_bn", CHANNELS, 1, 1)?,
            value_fc1: Dense::load(store, "value_fc1", SQUARES, VALUE_HIDDEN)?,
            value_fc2: Dense::load(store, "value_fc2", VALUE_HIDDEN, 1)?,
        })
    }

    /// Forward pass.
    ///
    /// `planes`: channel-first f32 slice of length `4 * 8 * 8 = 256`.
    /// Planes: my_pieces, opp_pieces, legal_moves, phase (count_occupied/64).
    ///
    /// Returns `(policy, value)` where `policy` sums to 1 (softmax applied)
    /// and `value` is in `[-1, 1]` (tanh applied).
    pub fn forward(&self, planes: &[f32]) -> ([f32; 64], f32) {
        assert_eq!(planes.len(), IN_PLANES * SQUARES, "expected 4×8×8 input");

        // (C, H, W) → [square][channel]
        let mut input = [0.0f32; SQUARES * IN_PLANES];
        for (c, plane) in planes.as_chunks::<SQUARES>().0.iter().enumerate() {
            for (square, &v) in plane.iter().enumerate() {
                input[square * IN_PLANES + c] = v;
            }
        }

        // ── Stem and residual tower ───────────────────────────────────────────
        let mut x = [0.0f32; SQUARES * CHANNELS];
        let mut hidden = [0.0f32; SQUARES * CHANNELS];
        let mut out = [0.0f32; SQUARES * CHANNELS];

        self.stem.forward(&input, &mut x);
        relu(&mut x);

        for (conv1, conv2) in &self.blocks {
            conv1.forward(&x, &mut hidden);
            relu(&mut hidden);
            conv2.forward(&hidden, &mut out);
            for (o, &residual) in out.iter_mut().zip(&x) {
                *o = (*o + residual).max(0.0);
            }
            std::mem::swap(&mut x, &mut out);
        }

        // ── Policy head ───────────────────────────────────────────────────────
        let mut p = [0.0f32; SQUARES * POLICY_CHANNELS];
        self.policy_conv.forward(&x, &mut p);
        relu(&mut p);
        let mut policy = [0.0f32; SQUARES];
        self.policy_fc.forward(&p, &mut policy);
        softmax(&mut policy);

        // ── Value head ────────────────────────────────────────────────────────
        let mut v = [0.0f32; SQUARES];
        self.value_conv.forward(&x, &mut v);
        relu(&mut v);
        let mut h = [0.0f32; VALUE_HIDDEN];
        self.value_fc1.forward(&v, &mut h);
        relu(&mut h);
        let mut value = [0.0f32; 1];
        self.value_fc2.forward(&h, &mut value);

        (policy, value[0].tanh())
    }

    /// Construct an `AlphaZeroNet` with all-zeros weights — useful for testing
    /// that the network topology is valid without needing real weight files.
    ///
    /// Available only when the `test-utils` feature is enabled.
    #[cfg(feature = "test-utils")]
    pub fn zeros() -> Self {
        AlphaZeroNet {
            stem: Conv::zeros(IN_PLANES, CHANNELS, 3),
            blocks: (0..BLOCKS)
                .map(|_| (Conv::zeros(CHANNELS, CHANNELS, 3), Conv::zeros(CHANNELS, CHANNELS, 3)))
                .collect(),
            policy_conv: Conv::zeros(CHANNELS, POLICY_CHANNELS, 1),
            policy_fc: Dense::zeros(SQUARES * POLICY_CHANNELS, SQUARES),
            value_conv: Conv::zeros(CHANNELS, 1, 1),
            value_fc1: Dense::zeros(SQUARES, VALUE_HIDDEN),
            value_fc2: Dense::zeros(VALUE_HIDDEN, 1),
        }
    }
}
