use crate::layers::BatchNorm2d;
use crate::layers::Conv2d;
use crate::layers::Linear;
use crate::layers::relu;
use crate::layers::softmax;
use crate::layers::tanh_inplace;
use crate::weights::WeightStore;

// ── Constants ────────────────────────────────────────────────────────────────

const BOARD: usize = 8;
const BN_EPS: f32 = 1e-5;

// ── Helper macros ─────────────────────────────────────────────────────────────

/// Require a tensor from the store, returning an error if it's absent.
macro_rules! require {
    ($store:expr, $name:expr) => {
        $store.get($name).ok_or_else(|| format!("missing weight: '{}'", $name))?
    };
}

/// Load a Conv2d from the weight store.
///
/// Expects tensors `{prefix}.weight` and optionally `{prefix}.bias`.
fn load_conv(
    store: &WeightStore,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    ksize: usize,
    pad: usize,
) -> Result<Conv2d, String> {
    let (w, _) = require!(store, &format!("{prefix}.weight"));
    let bias = store.get(&format!("{prefix}.bias")).map(|(b, _)| b.to_vec());
    Ok(Conv2d::new(w.to_vec(), bias, in_ch, out_ch, ksize, pad))
}

/// Load a BatchNorm2d from the weight store.
///
/// Expects `{prefix}.gamma`, `{prefix}.beta`, `{prefix}.mean`, `{prefix}.var`.
fn load_bn(store: &WeightStore, prefix: &str) -> Result<BatchNorm2d, String> {
    let (gamma, _) = require!(store, &format!("{prefix}.gamma"));
    let (beta, _) = require!(store, &format!("{prefix}.beta"));
    let (mean, _) = require!(store, &format!("{prefix}.mean"));
    let (var, _) = require!(store, &format!("{prefix}.var"));
    Ok(BatchNorm2d::new(gamma.to_vec(), beta.to_vec(), mean.to_vec(), var.to_vec(), BN_EPS))
}

/// Load a Linear from the weight store.
///
/// Expects `{prefix}.weight` and `{prefix}.bias`.
fn load_linear(
    store: &WeightStore,
    prefix: &str,
    in_f: usize,
    out_f: usize,
) -> Result<Linear, String> {
    let (w, _) = require!(store, &format!("{prefix}.weight"));
    let (b, _) = require!(store, &format!("{prefix}.bias"));
    Ok(Linear::new(w.to_vec(), b.to_vec(), in_f, out_f))
}

// ── ResBlock ─────────────────────────────────────────────────────────────────

struct ResBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
}

impl ResBlock {
    fn load(store: &WeightStore, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            conv1: load_conv(store, &format!("{prefix}.conv1"), 32, 32, 3, 1)?,
            bn1: load_bn(store, &format!("{prefix}.bn1"))?,
            conv2: load_conv(store, &format!("{prefix}.conv2"), 32, 32, 3, 1)?,
            bn2: load_bn(store, &format!("{prefix}.bn2"))?,
        })
    }

    /// `x` is `(32, 8, 8)`. Returns the same shape.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let residual = x.to_vec();

        let mut out = self.conv1.forward(x, BOARD, BOARD);
        out = self.bn1.forward(&out, BOARD, BOARD);
        relu(&mut out);

        out = self.conv2.forward(&out, BOARD, BOARD);
        out = self.bn2.forward(&out, BOARD, BOARD);

        // Add residual
        for (o, r) in out.iter_mut().zip(residual.iter()) {
            *o += r;
        }
        relu(&mut out);

        out
    }
}

// ── AlphaZeroNet ─────────────────────────────────────────────────────────────

/// AlphaZero-style network for 8×8 Reversi.
///
/// Input: 3 channel-first planes of shape `(3, 8, 8)`.
/// Output: `(policy: [f32; 64], value: f32)`.
pub struct AlphaZeroNet {
    // Stem
    stem_conv: Conv2d,
    stem_bn: BatchNorm2d,

    // Residual body (4 blocks)
    blocks: Vec<ResBlock>,

    // Policy head
    pol_conv: Conv2d,
    pol_bn: BatchNorm2d,
    pol_fc: Linear,

    // Value head
    val_conv: Conv2d,
    val_bn: BatchNorm2d,
    val_fc1: Linear,
    val_fc2: Linear,
}

impl AlphaZeroNet {
    /// Load the network from raw bytes in the binary weight format.
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        let store = WeightStore::load(bytes)?;
        Self::from_store(&store)
    }

    /// Build the network from an already-parsed `WeightStore`.
    pub fn from_store(store: &WeightStore) -> Result<Self, String> {
        let stem_conv = load_conv(store, "stem.conv", 3, 32, 3, 1)?;
        let stem_bn = load_bn(store, "stem.bn")?;

        let mut blocks = Vec::with_capacity(4);
        for i in 0..4 {
            blocks.push(ResBlock::load(store, &format!("block{i}"))?);
        }

        // Policy head: Conv2d(32→2, 1×1) → BN(2) → Flatten → Linear(128→64)
        let pol_conv = load_conv(store, "policy.conv", 32, 2, 1, 0)?;
        let pol_bn = load_bn(store, "policy.bn")?;
        let pol_fc = load_linear(store, "policy.fc", 2 * BOARD * BOARD, 64)?;

        // Value head: Conv2d(32→1, 1×1) → BN(1) → Flatten → Linear(64→64) → Linear(64→1)
        let val_conv = load_conv(store, "value.conv", 32, 1, 1, 0)?;
        let val_bn = load_bn(store, "value.bn")?;
        let val_fc1 = load_linear(store, "value.fc1", BOARD * BOARD, 64)?;
        let val_fc2 = load_linear(store, "value.fc2", 64, 1)?;

        Ok(Self {
            stem_conv,
            stem_bn,
            blocks,
            pol_conv,
            pol_bn,
            pol_fc,
            val_conv,
            val_bn,
            val_fc1,
            val_fc2,
        })
    }

    /// Forward pass.
    ///
    /// `planes`: channel-first f32 slice of length `3 * 8 * 8 = 192`.
    ///
    /// Returns `(policy, value)` where `policy` sums to 1 (softmax applied)
    /// and `value` is in `[-1, 1]` (tanh applied).
    pub fn forward(&self, planes: &[f32]) -> ([f32; 64], f32) {
        assert_eq!(planes.len(), 3 * BOARD * BOARD, "expected 3×8×8 input");

        // ── Stem ──────────────────────────────────────────────────────────────
        let mut x = self.stem_conv.forward(planes, BOARD, BOARD);
        x = self.stem_bn.forward(&x, BOARD, BOARD);
        relu(&mut x);

        // ── Residual body ─────────────────────────────────────────────────────
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // ── Policy head ───────────────────────────────────────────────────────
        let mut p = self.pol_conv.forward(&x, BOARD, BOARD);
        p = self.pol_bn.forward(&p, BOARD, BOARD);
        relu(&mut p);
        // Flatten: (2, 8, 8) → 128
        let mut p = self.pol_fc.forward(&p);
        softmax(&mut p);

        let mut policy = [0.0f32; 64];
        policy.copy_from_slice(&p);

        // ── Value head ────────────────────────────────────────────────────────
        let mut v = self.val_conv.forward(&x, BOARD, BOARD);
        v = self.val_bn.forward(&v, BOARD, BOARD);
        relu(&mut v);
        // Flatten: (1, 8, 8) → 64
        let mut v = self.val_fc1.forward(&v);
        relu(&mut v);
        let mut v = self.val_fc2.forward(&v);
        tanh_inplace(&mut v);

        let value = v[0];

        (policy, value)
    }

    /// Construct an `AlphaZeroNet` with all-zeros weights — useful for testing
    /// that the network topology is valid without needing real weight files.
    ///
    /// Available only when the `test-utils` feature is enabled.
    #[cfg(feature = "test-utils")]
    pub fn zeros() -> Self {
        fn conv(in_ch: usize, out_ch: usize, ksize: usize, pad: usize) -> Conv2d {
            let w = vec![0.0f32; out_ch * in_ch * ksize * ksize];
            let b = vec![0.0f32; out_ch];
            Conv2d::new(w, Some(b), in_ch, out_ch, ksize, pad)
        }

        fn bn(ch: usize) -> BatchNorm2d {
            BatchNorm2d::new(
                vec![1.0f32; ch],
                vec![0.0f32; ch],
                vec![0.0f32; ch],
                vec![1.0f32; ch],
                BN_EPS,
            )
        }

        fn linear(in_f: usize, out_f: usize) -> Linear {
            Linear::new(vec![0.0f32; out_f * in_f], vec![0.0f32; out_f], in_f, out_f)
        }

        fn res_block() -> ResBlock {
            ResBlock {
                conv1: conv(32, 32, 3, 1),
                bn1: bn(32),
                conv2: conv(32, 32, 3, 1),
                bn2: bn(32),
            }
        }

        Self {
            stem_conv: conv(3, 32, 3, 1),
            stem_bn: bn(32),
            blocks: (0..4).map(|_| res_block()).collect(),
            pol_conv: conv(32, 2, 1, 0),
            pol_bn: bn(2),
            pol_fc: linear(2 * BOARD * BOARD, 64),
            val_conv: conv(32, 1, 1, 0),
            val_bn: bn(1),
            val_fc1: linear(BOARD * BOARD, 64),
            val_fc2: linear(64, 1),
        }
    }
}
