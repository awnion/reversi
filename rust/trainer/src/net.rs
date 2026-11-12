//! The AlphaZero network on libtorch, for batched self-play inference.
//!
//! Weights come from the `.bin` export (`python/train/weights.py`, canonical
//! MLX layout). Loading happens in two steps so that only one thread ever
//! touches CUDA:
//!
//! 1. [`FoldedWeights::from_bin`] (any thread, CPU only): parse, check the
//!    shapes, fold every BatchNorm into the preceding convolution (eval mode,
//!    eps 1e-5, in f32) and reorder the `policy_fc` columns for a channel-first
//!    flatten;
//! 2. [`TorchNet::new`] (the GPU thread): upload, cast to half precision.
//!
//! The forward pass uses functional `conv2d`/`relu`/`linear`; the softmax over
//! the 64 logits and the value `tanh` are computed in f32, so the outputs are
//! move probabilities and values exactly like `TorchModel.infer` returns.

use reversi_nn::WeightStore;
use tch::Device;
use tch::Kind;
use tch::Tensor;

const CHANNELS: usize = 32;
const BLOCKS: usize = 4;
const IN_PLANES: usize = 4;
const POLICY_CHANNELS: usize = 2;
const SQUARES: usize = 64;
const VALUE_HIDDEN: usize = 64;
const BN_EPS: f32 = 1e-5;

/// A convolution with its BatchNorm folded in. `weight` is in the canonical
/// `(out, kH, kW, in)` order.
#[derive(Clone)]
pub struct FoldedConv {
    out_ch: usize,
    in_ch: usize,
    ksize: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// A fully connected layer, `weight` as `(out, in)`.
#[derive(Clone)]
pub struct Dense {
    out_f: usize,
    in_f: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// The network with BatchNorm folded, still in host memory.
#[derive(Clone)]
pub struct FoldedWeights {
    stem: FoldedConv,
    blocks: Vec<(FoldedConv, FoldedConv)>,
    policy_conv: FoldedConv,
    /// Columns in channel-first `(C, H, W)` flatten order.
    policy_fc: Dense,
    value_conv: FoldedConv,
    value_fc1: Dense,
    value_fc2: Dense,
}

fn tensor<'a>(store: &'a WeightStore, name: &str, shape: &[usize]) -> Result<&'a [f32], String> {
    let (data, actual) = store.get(name).ok_or_else(|| format!("missing weight: '{name}'"))?;
    if actual != shape {
        return Err(format!("tensor '{name}' has shape {actual:?}, expected {shape:?}"));
    }
    Ok(data)
}

impl FoldedConv {
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
        let per_out = ksize * ksize * in_ch;
        let weight = w.iter().enumerate().map(|(i, &x)| x * scale[i / per_out]).collect();
        let bias = (0..out_ch).map(|o| (b[o] - mean[o]) * scale[o] + beta[o]).collect();
        Ok(FoldedConv { out_ch, in_ch, ksize, weight, bias })
    }
}

impl Dense {
    fn load(store: &WeightStore, name: &str, in_f: usize, out_f: usize) -> Result<Self, String> {
        let weight = tensor(store, &format!("{name}.weight"), &[out_f, in_f])?.to_vec();
        let bias = tensor(store, &format!("{name}.bias"), &[out_f])?.to_vec();
        Ok(Dense { out_f, in_f, weight, bias })
    }
}

/// `policy_fc` takes the policy activations flattened as `(H, W, C)` (the
/// canonical layout); reorder its columns for a `(C, H, W)` flatten, i.e.
/// `new[o, c * 64 + s] = old[o, s * C + c]`.
pub fn hwc_to_chw_columns(weight: &[f32], out_f: usize, channels: usize) -> Vec<f32> {
    let in_f = channels * SQUARES;
    let mut out = vec![0.0; weight.len()];
    for o in 0..out_f {
        for s in 0..SQUARES {
            for c in 0..channels {
                out[o * in_f + c * SQUARES + s] = weight[o * in_f + s * channels + c];
            }
        }
    }
    out
}

impl FoldedWeights {
    /// Parse a `.bin` export. Tensors the network does not use (the training
    /// only `margin_fc`) are ignored.
    pub fn from_bin(bytes: &[u8]) -> Result<Self, String> {
        let store = WeightStore::load(bytes)?;
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let conv = |n: u8| {
                FoldedConv::load(
                    &store,
                    &format!("blocks.{i}.conv{n}"),
                    &format!("blocks.{i}.bn{n}"),
                    CHANNELS,
                    CHANNELS,
                    3,
                )
            };
            blocks.push((conv(1)?, conv(2)?));
        }
        let mut policy_fc = Dense::load(&store, "policy_fc", SQUARES * POLICY_CHANNELS, SQUARES)?;
        policy_fc.weight = hwc_to_chw_columns(&policy_fc.weight, SQUARES, POLICY_CHANNELS);
        Ok(FoldedWeights {
            stem: FoldedConv::load(
                &store,
                "stem.layers.0",
                "stem.layers.1",
                IN_PLANES,
                CHANNELS,
                3,
            )?,
            blocks,
            policy_conv: FoldedConv::load(
                &store,
                "policy_conv",
                "policy_bn",
                CHANNELS,
                POLICY_CHANNELS,
                1,
            )?,
            policy_fc,
            value_conv: FoldedConv::load(&store, "value_conv", "value_bn", CHANNELS, 1, 1)?,
            value_fc1: Dense::load(&store, "value_fc1", SQUARES, VALUE_HIDDEN)?,
            value_fc2: Dense::load(&store, "value_fc2", VALUE_HIDDEN, 1)?,
        })
    }
}

/// How the network runs on its device.
#[derive(Clone, Copy, Debug)]
pub struct NetOptions {
    pub device: Device,
    /// Half precision weights and activations (CUDA only).
    pub half: bool,
    /// Channels-last (NHWC) storage for activations and conv weights, which
    /// lets cuDNN use its tensor-core kernels.
    pub channels_last: bool,
    /// Allow TF32 in f32 convolutions (faster, about 1e-3 relative error).
    pub tf32: bool,
}

impl NetOptions {
    fn kind(&self) -> Kind {
        if self.half { Kind::Half } else { Kind::Float }
    }
}

struct Conv {
    weight: Tensor,
    bias: Tensor,
    padding: i64,
}

impl Conv {
    fn upload(conv: &FoldedConv, options: &NetOptions) -> Self {
        let (o, i, k) = (conv.out_ch as i64, conv.in_ch as i64, conv.ksize as i64);
        // (out, kH, kW, in) data viewed as (out, in, kH, kW): this already is
        // the channels-last memory layout of the OIHW weight.
        let weight = Tensor::from_slice(&conv.weight)
            .view([o, k, k, i])
            .permute([0, 3, 1, 2])
            .to_device(options.device)
            .to_kind(options.kind());
        let weight = if options.channels_last { weight } else { weight.contiguous() };
        let bias = Tensor::from_slice(&conv.bias).to_device(options.device).to_kind(options.kind());
        Conv { weight, bias, padding: k / 2 }
    }

    fn forward(&self, x: &Tensor, tf32: bool) -> Tensor {
        // `aten::_convolution`: like `conv2d`, but with an explicit TF32 switch
        // (cuDNN would otherwise use TF32 for f32 convolutions on Ampere).
        x.internal_convolution(
            &self.weight,
            Some(&self.bias),
            [1, 1],
            [self.padding, self.padding],
            [1, 1],
            false,
            [0, 0],
            1,
            true,
            false,
            true,
            tf32,
        )
    }
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn upload(dense: &Dense, options: &NetOptions) -> Self {
        let weight = Tensor::from_slice(&dense.weight)
            .view([dense.out_f as i64, dense.in_f as i64])
            .to_device(options.device)
            .to_kind(options.kind());
        let bias =
            Tensor::from_slice(&dense.bias).to_device(options.device).to_kind(options.kind());
        Linear { weight, bias }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        x.linear(&self.weight, Some(&self.bias))
    }
}

/// The network on its device. Create and use it on one thread only.
pub struct TorchNet {
    options: NetOptions,
    stem: Conv,
    blocks: Vec<(Conv, Conv)>,
    policy_conv: Conv,
    policy_fc: Linear,
    value_conv: Conv,
    value_fc1: Linear,
    value_fc2: Linear,
    /// `0..64` as int64 on the device, for unpacking bitboards.
    shifts: Tensor,
}

impl TorchNet {
    pub fn new(weights: &FoldedWeights, options: NetOptions) -> Self {
        TorchNet {
            options,
            stem: Conv::upload(&weights.stem, &options),
            blocks: weights
                .blocks
                .iter()
                .map(|(a, b)| (Conv::upload(a, &options), Conv::upload(b, &options)))
                .collect(),
            policy_conv: Conv::upload(&weights.policy_conv, &options),
            policy_fc: Linear::upload(&weights.policy_fc, &options),
            value_conv: Conv::upload(&weights.value_conv, &options),
            value_fc1: Linear::upload(&weights.value_fc1, &options),
            value_fc2: Linear::upload(&weights.value_fc2, &options),
            shifts: Tensor::arange(64, (Kind::Int64, options.device)),
        }
    }

    /// Network input `(N, 4, 8, 8)` (in the net's dtype and memory format)
    /// from packed positions: per position `[mine, theirs, legal, occupied]`,
    /// the first three as bitboards (`u64` bit patterns), the last the number
    /// of discs on the board. Same planes as `write_position_planes`.
    pub fn unpack(&self, packed: &[i64]) -> Tensor {
        let n = (packed.len() / 4) as i64;
        let kind = self.options.kind();
        let packed = Tensor::from_slice(packed).to_device(self.options.device).view([n, 4]);
        // (N, 1, 3) >> (1, 64, 1) → (N, 64, 3): square-major, i.e. NHWC.
        // Arithmetic shifts are fine: only the lowest bit is kept.
        let bits = packed
            .narrow(1, 0, 3)
            .unsqueeze(1)
            .bitwise_right_shift(&self.shifts.view([1, 64, 1]))
            .bitwise_and(1)
            .to_kind(kind);
        let phase = (packed.narrow(1, 3, 1).to_kind(Kind::Float) / 64.0)
            .to_kind(kind)
            .view([n, 1, 1])
            .expand([n, 64, 1], false);
        let nhwc = Tensor::cat(&[bits, phase], 2).view([n, 8, 8, 4]);
        self.layout(nhwc.permute([0, 3, 1, 2]))
    }

    /// Network input from channel-first f32 planes, `PLANES_LEN` per position.
    pub fn planes(&self, planes: &[f32]) -> Tensor {
        let n = (planes.len() / reversi_mcts::PLANES_LEN) as i64;
        let x = Tensor::from_slice(planes)
            .view([n, 4, 8, 8])
            .to_device(self.options.device)
            .to_kind(self.options.kind());
        if self.options.channels_last {
            // Physically NHWC, logically NCHW.
            x.permute([0, 2, 3, 1]).contiguous().permute([0, 3, 1, 2])
        } else {
            x
        }
    }

    fn layout(&self, x: Tensor) -> Tensor {
        if self.options.channels_last {
            x.permute([0, 2, 3, 1]).contiguous().permute([0, 3, 1, 2])
        } else {
            x.contiguous()
        }
    }

    /// Move probabilities `(N, 64)` and values `(N,)` concatenated into one
    /// f32 tensor `(N, 65)`, still on the device.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let tf32 = self.options.tf32;
        let mut x = self.stem.forward(x, tf32).relu_();
        for (conv1, conv2) in &self.blocks {
            let hidden = conv1.forward(&x, tf32).relu_();
            x = (conv2.forward(&hidden, tf32) + &x).relu_();
        }
        // `flatten` is over the logical (C, H, W) order whatever the memory
        // format; the columns of `policy_fc` were reordered to match.
        let p = self.policy_conv.forward(&x, tf32).relu_().flatten(1, -1);
        let logits = self.policy_fc.forward(&p).to_kind(Kind::Float);
        let policy = logits.softmax(-1, Kind::Float);
        let v = self.value_conv.forward(&x, tf32).relu_().flatten(1, -1);
        let hidden = self.value_fc1.forward(&v).relu_();
        let value = self.value_fc2.forward(&hidden).to_kind(Kind::Float).tanh();
        Tensor::cat(&[policy, value], 1)
    }
}

/// Pack one position for [`TorchNet::unpack`].
pub fn pack(board: reversi_minimax::board::Board, is_black: bool, legal: u64) -> [i64; 4] {
    let (mine, theirs) =
        if is_black { (board.black, board.white) } else { (board.white, board.black) };
    [mine as i64, theirs as i64, legal as i64, i64::from((board.black | board.white).count_ones())]
}

/// Deterministic pseudo-random numbers in `[-1, 1)`.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[cfg(test)]
    pub fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = self.0;
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (a >> 32) | (self.0 & 0xFFFF_FFFF_0000_0000)
    }
}

/// A `.bin` export with random weights and non-trivial BatchNorm stats.
pub fn random_bin(seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, positive: bool, scale: f32| {
        let n = shape.iter().product();
        let data = (0..n)
            .map(|_| {
                let x = rng.next();
                if positive { 0.5 + x.abs() } else { x * scale }
            })
            .collect();
        tensors.push((name, shape, data));
    };
    let mut conv = |conv: &str, bn: &str, i: usize, o: usize, k: usize| {
        add(format!("{conv}.weight"), vec![o, k, k, i], false, 0.3);
        add(format!("{conv}.bias"), vec![o], false, 0.1);
        add(format!("{bn}.weight"), vec![o], true, 1.0);
        add(format!("{bn}.bias"), vec![o], false, 0.1);
        add(format!("{bn}.running_mean"), vec![o], false, 0.2);
        add(format!("{bn}.running_var"), vec![o], true, 1.0);
    };
    conv("stem.layers.0", "stem.layers.1", 4, 32, 3);
    for b in 0..4 {
        for n in 1..=2 {
            conv(&format!("blocks.{b}.conv{n}"), &format!("blocks.{b}.bn{n}"), 32, 32, 3);
        }
    }
    conv("policy_conv", "policy_bn", 32, 2, 1);
    conv("value_conv", "value_bn", 32, 1, 1);
    let mut add = |name: &str, shape: Vec<usize>| {
        let n = shape.iter().product();
        tensors.push((name.to_owned(), shape, (0..n).map(|_| rng.next() * 0.2).collect()));
    };
    add("policy_fc.weight", vec![64, 128]);
    add("policy_fc.bias", vec![64]);
    add("value_fc1.weight", vec![64, 64]);
    add("value_fc1.bias", vec![64]);
    add("value_fc2.weight", vec![1, 64]);
    add("value_fc2.bias", vec![1]);
    add("margin_fc.weight", vec![1, 64]);
    add("margin_fc.bias", vec![1]);
    let refs: Vec<(&str, &[usize], &[f32])> =
        tensors.iter().map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice())).collect();
    WeightStore::write(&refs)
}

#[cfg(test)]
mod tests {
    use reversi_mcts::PLANES_LEN;
    use reversi_mcts::write_position_planes;
    use reversi_minimax::board::Board;

    use super::*;

    /// Random positions: boards, side to move and legal masks.
    fn random_positions(n: usize, seed: u64) -> Vec<(Board, bool, u64)> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| {
                let black = rng.bits() & rng.bits();
                let white = rng.bits() & !black;
                let legal = rng.bits() & !(black | white);
                (Board::new(black, white), rng.next() > 0.0, legal)
            })
            .collect()
    }

    fn cpu(channels_last: bool) -> NetOptions {
        NetOptions { device: Device::Cpu, half: false, channels_last, tf32: false }
    }

    fn to_vec(t: &Tensor) -> Vec<f32> {
        let mut out = vec![0.0; t.numel()];
        t.to_device(Device::Cpu).contiguous().copy_data(&mut out, t.numel());
        out
    }

    #[test]
    fn matches_reference_inference() {
        let _guard = tch::no_grad_guard();
        let bin = random_bin(7);
        let reference = reversi_nn::AlphaZeroNet::load(&bin).unwrap();
        let positions = random_positions(32, 11);
        let mut planes = vec![0.0f32; positions.len() * PLANES_LEN];
        for (&(board, is_black, legal), out) in
            positions.iter().zip(planes.as_chunks_mut::<PLANES_LEN>().0)
        {
            write_position_planes(board, is_black, legal, out);
        }
        for channels_last in [false, true] {
            let net = TorchNet::new(&FoldedWeights::from_bin(&bin).unwrap(), cpu(channels_last));
            let out = to_vec(&net.forward(&net.planes(&planes)));
            let mut max_diff = 0.0f32;
            for (i, row) in out.as_chunks::<65>().0.iter().enumerate() {
                let (policy, value) =
                    reference.forward(&planes[i * PLANES_LEN..(i + 1) * PLANES_LEN]);
                for (a, b) in row[..64].iter().zip(policy) {
                    max_diff = max_diff.max((a - b).abs());
                }
                max_diff = max_diff.max((row[64] - value).abs());
            }
            assert!(max_diff < 1e-4, "channels_last={channels_last}: max diff {max_diff}");
        }
    }

    #[test]
    fn unpack_matches_planes() {
        let positions = random_positions(16, 3);
        let mut planes = vec![0.0f32; positions.len() * PLANES_LEN];
        let mut packed = Vec::new();
        for (&(board, is_black, legal), out) in
            positions.iter().zip(planes.as_chunks_mut::<PLANES_LEN>().0)
        {
            write_position_planes(board, is_black, legal, out);
            packed.extend(pack(board, is_black, legal));
        }
        for channels_last in [false, true] {
            let net = TorchNet::new(
                &FoldedWeights::from_bin(&random_bin(1)).unwrap(),
                cpu(channels_last),
            );
            let unpacked = net.unpack(&packed);
            assert_eq!(unpacked.size(), vec![16, 4, 8, 8]);
            assert_eq!(to_vec(&unpacked), planes, "channels_last={channels_last}");
        }
    }

    /// The reordered `policy_fc` applied to the channel-first flatten equals
    /// the canonical weights applied to the (H, W, C) flatten, for both
    /// memory formats.
    #[test]
    fn policy_column_permutation_matches_flatten() {
        let mut rng = Lcg(5);
        let n = 3i64;
        let canonical: Vec<f32> = (0..64 * 128).map(|_| rng.next()).collect();
        let activations: Vec<f32> = (0..n * 128).map(|_| rng.next()).collect();
        let w = Tensor::from_slice(&canonical).view([64, 128]);
        let w_chw = Tensor::from_slice(&hwc_to_chw_columns(&canonical, 64, 2)).view([64, 128]);
        let p = Tensor::from_slice(&activations).view([n, 2, 8, 8]);
        let p_last = p.permute([0, 2, 3, 1]).contiguous().permute([0, 3, 1, 2]);
        let want = to_vec(&p.permute([0, 2, 3, 1]).flatten(1, -1).matmul(&w.tr()));
        for p in [&p, &p_last] {
            let got = to_vec(&p.flatten(1, -1).matmul(&w_chw.tr()));
            let max_diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_diff < 1e-5, "max diff {max_diff}");
        }
    }
}
