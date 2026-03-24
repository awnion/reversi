/// 2-D convolution — channel-first `(C, H, W)` layout throughout.
///
/// Weight layout: `[out_ch, in_ch, kH, kW]` (PyTorch convention).
pub struct Conv2d {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_ch: usize,
    out_ch: usize,
    ksize: usize,
    pad: usize,
}

impl Conv2d {
    pub fn new(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_ch: usize,
        out_ch: usize,
        ksize: usize,
        pad: usize,
    ) -> Self {
        debug_assert_eq!(
            weight.len(),
            out_ch * in_ch * ksize * ksize,
            "Conv2d weight size mismatch"
        );
        if let Some(ref b) = bias {
            debug_assert_eq!(b.len(), out_ch, "Conv2d bias size mismatch");
        }
        Self { weight, bias, in_ch, out_ch, ksize, pad }
    }

    /// Compute the forward pass.
    ///
    /// `input` must have length `in_ch * h * w`.
    /// Returns a `Vec<f32>` of length `out_ch * h_out * w_out`
    /// where `h_out = h` and `w_out = w` (when `pad = ksize / 2`).
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.in_ch * h * w);

        let h_out = h + 2 * self.pad - (self.ksize - 1);
        let w_out = w + 2 * self.pad - (self.ksize - 1);
        let mut output = vec![0.0f32; self.out_ch * h_out * w_out];

        for oc in 0..self.out_ch {
            let bias_val = self.bias.as_ref().map_or(0.0, |b| b[oc]);
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = bias_val;
                    for ic in 0..self.in_ch {
                        for kh in 0..self.ksize {
                            for kw in 0..self.ksize {
                                let ih = oh + kh;
                                let iw = ow + kw;
                                // `ih` and `iw` are in the padded coordinate space;
                                // subtract padding to get the real input index.
                                let src_h = ih as isize - self.pad as isize;
                                let src_w = iw as isize - self.pad as isize;
                                if src_h >= 0
                                    && src_h < h as isize
                                    && src_w >= 0
                                    && src_w < w as isize
                                {
                                    let in_idx = ic * h * w + src_h as usize * w + src_w as usize;
                                    let w_idx = oc * (self.in_ch * self.ksize * self.ksize)
                                        + ic * (self.ksize * self.ksize)
                                        + kh * self.ksize
                                        + kw;
                                    acc += input[in_idx] * self.weight[w_idx];
                                }
                            }
                        }
                    }
                    output[oc * h_out * w_out + oh * w_out + ow] = acc;
                }
            }
        }

        output
    }
}

/// Batch normalisation — inference mode only (uses running statistics).
///
/// `y = gamma * (x - mean) / sqrt(var + eps) + beta`
pub struct BatchNorm2d {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    mean: Vec<f32>,
    /// Pre-computed `1 / sqrt(var + eps)` per channel.
    inv_std: Vec<f32>,
}

impl BatchNorm2d {
    pub fn new(gamma: Vec<f32>, beta: Vec<f32>, mean: Vec<f32>, var: Vec<f32>, eps: f32) -> Self {
        debug_assert_eq!(gamma.len(), beta.len());
        debug_assert_eq!(gamma.len(), mean.len());
        debug_assert_eq!(gamma.len(), var.len());
        let inv_std = var.iter().map(|&v| 1.0 / (v + eps).sqrt()).collect();
        Self { gamma, beta, mean, inv_std }
    }

    /// `input` has layout `(C, H, W)` with length `C * h * w`.
    /// Returns the same shape.
    pub fn forward(&self, input: &[f32], h: usize, w: usize) -> Vec<f32> {
        let c = self.gamma.len();
        debug_assert_eq!(input.len(), c * h * w);

        let mut output = Vec::with_capacity(input.len());
        for ch in 0..c {
            let g = self.gamma[ch];
            let b = self.beta[ch];
            let m = self.mean[ch];
            let inv = self.inv_std[ch];
            let start = ch * h * w;
            let end = start + h * w;
            for &x in &input[start..end] {
                output.push(g * (x - m) * inv + b);
            }
        }
        output
    }
}

/// Fully-connected layer: `y = W x + b`.
///
/// Weight layout: `[out_features, in_features]` (row-major, PyTorch convention).
pub struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(weight: Vec<f32>, bias: Vec<f32>, in_features: usize, out_features: usize) -> Self {
        debug_assert_eq!(weight.len(), out_features * in_features);
        debug_assert_eq!(bias.len(), out_features);
        Self { weight, bias, in_features, out_features }
    }

    /// `input` has length `in_features`. Returns a `Vec<f32>` of length `out_features`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.in_features);
        let mut output = Vec::with_capacity(self.out_features);
        for o in 0..self.out_features {
            let row = &self.weight[o * self.in_features..(o + 1) * self.in_features];
            let mut acc = self.bias[o];
            for (&w, &x) in row.iter().zip(input.iter()) {
                acc += w * x;
            }
            output.push(acc);
        }
        output
    }
}

// ── Activation functions ─────────────────────────────────────────────────────

/// Rectified Linear Unit applied in-place.
pub fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Hyperbolic tangent applied in-place.
pub fn tanh_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.tanh();
    }
}

/// Stable softmax applied in-place.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}
