use reversi_nn::AlphaZeroNet;
use reversi_nn::ReferenceNet;
use reversi_nn::WeightStore;

// ── WeightStore roundtrip ─────────────────────────────────────────────────────

#[test]
fn weight_store_roundtrip() {
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let data_b: Vec<f32> = vec![0.5, -0.5];

    let bytes = WeightStore::write(&[
        ("tensor_a", &[2usize, 2], data_a.as_slice()),
        ("tensor_b", &[2usize], data_b.as_slice()),
    ]);

    let store = WeightStore::load(&bytes).expect("load failed");

    let (vals_a, shape_a) = store.get("tensor_a").expect("tensor_a missing");
    assert_eq!(shape_a, &[2, 2]);
    assert_eq!(vals_a, data_a.as_slice());

    let (vals_b, shape_b) = store.get("tensor_b").expect("tensor_b missing");
    assert_eq!(shape_b, &[2]);
    assert_eq!(vals_b, data_b.as_slice());

    assert!(store.get("no_such_tensor").is_none());
}

#[test]
fn weight_store_bad_magic() {
    let mut bytes = WeightStore::write(&[("x", &[1usize], &[1.0f32])]);
    bytes[0] ^= 0xFF; // corrupt the magic
    assert!(WeightStore::load(&bytes).is_err());
}

#[test]
fn weight_store_empty() {
    let bytes = WeightStore::write(&[]);
    let store = WeightStore::load(&bytes).expect("empty store should load");
    assert!(store.get("anything").is_none());
}

// ── Net forward pass with zero weights ───────────────────────────────────────

/// All-zero weights propagate zeros through the convolutions.
/// After the final softmax the policy should be uniform (1/64 each),
/// and after tanh(0) the value should be 0.
#[test]
fn net_forward_zero_weights() {
    let net = AlphaZeroNet::zeros();
    let planes = vec![0.0f32; 4 * 8 * 8];
    let (policy, value) = net.forward(&planes);

    // softmax of all-equal logits → uniform distribution
    let expected = 1.0 / 64.0;
    for (i, &p) in policy.iter().enumerate() {
        assert!((p - expected).abs() < 1e-5, "policy[{i}] = {p}, expected {expected}");
    }

    // sum should be ~1
    let sum: f32 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "policy sum = {sum}");

    // tanh(0) = 0
    assert!(value.abs() < 1e-5, "value = {value}");
    assert!((-1.0..=1.0).contains(&value), "value out of range: {value}");
}

// ── Random weights: optimised net vs reference ───────────────────────────────

/// Tiny LCG pseudo-random number generator.
struct Lcg(u64);

impl Lcg {
    /// Uniform in [-1, 1].
    fn next_f32(&mut self) -> f32 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 4.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

/// A full set of random weights in the MLX export layout, serialised to the
/// binary format.
fn random_mlx_weights(seed: u64) -> Vec<u8> {
    const CH: usize = 32;
    let mut rng = Lcg(seed);
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

    let mut conv = |rng: &mut Lcg, name: &str, out: usize, k: usize, inp: usize| {
        tensors.push((
            format!("{name}.weight"),
            vec![out, k, k, inp],
            rng.vec(out * k * k * inp, 0.1),
        ));
        tensors.push((format!("{name}.bias"), vec![out], rng.vec(out, 0.1)));
    };
    conv(&mut rng, "stem.layers.0", CH, 3, 4);
    for i in 0..4 {
        conv(&mut rng, &format!("blocks.{i}.conv1"), CH, 3, CH);
        conv(&mut rng, &format!("blocks.{i}.conv2"), CH, 3, CH);
    }
    conv(&mut rng, "policy_conv", 2, 1, CH);
    conv(&mut rng, "value_conv", 1, 1, CH);

    let mut bn = |rng: &mut Lcg, name: &str, ch: usize| {
        tensors.push((format!("{name}.weight"), vec![ch], rng.vec(ch, 1.0)));
        tensors.push((format!("{name}.bias"), vec![ch], rng.vec(ch, 0.5)));
        tensors.push((format!("{name}.running_mean"), vec![ch], rng.vec(ch, 0.5)));
        // Variances must be positive.
        let var = rng.vec(ch, 1.0).iter().map(|v| v.abs() + 0.1).collect();
        tensors.push((format!("{name}.running_var"), vec![ch], var));
    };
    bn(&mut rng, "stem.layers.1", CH);
    for i in 0..4 {
        bn(&mut rng, &format!("blocks.{i}.bn1"), CH);
        bn(&mut rng, &format!("blocks.{i}.bn2"), CH);
    }
    bn(&mut rng, "policy_bn", 2);
    bn(&mut rng, "value_bn", 1);

    let mut dense = |rng: &mut Lcg, name: &str, out: usize, inp: usize| {
        tensors.push((format!("{name}.weight"), vec![out, inp], rng.vec(out * inp, 0.1)));
        tensors.push((format!("{name}.bias"), vec![out], rng.vec(out, 0.1)));
    };
    dense(&mut rng, "policy_fc", 64, 128);
    dense(&mut rng, "value_fc1", 64, 64);
    dense(&mut rng, "value_fc2", 1, 64);

    let entries: Vec<(&str, &[usize], &[f32])> =
        tensors.iter().map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice())).collect();
    WeightStore::write(&entries)
}

/// Board-like inputs: sparse 0/1 planes plus a constant phase plane.
fn random_planes(rng: &mut Lcg) -> Vec<f32> {
    let mut planes = vec![0.0f32; 4 * 64];
    for v in &mut planes[..3 * 64] {
        *v = if rng.next_f32() > 0.3 { 1.0 } else { 0.0 };
    }
    let phase = rng.next_f32().abs();
    planes[3 * 64..].fill(phase);
    planes
}

/// The optimised network (folded BatchNorm, channel-last, zero skipping) must
/// compute the same function as the layer-by-layer reference.
#[test]
fn optimised_net_matches_reference() {
    for seed in [1, 42, 2026] {
        let bytes = random_mlx_weights(seed);
        let net = AlphaZeroNet::load(&bytes).expect("AlphaZeroNet::load failed");
        let reference = ReferenceNet::load(&bytes).expect("ReferenceNet::load failed");

        let mut rng = Lcg(seed ^ 0xABCD);
        for _ in 0..20 {
            let planes = random_planes(&mut rng);
            let (policy, value) = net.forward(&planes);
            let (want_policy, want_value) = reference.forward(&planes);

            let sum: f32 = policy.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "policy sum = {sum}");
            assert!((-1.0..=1.0).contains(&value), "value = {value} is outside [-1, 1]");

            for (i, (&got, &want)) in policy.iter().zip(&want_policy).enumerate() {
                assert!((got - want).abs() < 1e-4, "seed {seed}: policy[{i}] = {got}, want {want}");
            }
            assert!(
                (value - want_value).abs() < 1e-4,
                "seed {seed}: value = {value}, want {want_value}"
            );
        }
    }
}

// ── Exported champion files ──────────────────────────────────────────────────

/// The files written by `python/train/export.py` must load: the browser bots
/// embed exactly these bytes.
#[test]
fn exported_champions_load() {
    let champions: [(&str, &[u8]); 2] = [
        ("alphazero", include_bytes!("../../../weights/alphazero/champion.bin")),
        ("zerorev", include_bytes!("../../../weights/zerorev/champion.bin")),
    ];
    for (name, bytes) in champions {
        let net = AlphaZeroNet::load(bytes).unwrap_or_else(|err| panic!("{name}: {err}"));
        let (policy, value) = net.forward(&[0.0; 4 * 8 * 8]);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "{name}: policy sum = {sum}");
        assert!((-1.0..=1.0).contains(&value), "{name}: value = {value}");
    }
}
