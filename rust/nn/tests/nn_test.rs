use reversi_nn::AlphaZeroNet;
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
    let planes = vec![0.0f32; 3 * 8 * 8];
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

/// Build a net from a real binary blob (round-trip through WeightStore::write),
/// run a forward pass, and verify invariants.
#[test]
fn net_forward_random_weights() {
    use std::collections::HashMap;

    // ---------- tiny LCG pseudo-random number generator ----------
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to [-0.1, 0.1] — small values keep activations stable
            ((self.0 >> 33) as f32 / u32::MAX as f32) * 0.2 - 0.1
        }

        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }
    let mut rng = Lcg(42);

    // Helper to produce a non-negative variance vector (required for BatchNorm).
    let pos_var = |rng: &mut Lcg, n: usize| -> Vec<f32> {
        rng.vec(n).iter().map(|v| v.abs() + 0.1).collect()
    };

    const B: usize = 8;
    const CH: usize = 32;

    // ── Assemble all tensors for one network ──────────────────────────────
    let mut tensor_map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    macro_rules! t {
        ($name:expr, $shape:expr, $data:expr) => {
            tensor_map.insert($name.to_string(), ($shape.to_vec(), $data));
        };
    }

    // Stem
    t!("stem.conv.weight", [CH, 3, 3, 3], rng.vec(CH * 3 * 3 * 3));
    t!("stem.conv.bias", [CH], rng.vec(CH));
    t!("stem.bn.gamma", [CH], rng.vec(CH));
    t!("stem.bn.beta", [CH], rng.vec(CH));
    t!("stem.bn.mean", [CH], rng.vec(CH));
    t!("stem.bn.var", [CH], pos_var(&mut rng, CH));

    // Residual blocks
    for i in 0..4 {
        let pfx = format!("block{i}");
        t!(format!("{pfx}.conv1.weight"), [CH, CH, 3, 3], rng.vec(CH * CH * 9));
        t!(format!("{pfx}.conv1.bias"), [CH], rng.vec(CH));
        t!(format!("{pfx}.bn1.gamma"), [CH], rng.vec(CH));
        t!(format!("{pfx}.bn1.beta"), [CH], rng.vec(CH));
        t!(format!("{pfx}.bn1.mean"), [CH], rng.vec(CH));
        t!(format!("{pfx}.bn1.var"), [CH], pos_var(&mut rng, CH));
        t!(format!("{pfx}.conv2.weight"), [CH, CH, 3, 3], rng.vec(CH * CH * 9));
        t!(format!("{pfx}.conv2.bias"), [CH], rng.vec(CH));
        t!(format!("{pfx}.bn2.gamma"), [CH], rng.vec(CH));
        t!(format!("{pfx}.bn2.beta"), [CH], rng.vec(CH));
        t!(format!("{pfx}.bn2.mean"), [CH], rng.vec(CH));
        t!(format!("{pfx}.bn2.var"), [CH], pos_var(&mut rng, CH));
    }

    // Policy head
    t!("policy.conv.weight", [2, CH, 1, 1], rng.vec(2 * CH));
    t!("policy.conv.bias", [2], rng.vec(2));
    t!("policy.bn.gamma", [2], rng.vec(2));
    t!("policy.bn.beta", [2], rng.vec(2));
    t!("policy.bn.mean", [2], rng.vec(2));
    t!("policy.bn.var", [2], pos_var(&mut rng, 2));
    t!("policy.fc.weight", [64, 2 * B * B], rng.vec(64 * 2 * B * B));
    t!("policy.fc.bias", [64], rng.vec(64));

    // Value head
    t!("value.conv.weight", [1, CH, 1, 1], rng.vec(CH));
    t!("value.conv.bias", [1], rng.vec(1));
    t!("value.bn.gamma", [1], rng.vec(1));
    t!("value.bn.beta", [1], rng.vec(1));
    t!("value.bn.mean", [1], rng.vec(1));
    t!("value.bn.var", [1], pos_var(&mut rng, 1));
    t!("value.fc1.weight", [64, B * B], rng.vec(64 * B * B));
    t!("value.fc1.bias", [64], rng.vec(64));
    t!("value.fc2.weight", [1, 64], rng.vec(64));
    t!("value.fc2.bias", [1], rng.vec(1));

    // ── Serialise to binary blob ──────────────────────────────────────────
    let entries: Vec<(&str, &[usize], &[f32])> =
        tensor_map.iter().map(|(k, (s, d))| (k.as_str(), s.as_slice(), d.as_slice())).collect();

    let bytes = WeightStore::write(&entries);

    // ── Load and run forward pass ─────────────────────────────────────────
    let net = AlphaZeroNet::load(&bytes).expect("AlphaZeroNet::load failed");

    let mut planes = vec![0.0f32; 3 * 8 * 8];
    // Set a few bits to make the input non-trivial
    planes[0] = 1.0;
    planes[9] = 1.0;
    planes[64] = 1.0;
    planes[128] = 1.0;

    let (policy, value) = net.forward(&planes);

    // Invariant 1: policy sums to ~1
    let sum: f32 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "policy sum = {sum}, expected ~1.0");

    // Invariant 2: all policy values are non-negative
    for (i, &p) in policy.iter().enumerate() {
        assert!(p >= 0.0, "policy[{i}] = {p} is negative");
    }

    // Invariant 3: value is in [-1, 1]
    assert!((-1.0..=1.0).contains(&value), "value = {value} is outside [-1, 1]");
}
