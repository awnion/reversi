use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use reversi_minimax::board::Board;

use crate::batch::SelfPlayPool;
use crate::search::EvalFn;
use crate::search::MctsSearch;

// ── GameRecord ─────────────────────────────────────────────────────────────

/// A single position encountered during self-play.
pub struct PositionRecord {
    pub board: Board,
    pub is_black: bool,
    pub legal: u64,
    /// MCTS visit-count distribution (64 squares).
    pub mcts_policy: [f32; 64],
    /// Value target from this player's perspective: the game outcome (+1 win,
    /// -1 loss, 0 draw); in batched self-play mixed with the search value for
    /// searched and solved positions (`VALUE_TARGET_SEARCH_WEIGHT`).
    pub outcome: f32,
    /// Final disc difference from this player's perspective, divided by 64.
    pub margin: f32,
    pub policy_weight: f32,
    pub value_weight: f32,
}

/// Full record of one self-play game.
pub struct GameRecord {
    pub positions: Vec<PositionRecord>,
    /// Final disc counts.
    pub black_discs: u32,
    pub white_discs: u32,
}

// ── play_game ──────────────────────────────────────────────────────────────

/// The standard Reversi starting position (black to move).
fn starting_board() -> Board {
    // Black on d5(35) and e4(28), white on d4(27) and e5(36).
    Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36))
}

/// Play one game between two different evaluators.
/// Returns +1 if black wins, -1 if white wins, 0 if draw.
pub fn play_match(eval_black: &dyn EvalFn, eval_white: &dyn EvalFn, simulations: u32) -> f32 {
    play_match_from(starting_board(), true, eval_black, eval_white, simulations)
}

/// A position reached by `plies` uniformly random legal moves from the start,
/// as `(board, black_to_move)`. The same `seed` always gives the same
/// position, so both colour assignments of a pairing can share an opening.
pub fn random_opening(plies: u32, seed: u64) -> (Board, bool) {
    let mut rng = SimpleRng::from_seed(seed);
    let mut board = starting_board();
    let mut is_black = true;
    for _ in 0..plies {
        let mut legal = board.legal_moves(is_black);
        if legal == 0 {
            is_black = !is_black;
            legal = board.legal_moves(is_black);
            if legal == 0 {
                break;
            }
        }
        // Pick the n-th set bit of `legal`.
        let n = (rng.next_f32() * legal.count_ones() as f32) as u32;
        for _ in 0..n.min(legal.count_ones() - 1) {
            legal &= legal - 1;
        }
        board = board.apply_move(is_black, 1u64 << legal.trailing_zeros());
        is_black = !is_black;
    }
    (board, is_black)
}

/// Play a game from an arbitrary position, deterministically (no noise, the
/// most visited move is always played).
/// Returns +1 if black wins, -1 if white wins, 0 if draw.
pub fn play_match_from(
    mut board: Board,
    mut is_black: bool,
    eval_black: &dyn EvalFn,
    eval_white: &dyn EvalFn,
    simulations: u32,
) -> f32 {
    let mut pass_count = 0;

    loop {
        let legal = board.legal_moves(is_black);
        if legal == 0 {
            pass_count += 1;
            if pass_count >= 2 {
                break;
            }
            is_black = !is_black;
            continue;
        }
        pass_count = 0;

        let eval: &dyn EvalFn = if is_black { eval_black } else { eval_white };
        let mut mcts = MctsSearch::new(board, is_black);
        for _ in 0..simulations {
            mcts.simulate(eval);
        }
        if let Some(m) = mcts.best_move() {
            board = board.apply_move(is_black, m);
        }
        is_black = !is_black;
    }

    let b = board.black.count_ones();
    let w = board.white.count_ones();
    match b.cmp(&w) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => -1.0,
        std::cmp::Ordering::Equal => 0.0,
    }
}

/// Play one complete game of self-play using MCTS, returning all positions.
///
/// `simulations` — number of MCTS simulations per move.
pub fn play_game(eval: &dyn EvalFn, simulations: u32) -> GameRecord {
    let mut pool = SelfPlayPool::new(1, simulations);
    loop {
        let request = pool.collect()[0];
        if let Some(record) = pool.take_finished().pop() {
            return record;
        }
        let (policy, value) = eval.evaluate(request.board, request.is_black, request.legal);
        pool.apply(&policy, &[value]);
    }
}

pub(crate) fn add_root_noise(mcts: &mut MctsSearch, legal: u64, rng: &mut SimpleRng, epsilon: f32) {
    let child_count = legal.count_ones() as usize;
    if child_count == 0 {
        return;
    }

    // Sample positive noise weights and normalise them. This is enough to break
    // deterministic openings and keep the replay buffer from collapsing.
    let mut noise = vec![0.0_f32; child_count];
    let mut sum = 0.0_f32;
    for value in &mut noise {
        let u = f32::EPSILON + rng.next_f32() * (1.0 - f32::EPSILON);
        *value = -u.ln();
        sum += *value;
    }
    if sum <= 0.0 {
        return;
    }
    for value in &mut noise {
        *value /= sum;
    }
    mcts.add_root_noise(&noise, epsilon);
}

pub(crate) fn sample_move_from_policy(
    policy: [f32; 64],
    legal: u64,
    rng: &mut SimpleRng,
) -> Option<u64> {
    let mut moves = Vec::new();
    let mut weights = Vec::new();
    for (i, &weight) in policy.iter().enumerate() {
        if (legal >> i) & 1 == 1 {
            moves.push(1u64 << i);
            weights.push(weight.max(0.0));
        }
    }
    if moves.is_empty() {
        return None;
    }
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return Some(moves[0]);
    }
    let mut pick = rng.next_f32() * total;
    for (move_mask, weight) in moves.iter().zip(weights.iter()) {
        pick -= *weight;
        if pick <= 0.0 {
            return Some(*move_mask);
        }
    }
    moves.last().copied()
}

pub(crate) struct SimpleRng(u64);

impl SimpleRng {
    /// A generator with a fixed stream for the given seed.
    pub(crate) fn from_seed(seed: u64) -> Self {
        // SplitMix64 finaliser, so that nearby seeds give unrelated streams.
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        SimpleRng((z ^ (z >> 31)) | 1)
    }

    pub(crate) fn new() -> Self {
        // The clock alone is too coarse: a pool creates hundreds of games in
        // the same tick, and equal seeds would make them play identical games.
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E3779B97F4A7C15);
        // SplitMix64 finaliser, so that nearby seeds give unrelated streams.
        let mut z = (nanos ^ 0xA0761D6478BD642F).wrapping_add(unique);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        SimpleRng((z ^ (z >> 31)) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        let v = self.next_u64() >> 40;
        (v as f32) / ((1u32 << 24) as f32)
    }
}
