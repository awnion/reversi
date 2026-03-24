use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use reversi_minimax::board::Board;

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
    /// Game outcome from this player's perspective: +1 win, -1 loss, 0 draw.
    pub outcome: f32,
}

/// Full record of one self-play game.
pub struct GameRecord {
    pub positions: Vec<PositionRecord>,
    /// Final disc counts.
    pub black_discs: u32,
    pub white_discs: u32,
}

// ── play_game ──────────────────────────────────────────────────────────────

/// Play one game between two different evaluators.
/// Returns +1 if black wins, -1 if white wins, 0 if draw.
pub fn play_match(eval_black: &dyn EvalFn, eval_white: &dyn EvalFn, simulations: u32) -> f32 {
    let mut board = Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36));
    let mut is_black = true;
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
    const SELF_PLAY_TEMPERATURE_MOVES: usize = 12;
    const ROOT_NOISE_EPSILON: f32 = 0.25;

    // Standard Reversi start: black on d5(35) and e4(28), white on d4(27) and e5(36).
    let mut board = Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36));
    let mut is_black = true;
    let mut pass_count = 0;
    let mut ply = 0usize;
    let mut rng = SimpleRng::new();

    // Collect (board, is_black, legal, policy) before we know the outcome.
    let mut raw: Vec<(Board, bool, u64, [f32; 64])> = Vec::new();

    loop {
        let legal = board.legal_moves(is_black);

        if legal == 0 {
            pass_count += 1;
            if pass_count >= 2 {
                // Both sides have no moves — game over.
                break;
            }
            // Forced pass.
            is_black = !is_black;
            continue;
        }
        pass_count = 0;

        // Run MCTS.
        let mut mcts = MctsSearch::new(board, is_black);
        if simulations > 0 {
            mcts.simulate(eval);
            add_root_noise(&mut mcts, legal, &mut rng, ROOT_NOISE_EPSILON);
            for _ in 1..simulations {
                mcts.simulate(eval);
            }
        }

        let policy = mcts.policy_target();
        raw.push((board, is_black, legal, policy));

        // Sample from visit counts in the opening to keep self-play diverse.
        let selected = if ply < SELF_PLAY_TEMPERATURE_MOVES {
            sample_move_from_policy(policy, legal, &mut rng).or_else(|| mcts.best_move())
        } else {
            mcts.best_move()
        };
        if let Some(m) = selected {
            board = board.apply_move(is_black, m);
        }

        is_black = !is_black;
        ply += 1;
    }

    // Determine final outcome.
    let black_discs = board.black.count_ones();
    let white_discs = board.white.count_ones();
    let black_outcome: f32 = match black_discs.cmp(&white_discs) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => -1.0,
        std::cmp::Ordering::Equal => 0.0,
    };

    let positions = raw
        .into_iter()
        .map(|(b, ib, legal, policy)| {
            let outcome = if ib { black_outcome } else { -black_outcome };
            PositionRecord { board: b, is_black: ib, legal, mcts_policy: policy, outcome }
        })
        .collect();

    GameRecord { positions, black_discs, white_discs }
}

fn add_root_noise(mcts: &mut MctsSearch, legal: u64, rng: &mut SimpleRng, epsilon: f32) {
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

fn sample_move_from_policy(policy: [f32; 64], legal: u64, rng: &mut SimpleRng) -> Option<u64> {
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

struct SimpleRng(u64);

impl SimpleRng {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);
        SimpleRng(nanos ^ 0xA0761D6478BD642F)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f32(&mut self) -> f32 {
        let v = self.next_u64() >> 40;
        (v as f32) / ((1u32 << 24) as f32)
    }
}
