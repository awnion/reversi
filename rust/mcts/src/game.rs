use reversi_minimax::board::Board;

use crate::search::EvalFn;
use crate::search::MctsSearch;

// ── GameRecord ─────────────────────────────────────────────────────────────

/// A single position encountered during self-play.
pub struct PositionRecord {
    pub board: Board,
    pub is_black: bool,
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

/// Play one complete game of self-play using MCTS, returning all positions.
///
/// `simulations` — number of MCTS simulations per move.
pub fn play_game(eval: &dyn EvalFn, simulations: u32) -> GameRecord {
    // Standard Reversi start: black on d5(35) and e4(28), white on d4(27) and e5(36).
    let mut board = Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36));
    let mut is_black = true;
    let mut pass_count = 0;

    // Collect (board, is_black, policy) before we know the outcome.
    let mut raw: Vec<(Board, bool, [f32; 64])> = Vec::new();

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
        for _ in 0..simulations {
            mcts.simulate(eval);
        }

        let policy = mcts.policy_target();
        raw.push((board, is_black, policy));

        // Pick the best move and apply it.
        if let Some(m) = mcts.best_move() {
            board = board.apply_move(is_black, m);
        }

        is_black = !is_black;
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
        .map(|(b, ib, policy)| {
            let outcome = if ib { black_outcome } else { -black_outcome };
            PositionRecord { board: b, is_black: ib, mcts_policy: policy, outcome }
        })
        .collect();

    GameRecord { positions, black_discs, white_discs }
}
