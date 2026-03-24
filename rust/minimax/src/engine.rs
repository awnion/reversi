use crate::board::Board;
use crate::eval::evaluate;

/// Evaluates every legal move for `is_black` at the given depth.
/// Returns a list of `(move_bit_mask, score_for_current_player)` pairs.
pub fn evaluate_all_moves(board: Board, is_black: bool, depth: u8) -> Vec<(u64, i32)> {
    let legal = board.legal_moves(is_black);
    let mut results = Vec::new();
    for i in 0..64 {
        let m = 1u64 << i;
        if (legal & m) != 0 {
            let next = board.apply_move(is_black, m);
            // Search from opponent's perspective at depth-1; negate score.
            let (_, opp_score) = search(next, !is_black, depth.saturating_sub(1), -MAX_SCORE, MAX_SCORE);
            results.push((m, -opp_score));
        }
    }
    results
}

pub const MIN_SCORE: i32 = -100000;
pub const MAX_SCORE: i32 = 100000;

pub fn search(board: Board, is_black: bool, depth: u8, mut alpha: i32, beta: i32) -> (Option<u64>, i32) {
    if depth == 0 {
        return (None, evaluate(&board, is_black));
    }

    let legal_moves = board.legal_moves(is_black);
    if legal_moves == 0 {
        let opponent_moves = board.legal_moves(!is_black);
        if opponent_moves == 0 {
            // Game over
            return (None, evaluate(&board, is_black));
        } else {
            // Pass
            let (_, score) = search(board, !is_black, depth - 1, -beta, -alpha);
            return (None, -score);
        }
    }

    let mut best_move = None;
    let mut best_score = MIN_SCORE;

    for i in 0..64 {
        let m = 1u64 << i;
        if (legal_moves & m) != 0 {
            let next_board = board.apply_move(is_black, m);
            let (_, score) = search(next_board, !is_black, depth - 1, -beta, -alpha);
            let score = -score;

            if score > best_score {
                best_score = score;
                best_move = Some(m);
            }
            alpha = alpha.max(best_score);
            if alpha >= beta {
                break;
            }
        }
    }

    (best_move, best_score)
}
