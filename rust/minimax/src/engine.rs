use crate::board::Board;
use crate::eval::evaluate;

const MIN_SCORE: i32 = -100000;
const MAX_SCORE: i32 = 100000;

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
