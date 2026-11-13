//! Exact endgame solver: alpha-beta to the end of the game on the final disc
//! difference. Self-play hands positions with few empty squares to it instead
//! of the network, which saves evaluations and gives perfect play (and value
//! targets) for the last moves.

use reversi_minimax::board::Board;

/// Below this many empties the move order is not worth its cost.
const ORDERING_EMPTIES: u32 = 7;

/// Best move for the side to move and the final disc difference (own minus
/// opponent's discs, empties not counted) after perfect play by both sides.
/// The move is `None` when the side to move has to pass or the game is over.
pub fn solve(board: Board, is_black: bool) -> (Option<u64>, i32) {
    let (me, opp) = if is_black { (board.black, board.white) } else { (board.white, board.black) };
    let moves = legal(me, opp);
    if moves == 0 {
        return (None, negamax(me, opp, -65, 65, false));
    }
    let mut best = (None, -65);
    let mut alpha = -65;
    for m in ordered(me, opp, moves) {
        let (next_me, next_opp) = play(me, opp, m);
        let score = -negamax(next_opp, next_me, -65, -alpha, false);
        if score > best.1 {
            best = (Some(m), score);
            alpha = alpha.max(score);
        }
    }
    best
}

fn negamax(me: u64, opp: u64, mut alpha: i32, beta: i32, passed: bool) -> i32 {
    let moves = legal(me, opp);
    if moves == 0 {
        if passed {
            return me.count_ones() as i32 - opp.count_ones() as i32;
        }
        return -negamax(opp, me, -beta, -alpha, true);
    }
    let empties = (!(me | opp)).count_ones();
    if empties >= ORDERING_EMPTIES {
        for m in ordered(me, opp, moves) {
            let (next_me, next_opp) = play(me, opp, m);
            let score = -negamax(next_opp, next_me, -beta, -alpha, false);
            if score >= beta {
                return score;
            }
            alpha = alpha.max(score);
        }
        return alpha;
    }
    let mut best = -65;
    let mut rest = moves;
    while rest != 0 {
        let m = rest.isolate_lowest_one();
        rest &= rest - 1;
        let (next_me, next_opp) = play(me, opp, m);
        let score = -negamax(next_opp, next_me, -beta, -alpha, false);
        if score > best {
            best = score;
            if score >= beta {
                return score;
            }
            alpha = alpha.max(score);
        }
    }
    best
}

/// Moves sorted fastest-first: the fewest replies for the opponent, corners
/// first on ties. Cuts off far more than a fixed order.
fn ordered(me: u64, opp: u64, mut moves: u64) -> impl Iterator<Item = u64> {
    const CORNERS: u64 = 0x8100_0000_0000_0081;
    let mut list = [(0u32, 0u64); 32];
    let mut n = 0;
    while moves != 0 {
        let m = moves.isolate_lowest_one();
        moves &= moves - 1;
        let (next_me, next_opp) = play(me, opp, m);
        let replies = legal(next_opp, next_me).count_ones() * 2 + u32::from(m & CORNERS == 0);
        list[n] = (replies, m);
        n += 1;
    }
    list[..n].sort_unstable_by_key(|&(key, _)| key);
    list.into_iter().take(n).map(|(_, m)| m)
}

fn legal(me: u64, opp: u64) -> u64 {
    Board::new(me, opp).legal_moves(true)
}

fn play(me: u64, opp: u64, m: u64) -> (u64, u64) {
    let next = Board::new(me, opp).apply_move(true, m);
    (next.black, next.white)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plain minimax, no pruning.
    fn brute(me: u64, opp: u64, passed: bool) -> i32 {
        let moves = legal(me, opp);
        if moves == 0 {
            if passed {
                return me.count_ones() as i32 - opp.count_ones() as i32;
            }
            return -brute(opp, me, true);
        }
        let mut best = -65;
        let mut rest = moves;
        while rest != 0 {
            let m = rest.isolate_lowest_one();
            rest &= rest - 1;
            let (a, b) = play(me, opp, m);
            best = best.max(-brute(b, a, false));
        }
        best
    }

    #[test]
    fn matches_brute_force() {
        let mut seed = 7u64;
        let mut checked = 0;
        for _ in 0..60 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let (board, is_black) = crate::game::random_opening(52, seed);
            let (me, opp) =
                if is_black { (board.black, board.white) } else { (board.white, board.black) };
            if (!(me | opp)).count_ones() < 8 {
                continue; // the random game ended early
            }
            checked += 1;
            let (best, score) = solve(board, is_black);
            assert_eq!(score, brute(me, opp, false));
            if let Some(m) = best {
                let (a, b) = play(me, opp, m);
                assert_eq!(-brute(b, a, false), score, "the move reaches the score");
            }
        }
        assert!(checked >= 30, "only {checked} endgames checked");
    }
}
