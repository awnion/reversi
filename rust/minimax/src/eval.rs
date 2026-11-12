use crate::board::Board;

/// Score of a finished game: beyond any heuristic score, plus the disc margin
/// so that a bigger win scores higher.
pub const WIN: i32 = 10000;

/// Heuristic value of `board` for the side `is_black`: mobility, corners and
/// discs. A finished game scores `±(WIN + margin)`, or 0 for a draw.
pub fn evaluate(board: &Board, is_black: bool) -> i32 {
    let player = board.player_board(is_black);
    let opponent = board.opponent_board(is_black);

    let p_count = player.count_ones() as i32;
    let o_count = opponent.count_ones() as i32;

    let p_moves = board.legal_moves(is_black).count_ones() as i32;
    let o_moves = board.legal_moves(!is_black).count_ones() as i32;

    // Neither side can move (the board may still have empty squares).
    if p_moves == 0 && o_moves == 0 {
        let margin = p_count - o_count;
        return match margin.signum() {
            1 => WIN + margin,
            -1 => -WIN + margin,
            _ => 0,
        };
    }

    let mobility = (p_moves - o_moves) * 10;

    // Corners heuristic
    let corners = 0x8100000000000081u64;
    let p_corners = (player & corners).count_ones() as i32;
    let o_corners = (opponent & corners).count_ones() as i32;
    let corner_score = (p_corners - o_corners) * 100;

    mobility + corner_score + (p_count - o_count)
}
