use crate::board::Board;

pub fn evaluate(board: &Board, is_black: bool) -> i32 {
    let player = board.player_board(is_black);
    let opponent = board.opponent_board(is_black);

    let p_count = player.count_ones() as i32;
    let o_count = opponent.count_ones() as i32;

    if (player | opponent) == 0xFFFFFFFFFFFFFFFF {
        return if p_count > o_count {
            10000
        } else if p_count < o_count {
            -10000
        } else {
            0
        };
    }

    let p_moves = board.legal_moves(is_black).count_ones() as i32;
    let o_moves = board.legal_moves(!is_black).count_ones() as i32;

    let mobility = (p_moves - o_moves) * 10;

    // Corners heuristic
    let corners = 0x8100000000000081u64;
    let p_corners = (player & corners).count_ones() as i32;
    let o_corners = (opponent & corners).count_ones() as i32;
    let corner_score = (p_corners - o_corners) * 100;

    mobility + corner_score + (p_count - o_count)
}
