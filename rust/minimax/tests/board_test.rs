#[cfg(test)]
mod tests {
    use reversi_minimax::board::Board;

    #[test]
    fn test_initial_board() {
        // Initial setup
        // Black: D5, E4 => row 3 col 4, row 4 col 3 => indices 28, 35
        // White: D4, E5 => row 3 col 3, row 4 col 4 => indices 27, 36
        let black = (1u64 << 28) | (1u64 << 35);
        let white = (1u64 << 27) | (1u64 << 36);
        let board = Board::new(black, white);

        let black_moves = board.legal_moves(true);
        // Black legal moves on start: C4 (26), D3 (19), E6 (44), F5 (37)
        let expected_black_moves = (1u64 << 26) | (1u64 << 19) | (1u64 << 44) | (1u64 << 37);
        assert_eq!(black_moves, expected_black_moves);
    }
}
