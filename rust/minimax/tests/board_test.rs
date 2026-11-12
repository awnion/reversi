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

// ── search deadlines ──────────────────────────────────────────────────────

mod deadlines {
    use reversi_minimax::board::Board;
    use reversi_minimax::engine::Deadline;
    use reversi_minimax::engine::MAX_SCORE;
    use reversi_minimax::engine::MIN_SCORE;
    use reversi_minimax::engine::search;
    use reversi_minimax::engine::search_until;

    fn start() -> Board {
        Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36))
    }

    #[test]
    fn search_without_deadline_matches_plain_search() {
        let plain = search(start(), true, 6, MIN_SCORE, MAX_SCORE);
        let timed = search_until(start(), true, 6, MIN_SCORE, MAX_SCORE, &mut Deadline::never());
        assert_eq!(timed, Some(plain));
    }

    #[test]
    fn search_gives_up_once_the_deadline_has_passed() {
        // A clock that is already past the deadline: the search must stop at
        // the first clock check instead of finishing a deep iteration.
        let mut deadline = Deadline::at(|| 1.0, 0.0);
        let result = search_until(start(), true, 12, MIN_SCORE, MAX_SCORE, &mut deadline);
        assert_eq!(result, None);
        assert!(deadline.expired());
    }
}

// ── root search (split between threads with the `threads` feature) ────────

mod root_search {
    use reversi_minimax::board::Board;
    use reversi_minimax::engine::Deadline;
    use reversi_minimax::engine::MAX_SCORE;
    use reversi_minimax::engine::MIN_SCORE;
    use reversi_minimax::engine::search;
    use reversi_minimax::engine::search_root_until;

    /// A middle-game position: a few moves from the start.
    fn positions() -> Vec<(Board, bool)> {
        let mut board = Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36));
        let mut is_black = true;
        let mut out = vec![(board, is_black)];
        for _ in 0..8 {
            let legal = board.legal_moves(is_black);
            // Deterministic but not always the lowest move.
            let m = 1u64 << (63 - legal.leading_zeros());
            board = board.apply_move(is_black, m);
            is_black = !is_black;
            out.push((board, is_black));
        }
        out
    }

    #[test]
    fn root_search_finds_the_same_score_as_plain_search() {
        for (board, is_black) in positions() {
            for depth in 1..=5 {
                let (_, plain) = search(board, is_black, depth, MIN_SCORE, MAX_SCORE);
                for first in
                    [None, Some(1u64 << (63 - board.legal_moves(is_black).leading_zeros()))]
                {
                    let (m, score) =
                        search_root_until(board, is_black, depth, first, &mut Deadline::never())
                            .unwrap();
                    assert_eq!(score, plain, "depth {depth}");
                    let m = m.unwrap();
                    assert_ne!(board.legal_moves(is_black) & m, 0);
                    // The chosen move really scores that much.
                    let next = board.apply_move(is_black, m);
                    let (_, reply) = search(next, !is_black, depth - 1, MIN_SCORE, MAX_SCORE);
                    assert_eq!(-reply, plain);
                }
            }
        }
    }

    #[test]
    fn root_search_gives_up_once_the_deadline_has_passed() {
        let (board, is_black) = positions()[4];
        let mut deadline = Deadline::at(|| 1.0, 0.0);
        assert_eq!(search_root_until(board, is_black, 12, None, &mut deadline), None);
        assert!(deadline.expired());
    }
}
