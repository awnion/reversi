use reversi_mcts::MctsSearch;
use reversi_mcts::StaticEval;
use reversi_mcts::play_game;
use reversi_minimax::board::Board;

fn starting_board() -> Board {
    Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36))
}

#[test]
fn policy_target_sums_to_one() {
    let board = starting_board();
    let eval = StaticEval;
    let mut mcts = MctsSearch::new(board, true);
    for _ in 0..100 {
        mcts.simulate(&eval);
    }
    let policy = mcts.policy_target();
    let sum: f32 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "policy sum = {sum}");
}

#[test]
fn best_move_is_legal() {
    let board = starting_board();
    let eval = StaticEval;
    let mut mcts = MctsSearch::new(board, true);
    for _ in 0..50 {
        mcts.simulate(&eval);
    }
    let m = mcts.best_move().expect("should have a best move");
    let legal = board.legal_moves(true);
    assert_ne!(m & legal, 0, "best move must be a legal move");
}

#[test]
fn game_completes() {
    let eval = StaticEval;
    let record = play_game(&eval, 20);
    assert!(!record.positions.is_empty(), "game must have positions");
    assert_eq!(record.black_discs + record.white_discs, 64, "all squares filled at end of game");
}

#[test]
fn game_policies_sum_to_one() {
    let eval = StaticEval;
    let record = play_game(&eval, 20);
    for (i, pos) in record.positions.iter().enumerate() {
        let sum: f32 = pos.mcts_policy.iter().sum();
        if pos.policy_weight > 0.0 {
            assert!((sum - 1.0).abs() < 1e-5, "position {i} policy sum = {sum}");
        } else {
            let valid = sum.abs() < 1e-5 || (sum - 1.0).abs() < 1e-5;
            assert!(valid, "value-only position {i} policy sum = {sum}");
        }
    }
}

#[test]
fn outcomes_are_consistent() {
    let eval = StaticEval;
    let record = play_game(&eval, 20);
    let final_outcome = match record.black_discs.cmp(&record.white_discs) {
        std::cmp::Ordering::Greater => 1.0_f32,
        std::cmp::Ordering::Less => -1.0_f32,
        std::cmp::Ordering::Equal => 0.0_f32,
    };
    // The first position is always black's turn; its outcome should match final_outcome.
    if let Some(first) = record.positions.first() {
        assert!(first.is_black);
        assert_eq!(first.outcome, final_outcome);
    }
}
