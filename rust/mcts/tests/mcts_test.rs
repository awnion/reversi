use reversi_mcts::EvalFn;
use reversi_mcts::GameRecord;
use reversi_mcts::MctsSearch;
use reversi_mcts::PLANES_LEN;
use reversi_mcts::SelfPlayPool;
use reversi_mcts::SharedSearch;
use reversi_mcts::StaticEval;
use reversi_mcts::play_game;
use reversi_mcts::play_match_from;
use reversi_mcts::random_opening;
use reversi_mcts::run_threads;
use reversi_mcts::write_planes;
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

// ── split select/apply API ────────────────────────────────────────────────

#[test]
fn select_and_apply_matches_simulate() {
    let eval = StaticEval;
    let mut whole = MctsSearch::new(starting_board(), true);
    let mut split = MctsSearch::new(starting_board(), true);
    for _ in 0..300 {
        whole.simulate(&eval);
        if let Some(request) = split.select_leaf() {
            let (policy, value) = eval.evaluate(request.board, request.is_black, request.legal);
            split.apply_eval(&request, &policy, value);
        }
    }
    assert_eq!(whole.policy_target(), split.policy_target());
    assert_eq!(whole.best_move(), split.best_move());
}

// ── batched self-play ─────────────────────────────────────────────────────

/// Run `pool` with the static evaluator until `games` games have finished.
fn run_pool(pool: &mut SelfPlayPool, games: usize) -> Vec<GameRecord> {
    let eval = StaticEval;
    let mut records = Vec::new();
    let mut policies = Vec::new();
    let mut values = Vec::new();
    while records.len() < games {
        policies.clear();
        values.clear();
        for request in pool.collect() {
            assert_ne!(request.legal, 0, "requests always have a legal move");
            let (policy, value) = eval.evaluate(request.board, request.is_black, request.legal);
            policies.extend_from_slice(&policy);
            values.push(value);
        }
        records.extend(pool.take_finished());
        pool.apply(&policies, &values);
    }
    records
}

#[test]
fn pool_produces_valid_games() {
    let mut pool = SelfPlayPool::new(16, 20);
    let records = run_pool(&mut pool, 32);
    assert!(records.len() >= 32);

    for record in &records {
        assert!(record.black_discs + record.white_discs <= 64);
        let black_outcome = match record.black_discs.cmp(&record.white_discs) {
            std::cmp::Ordering::Greater => 1.0_f32,
            std::cmp::Ordering::Less => -1.0_f32,
            std::cmp::Ordering::Equal => 0.0_f32,
        };
        // At least one searched move plus the two terminal value samples.
        assert!(record.positions.len() >= 3);
        for pos in &record.positions {
            let expected = if pos.is_black { black_outcome } else { -black_outcome };
            assert_eq!(pos.outcome, expected);
            let black_margin = (record.black_discs as f32 - record.white_discs as f32) / 64.0;
            let margin = if pos.is_black { black_margin } else { -black_margin };
            assert_eq!(pos.margin, margin);
            assert_eq!(pos.margin.signum() * pos.outcome.abs(), pos.outcome);
            if pos.policy_weight > 0.0 {
                let sum: f32 = pos.mcts_policy.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "policy sum = {sum}");
                for (i, &p) in pos.mcts_policy.iter().enumerate() {
                    if (pos.legal >> i) & 1 == 0 {
                        assert_eq!(p, 0.0, "policy mass on illegal square {i}");
                    }
                }
            }
        }
    }
}

#[test]
fn pool_games_are_not_identical() {
    // Games created in the same clock tick must still get different seeds.
    let mut pool = SelfPlayPool::new(16, 20);
    let records = run_pool(&mut pool, 16);
    let trajectory = |r: &GameRecord| -> Vec<(u64, u64)> {
        r.positions.iter().map(|p| (p.board.black, p.board.white)).collect()
    };
    let first = trajectory(&records[0]);
    assert!(
        records.iter().any(|r| trajectory(r) != first),
        "all games in the pool played the same moves",
    );
}

#[test]
fn planes_encode_the_side_to_move() {
    let mut search = MctsSearch::new(starting_board(), true);
    let request = search.select_leaf().expect("the root needs an evaluation");
    let mut planes = [0.0_f32; PLANES_LEN];
    write_planes(&request, &mut planes);

    let ones = |plane: usize| -> Vec<usize> {
        (0..64).filter(|i| planes[plane * 64 + i] == 1.0).collect()
    };
    assert_eq!(ones(0), vec![28, 35], "black to move: own discs first");
    assert_eq!(ones(1), vec![27, 36]);
    assert_eq!(ones(2).len(), 4, "four legal opening moves");
    assert!(planes[3 * 64..].iter().all(|&v| v == 4.0 / 64.0), "phase = occupied / 64");
}

#[test]
fn staggered_pool_ramps_up_and_spreads_finishes() {
    let eval = StaticEval;
    let games = 8;
    let mut pool = SelfPlayPool::staggered(games, 4);
    let mut policies = Vec::new();
    let mut values = Vec::new();
    let mut first_batch = None;
    let mut max_batch = 0;
    // Rounds in which at least one game finished.
    let mut finish_rounds = Vec::new();
    for round in 0..4_000 {
        policies.clear();
        values.clear();
        let requests = pool.collect();
        first_batch.get_or_insert(requests.len());
        max_batch = max_batch.max(requests.len());
        for request in requests {
            let (policy, value) = eval.evaluate(request.board, request.is_black, request.legal);
            policies.extend_from_slice(&policy);
            values.push(value);
        }
        if !pool.take_finished().is_empty() {
            finish_rounds.push(round);
        }
        pool.apply(&policies, &values);
    }
    assert_eq!(first_batch, Some(1), "only the first game starts in round 0");
    assert_eq!(max_batch, games, "all games are running after the ramp-up");
    assert!(finish_rounds.len() > games, "games finish in many separate rounds");
}

// ── matches from random openings ──────────────────────────────────────────

#[test]
fn random_openings_are_reproducible_and_varied() {
    let (board, black_to_move) = random_opening(4, 7);
    assert_eq!((board.black, board.white, black_to_move), {
        let (b, t) = random_opening(4, 7);
        (b.black, b.white, t)
    });
    assert_eq!((board.black | board.white).count_ones(), 8, "four moves were played");
    assert!(black_to_move, "black is to move again after four plies");

    let distinct: std::collections::HashSet<(u64, u64)> = (0..32)
        .map(|seed| {
            let (b, _) = random_opening(4, seed);
            (b.black, b.white)
        })
        .collect();
    assert!(distinct.len() > 8, "only {} distinct openings from 32 seeds", distinct.len());
}

#[test]
fn match_from_opening_completes() {
    let eval = StaticEval;
    let (board, black_to_move) = random_opening(4, 3);
    let outcome = play_match_from(board, black_to_move, &eval, &eval, 20);
    assert!([-1.0, 0.0, 1.0].contains(&outcome));
}

// ── shared (multi-threaded) search ────────────────────────────────────────

#[test]
fn shared_search_counts_every_simulation() {
    use std::sync::atomic::AtomicU32;
    use std::sync::atomic::Ordering;

    let search = SharedSearch::new(starting_board(), true);
    let budget = AtomicU32::new(2_000);
    let take_one = || {
        budget.try_update(Ordering::Relaxed, Ordering::Relaxed, |left| left.checked_sub(1)).is_ok()
    };
    run_threads(&search, &StaticEval, 8, &take_one);

    let tree = search.into_inner();
    assert_eq!(tree.visits(), 2_000, "each simulation is one root visit");
    assert_eq!(tree.in_flight(), 0, "nothing is left marked as in flight");
    let sum: f32 = tree.policy_target().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "policy sum = {sum}");
    let best = tree.best_move().expect("a best move");
    assert_ne!(best & starting_board().legal_moves(true), 0);
}

#[test]
fn shared_search_on_one_thread_matches_plain_search() {
    let eval = StaticEval;
    let mut plain = MctsSearch::new(starting_board(), true);
    let shared = SharedSearch::new(starting_board(), true);
    for _ in 0..400 {
        plain.simulate(&eval);
        shared.simulate(&eval);
    }
    assert_eq!(shared.with(|tree| tree.policy_target()), plain.policy_target());
}

/// An evaluator that always thinks Black is ahead.
struct BlackIsBetter;

impl EvalFn for BlackIsBetter {
    fn evaluate(&self, _board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32) {
        let mut policy = [0.0_f32; 64];
        let n = legal.count_ones().max(1) as f32;
        for (i, p) in policy.iter_mut().enumerate() {
            if (legal >> i) & 1 == 1 {
                *p = 1.0 / n;
            }
        }
        (policy, if is_black { 0.5 } else { -0.5 })
    }
}

#[test]
fn root_value_is_from_the_side_to_move() {
    for is_black in [true, false] {
        let mut mcts = MctsSearch::new(starting_board(), is_black);
        assert_eq!(mcts.root_value(), 0.0);
        for _ in 0..200 {
            mcts.simulate(&BlackIsBetter);
        }
        let expected = if is_black { 0.5 } else { -0.5 };
        assert!((mcts.root_value() - expected).abs() < 1e-5, "{}", mcts.root_value());
    }
}
