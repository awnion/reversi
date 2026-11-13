use reversi_mcts::Budget;
use reversi_mcts::EvalFn;
use reversi_mcts::GameRecord;
use reversi_mcts::MatchPool;
use reversi_mcts::MctsSearch;
use reversi_mcts::PLANES_LEN;
use reversi_mcts::SelfPlayPool;
use reversi_mcts::SharedSearch;
use reversi_mcts::StaticEval;
use reversi_mcts::batch::ENDGAME_SOLVE_EMPTIES;
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
    // The first position is always black's turn and searched: its value target
    // is halfway between the final outcome and the search value (in [-1, 1]).
    if let Some(first) = record.positions.first() {
        assert!(first.is_black);
        assert!((first.outcome - 0.5 * final_outcome).abs() <= 0.5 + 1e-6, "{}", first.outcome);
    }
    // The terminal samples carry the outcome itself.
    let last = record.positions.last().expect("a finished game has positions");
    assert_eq!(last.outcome, if last.is_black { final_outcome } else { -final_outcome });
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
            let z = if pos.is_black { black_outcome } else { -black_outcome };
            if pos.policy_weight > 0.0 {
                // Searched or solved: halfway between the result and the
                // search value, which lies in [-1, 1].
                assert!((pos.outcome - z).abs() <= 1.0 + 1e-6, "{} vs {z}", pos.outcome);
                assert!(pos.outcome.abs() <= 1.0);
            } else {
                assert_eq!(pos.outcome, z);
            }
            let black_margin = (record.black_discs as f32 - record.white_discs as f32) / 64.0;
            let margin = if pos.is_black { black_margin } else { -black_margin };
            assert_eq!(pos.margin, margin);
            assert_eq!(pos.margin.signum() * z.abs(), z);
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
fn playout_cap_records_only_full_searches() {
    // Searched positions (before the endgame solver) per game: all of them
    // with a fixed budget, about a quarter with a capped one.
    let searched = |budget: Budget| {
        let mut pool = SelfPlayPool::new(16, budget);
        let records = run_pool(&mut pool, 32);
        let positions: usize = records
            .iter()
            .flat_map(|r| &r.positions)
            .filter(|p| {
                let empties = 64 - (p.board.black | p.board.white).count_ones();
                p.policy_weight > 0.0 && empties > ENDGAME_SOLVE_EMPTIES
            })
            .count();
        positions as f64 / records.len() as f64
    };
    let fixed = searched(Budget::fixed(20));
    let capped = searched(Budget::capped(20, 4, 0.25));
    assert!(fixed > 30.0, "fixed: {fixed} searched positions per game");
    assert!(capped < 0.5 * fixed, "capped: {capped}, fixed: {fixed}");
    assert!(capped > 0.1 * fixed, "capped: {capped}, fixed: {fixed}");
    assert_eq!(Budget::capped(20, 20, 0.25), Budget::fixed(20));
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

#[test]
fn into_child_keeps_the_subtree() {
    let board = starting_board();
    let eval = StaticEval;
    let mut mcts = MctsSearch::new(board, true);
    for _ in 0..200 {
        mcts.simulate(&eval);
    }
    let m = mcts.best_move().expect("a move");
    // Root visits = 1 (its own expansion) + the visits of its children.
    let child_visits = (mcts.policy_target()[m.trailing_zeros() as usize]
        * (mcts.visits() - 1) as f32)
        .round() as u32;
    let child = mcts.into_child(m).expect("the best move was visited");
    assert_eq!(child.root(), (board.apply_move(true, m), false));
    assert_eq!(child.visits(), child_visits);
    assert!(child.root_expanded());

    let mut child = child;
    for _ in 0..50 {
        child.simulate(&eval);
    }
    assert_eq!(child.visits(), child_visits + 50);
    let sum: f32 = child.policy_target().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "policy sum = {sum}");
}

#[test]
fn self_play_games_end_with_solved_moves() {
    // The last moves are solved exactly: one-hot policy targets. A game may
    // also end early, before the solver takes over; those are skipped.
    let mut pool = SelfPlayPool::new(8, 8);
    let mut checked = 0;
    for record in run_pool(&mut pool, 8) {
        let last =
            record.positions.iter().rev().find(|p| p.policy_weight > 0.0).expect("a policy sample");
        let empties = 64 - (last.board.black | last.board.white).count_ones();
        if empties > ENDGAME_SOLVE_EMPTIES {
            continue;
        }
        assert_eq!(last.mcts_policy.iter().filter(|&&p| p == 1.0).count(), 1);
        checked += 1;
    }
    assert!(checked > 0, "no game reached the endgame");
}

#[test]
fn values_stay_exact_after_many_simulations() {
    // A finished game (full board, black 40 : white 24). Every simulation
    // backs up the same exact result; past 2^24 of them an f32 sum stops
    // growing and the average drifts towards zero.
    let black = !0u64 >> 24;
    let board = Board::new(black, !black);
    let mut mcts = MctsSearch::new(board, false);
    for _ in 0..20_000_000 {
        assert!(mcts.select_leaf().is_none(), "a finished game needs no evaluation");
    }
    assert_eq!(mcts.root_value(), -1.0, "white has lost");
}

/// Plays a [`MatchPool`] to the end with `evals[net]` for each network.
fn play_arena(pool: &mut MatchPool, evals: [&dyn EvalFn; 2]) {
    loop {
        let (requests, nets) = pool.collect();
        if requests.is_empty() {
            return;
        }
        let (requests, nets) = (requests.to_vec(), nets.to_vec());
        let mut policies = Vec::new();
        let mut values = Vec::new();
        for (request, &net) in requests.iter().zip(&nets) {
            let (policy, value) =
                evals[net as usize].evaluate(request.board, request.is_black, request.legal);
            policies.extend_from_slice(&policy);
            values.push(value);
        }
        pool.apply(&policies, &values);
    }
}

#[test]
fn arena_pairs_of_equal_networks_split_the_points() {
    let mut pool = MatchPool::new(3..7, 4, 11, 32, 4);
    play_arena(&mut pool, [&StaticEval, &StaticEval]);
    assert!(pool.done());
    let points = pool.pair_points();
    assert_eq!(points.iter().map(|&(pair, _)| pair).collect::<Vec<_>>(), vec![3, 4, 5, 6]);
    // Same network on both sides, no noise: each pair is one game played
    // twice with the colours swapped.
    for (pair, p) in points {
        assert_eq!(p, 1.0, "pair {pair}");
    }
}

/// [`StaticEval`]'s hand-written score scaled to matter, optionally upside down.
struct Scaled(f32);

impl EvalFn for Scaled {
    fn evaluate(&self, board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32) {
        let (policy, value) = StaticEval.evaluate(board, is_black, legal);
        (policy, (self.0 * value).clamp(-1.0, 1.0))
    }
}

#[test]
fn arena_scores_the_stronger_network() {
    let mut pool = MatchPool::new(0..8, 2, 5, 64, 8);
    play_arena(&mut pool, [&Scaled(100.0), &Scaled(-100.0)]);
    let total: f32 = pool.pair_points().iter().map(|&(_, p)| p).sum();
    assert!(total >= 12.0, "the right way up scored {total} of 16");
}
