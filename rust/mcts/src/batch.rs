//! Batched self-play: many games advance in lockstep so that their leaf
//! evaluations can be sent to the network as one batch.
//!
//! One round is [`SelfPlayPool::collect`] → evaluate the returned requests →
//! [`SelfPlayPool::apply`]. Every game contributes at most one request per
//! round, so the batch size is (at most) the number of games in the pool.
//!
//! Games that start together also finish together, which would deliver
//! training data in bursts. [`SelfPlayPool::staggered`] spreads the starts over
//! roughly one game length so that games complete at a steady rate.

use reversi_minimax::board::Board;

use crate::endgame::solve;
use crate::game::GameRecord;
use crate::game::PositionRecord;
use crate::game::SimpleRng;
use crate::game::add_root_noise;
use crate::game::random_moves;
use crate::game::sample_move_from_policy;
use crate::search::EvalRequest;
use crate::search::MctsSearch;

/// Moves sampled from the visit distribution at the start of each game.
const SELF_PLAY_TEMPERATURE_MOVES: usize = 12;
const ROOT_NOISE_EPSILON: f32 = 0.25;
/// Share of the search's own evaluation in the value target: the target is
/// `(1 - w) · z + w · q`, with `z` the game result and `q` the search value of
/// the position. The result alone is noisy (sampled opening moves, root noise,
/// later mistakes); the search value is less noisy but biased by the network.
const VALUE_TARGET_SEARCH_WEIGHT: f32 = 0.2;
/// Positions with at most this many empty squares are solved exactly
/// ([`crate::endgame::solve`]) instead of searched.
pub const ENDGAME_SOLVE_EMPTIES: u32 = 12;
/// Each game starts with a uniformly random number of uniformly random legal
/// moves in this range, not recorded: without them most games repeat earlier
/// ones move for move.
const RANDOM_OPENING_PLIES: std::ops::RangeInclusive<u32> = 4..=8;
/// Share of solved positions (at most [`ENDGAME_SOLVE_EMPTIES`] empty squares,
/// the final board included) that become training samples; recording all of
/// them filled half of the replay buffer with endgames.
const SOLVED_RECORD_FRACTION: f32 = 0.25;

/// Search budget per move: playout cap randomization (KataGo). A random
/// `full_fraction` of the moves get `full` simulations, root noise and a
/// training sample; the rest a quick `fast` search that only picks the move.
/// Many more games for the same work, and every recorded position still has a
/// full-strength policy target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    pub full: u32,
    pub fast: u32,
    pub full_fraction: f32,
}

impl Budget {
    /// Every move searched with `simulations` and recorded.
    pub fn fixed(simulations: u32) -> Self {
        Budget { full: simulations, fast: simulations, full_fraction: 1.0 }
    }

    /// `full` simulations on `full_fraction` of the moves, `fast` on the rest.
    pub fn capped(full: u32, fast: u32, full_fraction: f32) -> Self {
        if fast >= full || full_fraction >= 1.0 {
            return Self::fixed(full);
        }
        Budget { full, fast: fast.max(1), full_fraction: full_fraction.max(0.0) }
    }

    /// Average simulations per move.
    fn expected(&self) -> u32 {
        let f = self.full_fraction;
        (f * self.full as f32 + (1.0 - f) * self.fast as f32).round() as u32
    }

    /// Draw the search for the next move: `(simulations, full)`.
    fn draw(&self, rng: &mut SimpleRng) -> (u32, bool) {
        if self.full_fraction >= 1.0 || rng.next_f32() < self.full_fraction {
            (self.full, true)
        } else {
            (self.fast, false)
        }
    }
}

impl From<u32> for Budget {
    fn from(simulations: u32) -> Self {
        Budget::fixed(simulations)
    }
}

/// Number of `f32`s per position in the network input (4 planes of 8×8).
pub const PLANES_LEN: usize = 4 * 64;

/// Write the network input for `request` into `out` (`PLANES_LEN` floats):
/// my discs, opponent discs, legal moves, phase (occupied / 64).
pub fn write_planes(request: &EvalRequest, out: &mut [f32]) {
    write_position_planes(request.board, request.is_black, request.legal, out);
}

/// Write the network input for a position into `out` (`PLANES_LEN` floats):
/// the side to move's discs, the opponent's, `legal`, and the phase plane
/// (occupied squares / 64). Same encoding as `python/train/planes.py`.
pub fn write_position_planes(board: Board, is_black: bool, legal: u64, out: &mut [f32]) {
    let (mine, theirs) =
        if is_black { (board.black, board.white) } else { (board.white, board.black) };
    let phase = (board.black | board.white).count_ones() as f32 / 64.0;
    let (my_plane, rest) = out[..PLANES_LEN].split_at_mut(64);
    let (opp_plane, rest) = rest.split_at_mut(64);
    let (legal_plane, phase_plane) = rest.split_at_mut(64);
    for i in 0..64 {
        my_plane[i] = ((mine >> i) & 1) as f32;
        opp_plane[i] = ((theirs >> i) & 1) as f32;
        legal_plane[i] = ((legal >> i) & 1) as f32;
    }
    phase_plane.fill(phase);
}

/// A raw training sample, before the game outcome is known.
struct Sample {
    board: Board,
    is_black: bool,
    legal: u64,
    policy: [f32; 64],
    policy_weight: f32,
    value_weight: f32,
    /// The search's value of the position for the side to move, if searched
    /// or solved.
    q: Option<f32>,
}

/// One self-play game in progress.
struct Game {
    board: Board,
    is_black: bool,
    ply: usize,
    samples: Vec<Sample>,
    rng: SimpleRng,
    /// Search for the current move; `None` between moves.
    search: Option<MctsSearch>,
    sims_done: u32,
    /// Simulations for the current move, and whether it is a full search
    /// (noised and recorded) or a fast one.
    target: u32,
    full: bool,
    /// The subtree of the last move played, reused by the next search.
    reuse: Option<MctsSearch>,
    /// Root noise has been added to the current search.
    noised: bool,
}

impl Game {
    fn new() -> Self {
        let mut rng = SimpleRng::new();
        let (low, high) = (*RANDOM_OPENING_PLIES.start(), *RANDOM_OPENING_PLIES.end());
        let plies = low + ((rng.next_f32() * (high - low + 1) as f32) as u32).min(high - low);
        // Standard Reversi start: black on d5(35) and e4(28), white on d4(27) and e5(36).
        let start = Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36));
        let (board, is_black) = random_moves(start, true, plies, &mut rng);
        Game {
            board,
            is_black,
            ply: plies as usize,
            samples: Vec::new(),
            rng,
            search: None,
            sims_done: 0,
            target: 0,
            full: true,
            reuse: None,
            noised: false,
        }
    }

    /// Advance until the game needs an evaluation (`Ok`) or is over (`Err`
    /// with the finished record).
    fn advance(&mut self, budget: &Budget) -> Result<EvalRequest, GameRecord> {
        loop {
            if self.search.is_none() {
                // Skip forced passes; two in a row end the game.
                let mut legal = self.board.legal_moves(self.is_black);
                if legal == 0 {
                    self.is_black = !self.is_black;
                    self.reuse = self.reuse.take().and_then(|tree| tree.into_child(0));
                    legal = self.board.legal_moves(self.is_black);
                    if legal == 0 {
                        return Err(self.finish());
                    }
                }
                let empties = 64 - (self.board.black | self.board.white).count_ones();
                if empties <= ENDGAME_SOLVE_EMPTIES {
                    // Perfect play to the end, no network needed.
                    self.reuse = None;
                    let (best, score) = solve(self.board, self.is_black);
                    let m = best.expect("the side to move has a move");
                    if self.rng.next_f32() < SOLVED_RECORD_FRACTION {
                        let mut policy = [0.0; 64];
                        policy[m.trailing_zeros() as usize] = 1.0;
                        self.record_and_play(legal, policy, 1.0, Some(score.signum() as f32), m);
                    } else {
                        self.play(m);
                    }
                    continue;
                }
                if legal.count_ones() == 1 {
                    // Nothing to search; the position still teaches the value.
                    let mut policy = [0.0; 64];
                    policy[legal.trailing_zeros() as usize] = 1.0;
                    self.reuse = self.reuse.take().and_then(|tree| tree.into_child(legal));
                    self.record_and_play(legal, policy, 0.0, None, legal);
                    continue;
                }
                let search = match self.reuse.take() {
                    Some(tree) if tree.root() == (self.board, self.is_black) => tree,
                    _ => MctsSearch::new(self.board, self.is_black),
                };
                self.sims_done = search.visits();
                (self.target, self.full) = budget.draw(&mut self.rng);
                self.noised = false;
                self.search = Some(search);
                self.add_noise_once(legal);
            }

            if self.sims_done >= self.target {
                self.play_move();
                continue;
            }

            let search = self.search.as_mut().expect("search exists while simulating");
            match search.select_leaf() {
                Some(request) => return Ok(request),
                // Terminal leaf: the simulation is already backed up.
                None => self.sims_done += 1,
            }
        }
    }

    /// Perturb the root priors once they exist (after the first evaluation,
    /// or right away for a reused tree). Fast searches stay noise-free: they
    /// only pick the move.
    fn add_noise_once(&mut self, legal: u64) {
        let search = self.search.as_mut().expect("a search in progress");
        if self.full && !self.noised && search.root_expanded() {
            add_root_noise(search, legal, &mut self.rng, ROOT_NOISE_EPSILON);
            self.noised = true;
        }
    }

    /// Feed the evaluation for the request returned by [`Game::advance`].
    fn apply(&mut self, request: &EvalRequest, policy: &[f32], value: f32) {
        let legal = self.board.legal_moves(self.is_black);
        let search = self.search.as_mut().expect("apply follows a pending request");
        search.apply_eval(request, policy, value);
        self.sims_done += 1;
        self.add_noise_once(legal);
    }

    /// Record the searched position (full searches only) and play the chosen
    /// move.
    fn play_move(&mut self) {
        let search = self.search.take().expect("a finished search");
        let legal = self.board.legal_moves(self.is_black);
        let policy = search.policy_target();

        // Sample from visit counts in the opening to keep self-play diverse.
        let selected = if self.ply < SELF_PLAY_TEMPERATURE_MOVES {
            sample_move_from_policy(policy, legal, &mut self.rng).or_else(|| search.best_move())
        } else {
            search.best_move()
        };
        let m = selected.expect("a searched position has a legal move");
        let q = search.root_value();
        self.reuse = search.into_child(m);
        if self.full {
            self.record_and_play(legal, policy, 1.0, Some(q), m);
        } else {
            self.play(m);
        }
    }

    /// Add the training sample for the current position and play `m`.
    fn record_and_play(
        &mut self,
        legal: u64,
        policy: [f32; 64],
        policy_weight: f32,
        q: Option<f32>,
        m: u64,
    ) {
        let sample = |policy_weight, value_weight| Sample {
            board: self.board,
            is_black: self.is_black,
            legal,
            policy,
            policy_weight,
            value_weight,
            q,
        };
        self.samples.push(sample(policy_weight, 1.0));
        if (self.board.black | self.board.white).count_ones() == 63 {
            // Forced last move: an extra value-only sample.
            self.samples.push(sample(0.0, 2.0));
        }
        self.play(m);
    }

    fn play(&mut self, m: u64) {
        self.board = self.board.apply_move(self.is_black, m);
        self.is_black = !self.is_black;
        self.ply += 1;
    }

    fn finish(&mut self) -> GameRecord {
        let board = self.board;
        let black_discs = board.black.count_ones();
        let white_discs = board.white.count_ones();
        let black_margin = (black_discs as f32 - white_discs as f32) / 64.0;
        let black_outcome: f32 = match black_discs.cmp(&white_discs) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
        };

        // Value-only samples for the terminal board, from both sides.
        let terminal = |is_black: bool| PositionRecord {
            board,
            is_black,
            legal: 0,
            mcts_policy: [0.0; 64],
            outcome: if is_black { black_outcome } else { -black_outcome },
            margin: if is_black { black_margin } else { -black_margin },
            policy_weight: 0.0,
            value_weight: 2.0,
            search_value: f32::NAN,
        };
        let keep_terminal = [
            self.rng.next_f32() < SOLVED_RECORD_FRACTION,
            self.rng.next_f32() < SOLVED_RECORD_FRACTION,
        ];
        let positions = std::mem::take(&mut self.samples)
            .into_iter()
            .map(|s| {
                let z = if s.is_black { black_outcome } else { -black_outcome };
                let w = VALUE_TARGET_SEARCH_WEIGHT;
                let value = s.q.map_or(z, |q| (1.0 - w) * z + w * q);
                (s, value)
            })
            .map(|(s, value)| PositionRecord {
                board: s.board,
                is_black: s.is_black,
                legal: s.legal,
                mcts_policy: s.policy,
                outcome: value,
                margin: if s.is_black { black_margin } else { -black_margin },
                policy_weight: s.policy_weight,
                value_weight: s.value_weight,
                search_value: s.q.unwrap_or(f32::NAN),
            })
            .chain(
                [true, false]
                    .into_iter()
                    .zip(keep_terminal)
                    .filter(|(_, k)| *k)
                    .map(|(b, _)| terminal(b)),
            )
            .collect();

        GameRecord { positions, black_discs, white_discs }
    }
}

/// Typical number of searched moves in a game, used to spread game starts.
const TYPICAL_GAME_MOVES: u64 = 60;

/// A set of self-play games advanced together, one simulation per round.
pub struct SelfPlayPool {
    budget: Budget,
    games: Vec<Game>,
    /// `start_round[i]`: first round in which game `i` takes part.
    start_round: Vec<u64>,
    round: u64,
    /// Games at or beyond this index are not replaced when they finish.
    limit: usize,
    /// `retired[i]`: game `i` finished beyond the limit and is left idle.
    retired: Vec<bool>,
    /// Requests of the current round; `requests[i]` belongs to `owners[i]`.
    requests: Vec<EvalRequest>,
    owners: Vec<usize>,
    finished: Vec<GameRecord>,
}

impl SelfPlayPool {
    /// `games` concurrent games with a search `budget` per move (a plain
    /// number of simulations, or a [`Budget`]); at least one simulation, so
    /// that every move has a policy target.
    pub fn new(games: usize, budget: impl Into<Budget>) -> Self {
        let mut budget = budget.into();
        budget.full = budget.full.max(1);
        budget.fast = budget.fast.max(1);
        SelfPlayPool {
            budget,
            games: (0..games).map(|_| Game::new()).collect(),
            start_round: vec![0; games],
            round: 0,
            limit: games,
            retired: vec![false; games],
            requests: Vec::with_capacity(games),
            owners: Vec::with_capacity(games),
            finished: Vec::new(),
        }
    }

    /// Like [`SelfPlayPool::new`], but the games start one after another over
    /// about one game length. Batches grow from one request to `games` during
    /// that ramp-up; afterwards games finish evenly instead of all at once.
    pub fn staggered(games: usize, budget: impl Into<Budget>) -> Self {
        let mut pool = Self::new(games, budget);
        let game_rounds = u64::from(pool.budget.expected()) * TYPICAL_GAME_MOVES;
        for (i, start) in pool.start_round.iter_mut().enumerate() {
            *start = game_rounds * i as u64 / games.max(1) as u64;
        }
        pool
    }

    /// Games in play once the change of [`SelfPlayPool::set_games`] has
    /// settled.
    pub fn games(&self) -> usize {
        self.limit
    }

    /// Play `games` games from now on: new ones start in the next round;
    /// above the new number, games play to the end and are not replaced.
    pub fn set_games(&mut self, games: usize) {
        let games = games.max(1);
        while self.games.len() < games {
            self.games.push(Game::new());
            self.start_round.push(self.round);
            self.retired.push(false);
        }
        for index in 0..games {
            if self.retired[index] {
                self.games[index] = Game::new();
                self.start_round[index] = self.round;
                self.retired[index] = false;
            }
        }
        self.limit = games;
    }

    /// Advance every game to its next evaluation and return the batch.
    ///
    /// Games that finish on the way are moved to the finished list and
    /// replaced by fresh ones (below the limit of
    /// [`SelfPlayPool::set_games`]), so the batch has one request per game in
    /// play.
    pub fn collect(&mut self) -> &[EvalRequest] {
        assert!(self.requests.is_empty(), "collect() called twice without apply()");
        let round = self.round;
        self.round += 1;
        for (index, game) in self.games.iter_mut().enumerate() {
            if self.retired[index] || self.start_round[index] > round {
                continue;
            }
            let request = loop {
                match game.advance(&self.budget) {
                    Ok(request) => break Some(request),
                    Err(record) => {
                        self.finished.push(record);
                        if index >= self.limit {
                            self.retired[index] = true;
                            break None;
                        }
                        *game = Game::new();
                    }
                }
            };
            let Some(request) = request else { continue };
            self.requests.push(request);
            self.owners.push(index);
        }
        &self.requests
    }

    /// Apply the evaluations of the batch returned by the last
    /// [`SelfPlayPool::collect`]: `policies` has 64 priors per request,
    /// `values` one value per request, both in request order.
    pub fn apply(&mut self, policies: &[f32], values: &[f32]) {
        assert_eq!(values.len(), self.requests.len(), "one value per request");
        assert_eq!(policies.len(), self.requests.len() * 64, "64 priors per request");
        for (i, (request, &owner)) in self.requests.iter().zip(&self.owners).enumerate() {
            self.games[owner].apply(request, &policies[i * 64..(i + 1) * 64], values[i]);
        }
        self.requests.clear();
        self.owners.clear();
    }

    /// Games completed since the last call.
    pub fn take_finished(&mut self) -> Vec<GameRecord> {
        std::mem::take(&mut self.finished)
    }
}
