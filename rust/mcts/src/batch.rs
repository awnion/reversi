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

use crate::game::GameRecord;
use crate::game::PositionRecord;
use crate::game::SimpleRng;
use crate::game::add_root_noise;
use crate::game::sample_move_from_policy;
use crate::search::EvalRequest;
use crate::search::MctsSearch;

/// Moves sampled from the visit distribution at the start of each game.
const SELF_PLAY_TEMPERATURE_MOVES: usize = 12;
const ROOT_NOISE_EPSILON: f32 = 0.25;

/// Number of `f32`s per position in the network input (4 planes of 8×8).
pub const PLANES_LEN: usize = 4 * 64;

/// Write the network input for `request` into `out` (`PLANES_LEN` floats):
/// my discs, opponent discs, legal moves, phase (occupied / 64).
pub fn write_planes(request: &EvalRequest, out: &mut [f32]) {
    let board = request.board;
    let (mine, theirs) =
        if request.is_black { (board.black, board.white) } else { (board.white, board.black) };
    let phase = (board.black | board.white).count_ones() as f32 / 64.0;
    let (my_plane, rest) = out[..PLANES_LEN].split_at_mut(64);
    let (opp_plane, rest) = rest.split_at_mut(64);
    let (legal_plane, phase_plane) = rest.split_at_mut(64);
    for i in 0..64 {
        my_plane[i] = ((mine >> i) & 1) as f32;
        opp_plane[i] = ((theirs >> i) & 1) as f32;
        legal_plane[i] = ((request.legal >> i) & 1) as f32;
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
}

impl Game {
    fn new() -> Self {
        Game {
            // Standard Reversi start: black on d5(35) and e4(28), white on d4(27) and e5(36).
            board: Board::new((1u64 << 28) | (1u64 << 35), (1u64 << 27) | (1u64 << 36)),
            is_black: true,
            ply: 0,
            samples: Vec::new(),
            rng: SimpleRng::new(),
            search: None,
            sims_done: 0,
        }
    }

    /// Advance until the game needs an evaluation (`Ok`) or is over (`Err`
    /// with the finished record).
    fn advance(&mut self, simulations: u32) -> Result<EvalRequest, GameRecord> {
        loop {
            if self.search.is_none() {
                // Skip forced passes; two in a row end the game.
                if self.board.legal_moves(self.is_black) == 0 {
                    self.is_black = !self.is_black;
                    if self.board.legal_moves(self.is_black) == 0 {
                        return Err(self.finish());
                    }
                }
                self.search = Some(MctsSearch::new(self.board, self.is_black));
                self.sims_done = 0;
            }

            if self.sims_done >= simulations {
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

    /// Feed the evaluation for the request returned by [`Game::advance`].
    fn apply(&mut self, request: &EvalRequest, policy: &[f32], value: f32) {
        let legal = self.board.legal_moves(self.is_black);
        let search = self.search.as_mut().expect("apply follows a pending request");
        search.apply_eval(request, policy, value);
        self.sims_done += 1;
        if self.sims_done == 1 {
            // The first simulation expands the root; perturb its priors.
            add_root_noise(search, legal, &mut self.rng, ROOT_NOISE_EPSILON);
        }
    }

    /// Record the searched position and play the chosen move.
    fn play_move(&mut self) {
        let search = self.search.take().expect("a finished search");
        let legal = self.board.legal_moves(self.is_black);
        let policy = search.policy_target();

        let sample = |policy_weight, value_weight| Sample {
            board: self.board,
            is_black: self.is_black,
            legal,
            policy,
            policy_weight,
            value_weight,
        };
        self.samples.push(sample(1.0, 1.0));
        if (self.board.black | self.board.white).count_ones() == 63 {
            // Forced last move: an extra value-only sample.
            self.samples.push(sample(0.0, 2.0));
        }

        // Sample from visit counts in the opening to keep self-play diverse.
        let selected = if self.ply < SELF_PLAY_TEMPERATURE_MOVES {
            sample_move_from_policy(policy, legal, &mut self.rng).or_else(|| search.best_move())
        } else {
            search.best_move()
        };
        if let Some(m) = selected {
            self.board = self.board.apply_move(self.is_black, m);
        }
        self.is_black = !self.is_black;
        self.ply += 1;
    }

    fn finish(&mut self) -> GameRecord {
        let board = self.board;
        let black_discs = board.black.count_ones();
        let white_discs = board.white.count_ones();
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
            policy_weight: 0.0,
            value_weight: 2.0,
        };
        let positions = std::mem::take(&mut self.samples)
            .into_iter()
            .map(|s| PositionRecord {
                board: s.board,
                is_black: s.is_black,
                legal: s.legal,
                mcts_policy: s.policy,
                outcome: if s.is_black { black_outcome } else { -black_outcome },
                policy_weight: s.policy_weight,
                value_weight: s.value_weight,
            })
            .chain([terminal(true), terminal(false)])
            .collect();

        GameRecord { positions, black_discs, white_discs }
    }
}

/// Typical number of searched moves in a game, used to spread game starts.
const TYPICAL_GAME_MOVES: u64 = 60;

/// A set of self-play games advanced together, one simulation per round.
pub struct SelfPlayPool {
    simulations: u32,
    games: Vec<Game>,
    /// `start_round[i]`: first round in which game `i` takes part.
    start_round: Vec<u64>,
    round: u64,
    /// Requests of the current round; `requests[i]` belongs to `owners[i]`.
    requests: Vec<EvalRequest>,
    owners: Vec<usize>,
    finished: Vec<GameRecord>,
}

impl SelfPlayPool {
    /// `games` concurrent games with `simulations` MCTS simulations per move
    /// (at least one, so that every move has a policy target).
    pub fn new(games: usize, simulations: u32) -> Self {
        SelfPlayPool {
            simulations: simulations.max(1),
            games: (0..games).map(|_| Game::new()).collect(),
            start_round: vec![0; games],
            round: 0,
            requests: Vec::with_capacity(games),
            owners: Vec::with_capacity(games),
            finished: Vec::new(),
        }
    }

    /// Like [`SelfPlayPool::new`], but the games start one after another over
    /// about one game length. Batches grow from one request to `games` during
    /// that ramp-up; afterwards games finish evenly instead of all at once.
    pub fn staggered(games: usize, simulations: u32) -> Self {
        let mut pool = Self::new(games, simulations);
        let game_rounds = u64::from(pool.simulations) * TYPICAL_GAME_MOVES;
        for (i, start) in pool.start_round.iter_mut().enumerate() {
            *start = game_rounds * i as u64 / games.max(1) as u64;
        }
        pool
    }

    /// Advance every game to its next evaluation and return the batch.
    ///
    /// Games that finish on the way are moved to the finished list and
    /// replaced by fresh ones, so the batch has one request per started game.
    pub fn collect(&mut self) -> &[EvalRequest] {
        assert!(self.requests.is_empty(), "collect() called twice without apply()");
        let round = self.round;
        self.round += 1;
        for (index, game) in self.games.iter_mut().enumerate() {
            if self.start_round[index] > round {
                continue;
            }
            let request = loop {
                match game.advance(self.simulations) {
                    Ok(request) => break request,
                    Err(record) => {
                        self.finished.push(record);
                        *game = Game::new();
                    }
                }
            };
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
