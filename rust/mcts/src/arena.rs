//! Batched matches between two networks: many games advance in lockstep, so
//! that their leaf evaluations reach a GPU as large batches.
//!
//! Each game is one half of a pair: a random opening played once with each
//! network as black. Each player searches its own tree, without noise, and
//! plays the most visited move; positions with at most
//! [`ENDGAME_SOLVE_EMPTIES`] empty squares are solved exactly, as in self-play.
//!
//! One round is [`MatchPool::collect`] → evaluate every request with the
//! network named next to it → [`MatchPool::apply`]. A game selects up
//! to `leaves_per_round` leaves per round, kept apart by virtual loss.

use reversi_minimax::board::Board;

use crate::batch::ENDGAME_SOLVE_EMPTIES;
use crate::endgame::solve;
use crate::game::random_opening;
use crate::search::EvalRequest;
use crate::search::MctsSearch;

/// One game of a pair.
struct ArenaGame {
    pair: usize,
    /// Network 0 plays black.
    first_is_black: bool,
    board: Board,
    is_black: bool,
    search: Option<MctsSearch>,
    /// Simulations started for the current move (applied, in flight or
    /// terminal).
    started: u32,
    /// Points of network 0, once the game is over.
    result: Option<f32>,
}

impl ArenaGame {
    fn net_to_move(&self) -> u8 {
        u8::from(self.is_black != self.first_is_black)
    }

    /// Play forced passes, solved endgames and finished searches until the
    /// game needs a search (`true`) or is over (`false`).
    fn advance(&mut self, simulations: u32) -> bool {
        loop {
            if self.result.is_some() {
                return false;
            }
            if let Some(search) = &self.search {
                if self.started < simulations || search.in_flight() > 0 {
                    return true;
                }
                let m = search.best_move().expect("a searched position has a move");
                self.search = None;
                self.play(m);
                continue;
            }
            let mut legal = self.board.legal_moves(self.is_black);
            if legal == 0 {
                self.is_black = !self.is_black;
                legal = self.board.legal_moves(self.is_black);
                if legal == 0 {
                    self.finish();
                    return false;
                }
            }
            let empties = 64 - (self.board.black | self.board.white).count_ones();
            if empties <= ENDGAME_SOLVE_EMPTIES {
                let (best, _) = solve(self.board, self.is_black);
                self.play(best.expect("the side to move has a move"));
                continue;
            }
            if legal.count_ones() == 1 {
                self.play(legal);
                continue;
            }
            self.search = Some(MctsSearch::new(self.board, self.is_black));
            self.started = 0;
        }
    }

    fn play(&mut self, m: u64) {
        self.board = self.board.apply_move(self.is_black, m);
        self.is_black = !self.is_black;
    }

    fn finish(&mut self) {
        let (black, white) = (self.board.black.count_ones(), self.board.white.count_ones());
        let black_points = match black.cmp(&white) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => 0.0,
            std::cmp::Ordering::Equal => 0.5,
        };
        self.result = Some(if self.first_is_black { black_points } else { 1.0 - black_points });
    }
}

/// Games of a match between network 0 and network 1.
pub struct MatchPool {
    games: Vec<ArenaGame>,
    simulations: u32,
    leaves_per_round: u32,
    /// Requests of the current round: `requests[i]` belongs to game
    /// `owners[i]` and is for network `nets[i]`.
    requests: Vec<EvalRequest>,
    owners: Vec<usize>,
    nets: Vec<u8>,
}

impl MatchPool {
    /// Both games of every pair in `pairs`; pair `p` starts from
    /// [`random_opening`]`(opening_plies, seed + p)`.
    pub fn new(
        pairs: std::ops::Range<usize>,
        opening_plies: u32,
        seed: u64,
        simulations: u32,
        leaves_per_round: u32,
    ) -> Self {
        let mut games = Vec::with_capacity(2 * pairs.len());
        for pair in pairs {
            let (board, is_black) = random_opening(opening_plies, seed.wrapping_add(pair as u64));
            for first_is_black in [true, false] {
                games.push(ArenaGame {
                    pair,
                    first_is_black,
                    board,
                    is_black,
                    search: None,
                    started: 0,
                    result: None,
                });
            }
        }
        MatchPool {
            games,
            simulations: simulations.max(1),
            leaves_per_round: leaves_per_round.max(1),
            requests: Vec::new(),
            owners: Vec::new(),
            nets: Vec::new(),
        }
    }

    /// Advance every game to its next evaluations and return them with the
    /// network for each; empty once every game is over.
    pub fn collect(&mut self) -> (&[EvalRequest], &[u8]) {
        assert!(self.requests.is_empty(), "collect() called twice without apply()");
        // A round can end up without requests (only terminal leaves) while
        // games go on; collect again then.
        while self.requests.is_empty() && !self.done() {
            self.collect_round();
        }
        (&self.requests, &self.nets)
    }

    fn collect_round(&mut self) {
        for (index, game) in self.games.iter_mut().enumerate() {
            // Terminal leaves need no evaluation: select again, a bounded
            // number of times so that a solved tree cannot spin forever.
            let mut tries = 4 * self.leaves_per_round;
            let mut selected = 0;
            while selected < self.leaves_per_round && tries > 0 && game.advance(self.simulations) {
                if game.started >= self.simulations {
                    break;
                }
                tries -= 1;
                let net = game.net_to_move();
                let search = game.search.as_mut().expect("advance() left a search");
                game.started += 1;
                if let Some(request) = search.select_leaf_shared() {
                    self.requests.push(request);
                    self.owners.push(index);
                    self.nets.push(net);
                    selected += 1;
                }
            }
        }
    }

    /// Evaluations for the last batch, in request order: 64 priors and one
    /// value per request, each from the network `collect` named.
    pub fn apply(&mut self, policies: &[f32], values: &[f32]) {
        assert_eq!(values.len(), self.requests.len(), "one value per request");
        assert_eq!(policies.len(), self.requests.len() * 64, "64 priors per request");
        for (i, (request, &owner)) in self.requests.iter().zip(&self.owners).enumerate() {
            let search = self.games[owner].search.as_mut().expect("a search awaits its leaf");
            search.apply_eval(request, &policies[i * 64..(i + 1) * 64], values[i]);
        }
        self.requests.clear();
        self.owners.clear();
        self.nets.clear();
    }

    /// Every game is over.
    pub fn done(&mut self) -> bool {
        let simulations = self.simulations;
        self.games.iter_mut().all(|game| !game.advance(simulations))
    }

    /// Points of network 0 per pair (0 to 2), for the pairs of this pool.
    pub fn pair_points(&self) -> Vec<(usize, f32)> {
        let mut points: Vec<(usize, f32)> = Vec::new();
        for game in &self.games {
            let p = game.result.unwrap_or(0.5);
            match points.last_mut() {
                Some((pair, total)) if *pair == game.pair => *total += p,
                _ => points.push((game.pair, p)),
            }
        }
        points
    }
}
