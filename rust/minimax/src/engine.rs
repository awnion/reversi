use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use crate::board::Board;
use crate::eval::evaluate;

/// A point in time after which a search must give up.
///
/// The clock is only read every [`Deadline::CHECK_EVERY`] nodes, so a search
/// overruns by at most that many nodes instead of by a whole extra depth.
pub struct Deadline {
    /// `(clock, at)`: the search expires once `clock() >= at`.
    limit: Option<(fn() -> f64, f64)>,
    nodes: u32,
    expired: bool,
}

impl Deadline {
    const CHECK_EVERY: u32 = 1024;

    /// Expires once `clock()` reaches `at` (same unit as the clock).
    pub fn at(clock: fn() -> f64, at: f64) -> Self {
        Deadline { limit: Some((clock, at)), nodes: 0, expired: false }
    }

    /// Never expires.
    pub fn never() -> Self {
        Deadline { limit: None, nodes: 0, expired: false }
    }

    pub fn expired(&self) -> bool {
        self.expired
    }

    /// A deadline for another thread: same limit, its own node counter.
    pub fn fork(&self) -> Self {
        Deadline { limit: self.limit, nodes: 0, expired: self.expired }
    }

    /// Count one node; returns true once the time is up.
    fn tick(&mut self) -> bool {
        if self.expired {
            return true;
        }
        self.nodes = self.nodes.wrapping_add(1);
        if self.nodes.is_multiple_of(Self::CHECK_EVERY)
            && let Some((clock, at)) = self.limit
            && clock() >= at
        {
            self.expired = true;
        }
        self.expired
    }
}

/// Evaluates every legal move for `is_black` at the given depth.
/// Returns a list of `(move_bit_mask, score_for_current_player)` pairs.
pub fn evaluate_all_moves(board: Board, is_black: bool, depth: u8) -> Vec<(u64, i32)> {
    evaluate_all_moves_until(board, is_black, depth, &mut Deadline::never())
        .expect("a search without a deadline always completes")
}

/// Like [`evaluate_all_moves`], but returns `None` if the deadline passes
/// before every move has been searched. Moves are searched in parallel when
/// the `threads` feature is on.
pub fn evaluate_all_moves_until(
    board: Board,
    is_black: bool,
    depth: u8,
    deadline: &mut Deadline,
) -> Option<Vec<(u64, i32)>> {
    let moves = moves_of(board.legal_moves(is_black));
    let results = par_map(&moves, |&m| {
        let mut deadline = deadline.fork();
        let next = board.apply_move(is_black, m);
        // Search from opponent's perspective at depth-1; negate score.
        let (_, opp_score) = search_until(
            next,
            !is_black,
            depth.saturating_sub(1),
            -MAX_SCORE,
            MAX_SCORE,
            &mut deadline,
        )?;
        Some((m, -opp_score))
    });
    let results: Option<Vec<_>> = results.into_iter().collect();
    if results.is_none() {
        deadline.expired = true;
    }
    results
}

/// Alpha-beta search from the root, splitting the work between threads (with
/// the `threads` feature; otherwise one move after another).
///
/// `first` (usually the best move of the previous depth) is searched first.
/// Returns `None` if the deadline passes first.
pub fn search_root_until(
    board: Board,
    is_black: bool,
    depth: u8,
    first: Option<u64>,
    deadline: &mut Deadline,
) -> Option<(Option<u64>, i32)> {
    split_search(board, is_black, depth, MIN_SCORE, MAX_SCORE, first, deadline)
}

/// Below this depth a node is searched on one thread: splitting would cost
/// more than it saves.
const SPLIT_DEPTH: u8 = 4;

/// Principal-variation splitting: the first move is searched alone (itself
/// split the same way, one level down) to get a good bound, then the other
/// moves run in parallel, each starting from the best score found so far.
/// Fail-soft, like [`search_until`].
fn split_search(
    board: Board,
    is_black: bool,
    depth: u8,
    alpha: i32,
    beta: i32,
    first: Option<u64>,
    deadline: &mut Deadline,
) -> Option<(Option<u64>, i32)> {
    let legal = board.legal_moves(is_black);
    if depth < SPLIT_DEPTH || legal.count_ones() < 2 {
        return search_until(board, is_black, depth, alpha, beta, deadline);
    }

    let mut moves = moves_of(legal);
    // Likely best moves first: the hint, then by a static look one move
    // ahead. A good first move gives the parallel moves a tight bound.
    moves.sort_by_cached_key(|&m| evaluate(&board.apply_move(is_black, m), !is_black));
    if let Some(at) = first.and_then(|f| moves.iter().position(|&m| m == f)) {
        moves[..=at].rotate_right(1);
    }

    let next = board.apply_move(is_black, moves[0]);
    let (_, score) = split_search(next, !is_black, depth - 1, -beta, -alpha, None, deadline)?;
    let mut best = (moves[0], -score);
    if best.1 >= beta {
        return Some((Some(best.0), best.1));
    }

    let bound = AtomicI32::new(alpha.max(best.1));
    let rest = par_map(&moves[1..], |&m| {
        let alpha = bound.load(Ordering::Relaxed);
        if alpha >= beta {
            // Another move already refuted this node.
            return Some(None);
        }
        let mut deadline = deadline.fork();
        let next = board.apply_move(is_black, m);
        let (_, score) = search_until(next, !is_black, depth - 1, -beta, -alpha, &mut deadline)?;
        bound.fetch_max(-score, Ordering::Relaxed);
        Some(Some((m, -score)))
    });
    // Moves keep their search order, so ties go to the earlier move, as in a
    // sequential search.
    for result in rest {
        match result {
            None => {
                deadline.expired = true;
                return None;
            }
            Some(Some((m, score))) if score > best.1 => best = (m, score),
            Some(_) => {}
        }
    }
    Some((Some(best.0), best.1))
}

/// The set bits of `moves`, lowest first.
fn moves_of(mut moves: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(moves.count_ones() as usize);
    while moves != 0 {
        let m = 1u64 << moves.trailing_zeros();
        out.push(m);
        moves ^= m;
    }
    out
}

/// `items.map(f)`, on the rayon thread pool with the `threads` feature.
#[cfg(feature = "threads")]
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    use rayon::prelude::*;
    items.par_iter().map(f).collect()
}

#[cfg(not(feature = "threads"))]
fn par_map<T, R>(items: &[T], f: impl Fn(&T) -> R) -> Vec<R> {
    items.iter().map(f).collect()
}

pub const MIN_SCORE: i32 = -100000;
pub const MAX_SCORE: i32 = 100000;

/// Alpha-beta search to a fixed depth. Returns `(best_move, score)`.
pub fn search(
    board: Board,
    is_black: bool,
    depth: u8,
    alpha: i32,
    beta: i32,
) -> (Option<u64>, i32) {
    search_until(board, is_black, depth, alpha, beta, &mut Deadline::never())
        .expect("a search without a deadline always completes")
}

/// Like [`search`], but returns `None` if the deadline passes first. A search
/// that was cut short has no usable result.
pub fn search_until(
    board: Board,
    is_black: bool,
    depth: u8,
    mut alpha: i32,
    beta: i32,
    deadline: &mut Deadline,
) -> Option<(Option<u64>, i32)> {
    if deadline.tick() {
        return None;
    }
    if depth == 0 {
        return Some((None, evaluate(&board, is_black)));
    }

    let legal_moves = board.legal_moves(is_black);
    if legal_moves == 0 {
        let opponent_moves = board.legal_moves(!is_black);
        if opponent_moves == 0 {
            // Game over
            return Some((None, evaluate(&board, is_black)));
        } else {
            // Pass
            let (_, score) = search_until(board, !is_black, depth - 1, -beta, -alpha, deadline)?;
            return Some((None, -score));
        }
    }

    let mut best_move = None;
    let mut best_score = MIN_SCORE;

    for i in 0..64 {
        let m = 1u64 << i;
        if (legal_moves & m) != 0 {
            let next_board = board.apply_move(is_black, m);
            let (_, score) =
                search_until(next_board, !is_black, depth - 1, -beta, -alpha, deadline)?;
            let score = -score;

            if score > best_score {
                best_score = score;
                best_move = Some(m);
            }
            alpha = alpha.max(best_score);
            if alpha >= beta {
                break;
            }
        }
    }

    Some((best_move, best_score))
}
