use reversi_minimax::board::Board;
use reversi_minimax::eval::evaluate;

/// Exploration constant for PUCT formula.
const C_PUCT: f32 = 1.5;

// ── EvalFn ────────────────────────────────────────────────────────────────────

/// Pluggable position evaluator.
///
/// Returns `(policy, value)` where:
/// - `policy[i]` is the prior probability for placing a disc at bit-index `i`
///   (non-legal squares should have probability 0)
/// - `value` ∈ [-1, 1] from the **current player's** perspective
pub trait EvalFn {
    fn evaluate(&self, board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32);
}

/// Phase-1 evaluator: uniform prior over legal moves + normalized minimax score.
pub struct StaticEval;

impl EvalFn for StaticEval {
    fn evaluate(&self, board: Board, is_black: bool, legal: u64) -> ([f32; 64], f32) {
        let score = evaluate(&board, is_black);
        let value = (score as f32 / 10_000.0).clamp(-1.0, 1.0);

        let n_legal = legal.count_ones() as f32;
        let mut policy = [0.0_f32; 64];
        if n_legal > 0.0 {
            for (i, p) in policy.iter_mut().enumerate() {
                if (legal >> i) & 1 == 1 {
                    *p = 1.0 / n_legal;
                }
            }
        }

        (policy, value)
    }
}

// ── EvalRequest ───────────────────────────────────────────────────────────────

/// A leaf position waiting for a network evaluation.
///
/// Produced by [`MctsSearch::select_leaf`], consumed by
/// [`MctsSearch::apply_eval`].
#[derive(Clone, Copy, Debug)]
pub struct EvalRequest {
    /// Index of the leaf inside its tree.
    node: usize,
    pub board: Board,
    /// Side to evaluate for (the opponent of the leaf's player when `pass`).
    pub is_black: bool,
    /// Legal moves of `is_black` on `board`; never empty.
    pub legal: u64,
    /// The leaf's player has no moves and must pass.
    pass: bool,
    /// The simulation is counted as in flight along its path (shared search).
    in_flight: bool,
}

// ── Node ──────────────────────────────────────────────────────────────────────

/// No parent (the root).
const NO_PARENT: u32 = u32::MAX;

/// A tree node. The children of a node are created together and stored next
/// to each other, so a node only records where they start and how many there
/// are: no allocation per node, and a subtree copies as plain data.
#[derive(Clone, Copy)]
struct Node {
    board: Board,
    is_black: bool,
    /// The move (bitmask) that produced this node from its parent. 0 = root or pass.
    move_mask: u64,
    /// Prior probability assigned by the parent's eval.
    p: f32,
    /// Visit count.
    n: u32,
    /// Accumulated value from **this node's player's** perspective.
    /// f64: an f32 sum stops counting past 2^24 visits, and an analysis
    /// left running on a nearly finished game gets there.
    w: f64,
    /// Simulations that passed through this node and are still waiting for
    /// their evaluation. Always 0 in a single-threaded search.
    in_flight: u32,
    /// Index of the parent, or [`NO_PARENT`].
    parent: u32,
    /// Index of the first child; the children are `first_child..first_child +
    /// child_count`.
    first_child: u32,
    child_count: u32,
    expanded: bool,
}

impl Node {
    fn new(board: Board, is_black: bool, move_mask: u64, p: f32, parent: u32) -> Self {
        Node {
            board,
            is_black,
            move_mask,
            p,
            n: 0,
            w: 0.0,
            in_flight: 0,
            parent,
            first_child: 0,
            child_count: 0,
            expanded: false,
        }
    }

    fn children(&self) -> std::ops::Range<usize> {
        self.first_child as usize..(self.first_child + self.child_count) as usize
    }

    fn parent(&self) -> Option<usize> {
        (self.parent != NO_PARENT).then_some(self.parent as usize)
    }
}

// ── MctsSearch ────────────────────────────────────────────────────────────────

/// Single-position MCTS tree.
///
/// Create with [`MctsSearch::new`], run [`MctsSearch::simulate`] N times,
/// then call [`MctsSearch::best_move`] or [`MctsSearch::policy_target`].
pub struct MctsSearch {
    nodes: Vec<Node>,
}

impl MctsSearch {
    pub fn new(board: Board, is_black: bool) -> Self {
        MctsSearch { nodes: vec![Node::new(board, is_black, 0, 1.0, NO_PARENT)] }
    }

    /// Run one MCTS simulation: select → expand+eval → backup.
    pub fn simulate(&mut self, eval: &dyn EvalFn) {
        if let Some(request) = self.select_leaf() {
            let (policy, value) = eval.evaluate(request.board, request.is_black, request.legal);
            self.apply_eval(&request, &policy, value);
        }
    }

    /// First half of a simulation: walk down to a leaf.
    ///
    /// Returns the position that needs a network evaluation, or `None` when the
    /// leaf was terminal and the simulation has already been backed up. Pair
    /// every `Some` with exactly one [`MctsSearch::apply_eval`] before selecting
    /// again; this split lets a caller batch evaluations across many trees.
    pub fn select_leaf(&mut self) -> Option<EvalRequest> {
        self.select_leaf_impl(false)
    }

    /// [`MctsSearch::select_leaf`] for a search shared between threads: the
    /// simulation is marked as in flight along its path ("virtual loss"), so
    /// that other threads selecting meanwhile are steered to other leaves.
    /// [`MctsSearch::apply_eval`] clears the mark.
    pub fn select_leaf_shared(&mut self) -> Option<EvalRequest> {
        self.select_leaf_impl(true)
    }

    fn select_leaf_impl(&mut self, shared: bool) -> Option<EvalRequest> {
        let idx = self.select(0);
        let board = self.nodes[idx].board;
        let is_black = self.nodes[idx].is_black;

        let legal = board.legal_moves(is_black);
        if legal != 0 {
            self.mark_in_flight(idx, shared);
            return Some(EvalRequest {
                node: idx,
                board,
                is_black,
                legal,
                pass: false,
                in_flight: shared,
            });
        }

        let opp_legal = board.legal_moves(!is_black);
        if opp_legal != 0 {
            // Forced pass: the position is evaluated from the opponent's side.
            self.mark_in_flight(idx, shared);
            return Some(EvalRequest {
                node: idx,
                board,
                is_black: !is_black,
                legal: opp_legal,
                pass: true,
                in_flight: shared,
            });
        }

        // Terminal position: count discs.
        let b = board.black.count_ones() as i32;
        let w = board.white.count_ones() as i32;
        let outcome: f32 = match b.cmp(&w) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
        };
        self.nodes[idx].expanded = true;
        self.backup(idx, if is_black { outcome } else { -outcome });
        None
    }

    /// Second half of a simulation: expand the leaf from `request` with the
    /// evaluator's answer and back the value up.
    ///
    /// `policy` holds one prior per square (64 entries); `value` is from the
    /// perspective of `request.is_black`.
    pub fn apply_eval(&mut self, request: &EvalRequest, policy: &[f32], value: f32) {
        let idx = request.node;
        let board = request.board;

        if request.in_flight {
            let mut node = Some(idx);
            while let Some(i) = node {
                self.nodes[i].in_flight -= 1;
                node = self.nodes[i].parent();
            }
        }
        // In a shared search another thread may have expanded this leaf in the
        // meantime; the evaluation still counts as a visit.
        if self.nodes[idx].expanded {
            self.backup(idx, if request.pass { -value } else { value });
            return;
        }

        if request.pass {
            // `request.is_black` is the opponent, who moves next from this node.
            let first = self.nodes.len() as u32;
            self.nodes.push(Node::new(board, request.is_black, 0, 1.0, idx as u32));
            let node = &mut self.nodes[idx];
            node.first_child = first;
            node.child_count = 1;
            node.expanded = true;
            // Value from the passing player's perspective = -opponent's value.
            self.backup(idx, -value);
            return;
        }

        let is_black = request.is_black;
        let first = self.nodes.len() as u32;
        let mut legal = request.legal;
        while legal != 0 {
            let i = legal.trailing_zeros() as usize;
            let move_mask = 1u64 << i;
            legal &= legal - 1;
            let child = board.apply_move(is_black, move_mask);
            self.nodes.push(Node::new(child, !is_black, move_mask, policy[i], idx as u32));
        }
        let node = &mut self.nodes[idx];
        node.first_child = first;
        node.child_count = request.legal.count_ones();
        node.expanded = true;
        self.backup(idx, value);
    }

    /// The search below the root's child reached by `move_mask` (`0` = the
    /// pass), as a search of its own: playing a move keeps the visits already
    /// spent on the position that follows. `None` when that child was never
    /// visited.
    pub fn into_child(self, move_mask: u64) -> Option<MctsSearch> {
        let old = self.nodes;
        let child = old[0].children().find(|&c| old[c].move_mask == move_mask)?;
        if old[child].n == 0 {
            return None;
        }
        // Copy the subtree breadth-first. The new array is its own queue, and
        // siblings stay next to each other: they are appended together.
        let mut nodes: Vec<Node> = Vec::with_capacity(old.len());
        let mut root = old[child];
        root.parent = NO_PARENT;
        root.move_mask = 0;
        root.p = 1.0;
        nodes.push(root);
        let mut next = 0;
        while next < nodes.len() {
            let children = nodes[next].children();
            if !children.is_empty() {
                nodes[next].first_child = nodes.len() as u32;
                nodes.extend(old[children].iter().map(|&c| Node { parent: next as u32, ..c }));
            }
            next += 1;
        }
        Some(MctsSearch { nodes })
    }

    /// The position the search is about.
    pub fn root(&self) -> (Board, bool) {
        (self.nodes[0].board, self.nodes[0].is_black)
    }

    /// The root has been evaluated and has children with priors.
    pub fn root_expanded(&self) -> bool {
        self.nodes[0].expanded
    }

    /// Move with the highest visit count (deterministic, no temperature).
    pub fn best_move(&self) -> Option<u64> {
        self.nodes[0].children().max_by_key(|&c| self.nodes[c].n).map(|c| self.nodes[c].move_mask)
    }

    /// Normalised visit-count distribution over all 64 squares.
    /// Used as the policy target when building training records.
    pub fn policy_target(&self) -> [f32; 64] {
        let children = &self.nodes[self.nodes[0].children()];
        let total: u32 = children.iter().map(|c| c.n).sum();
        let mut policy = [0.0_f32; 64];
        if total > 0 {
            for child in children {
                if child.move_mask != 0 {
                    let i = child.move_mask.trailing_zeros() as usize;
                    policy[i] = child.n as f32 / total as f32;
                }
            }
        }
        policy
    }

    /// Blend root priors with externally-supplied exploration noise.
    ///
    /// `noise` must have one entry per root child and sum to 1.
    pub fn add_root_noise(&mut self, noise: &[f32], epsilon: f32) {
        let children = self.nodes[0].children();
        if children.is_empty() || children.len() != noise.len() {
            return;
        }
        for (child, &n) in self.nodes[children].iter_mut().zip(noise) {
            child.p = (1.0 - epsilon) * child.p + epsilon * n;
        }
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// Expected result for the side to move at the root, from −1 (a loss) to
    /// +1 (a win): the value of its most visited move, the one it would play.
    /// Before any move has been visited, the root's own average.
    pub fn root_value(&self) -> f32 {
        let root = &self.nodes[0];
        let best = self.nodes[root.children()].iter().max_by_key(|c| c.n);
        match best {
            // A child's value is from the opponent's side.
            Some(child) if child.n > 0 => (-child.w / f64::from(child.n)) as f32,
            _ if root.n > 0 => (root.w / f64::from(root.n)) as f32,
            _ => 0.0,
        }
    }

    /// Nodes in the tree; each one costs memory.
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total visits of the root, i.e. the number of finished simulations.
    pub fn visits(&self) -> u32 {
        self.nodes[0].n
    }

    /// Simulations selected but not yet applied (shared search only).
    pub fn in_flight(&self) -> u32 {
        self.nodes[0].in_flight
    }

    fn mark_in_flight(&mut self, leaf: usize, shared: bool) {
        if !shared {
            return;
        }
        let mut node = Some(leaf);
        while let Some(i) = node {
            self.nodes[i].in_flight += 1;
            node = self.nodes[i].parent();
        }
    }

    fn select(&self, mut idx: usize) -> usize {
        loop {
            let node = &self.nodes[idx];
            if !node.expanded || node.child_count == 0 {
                return idx;
            }
            // The exploration factor is the same for every child: once per node.
            let explore = C_PUCT * ((node.n + node.in_flight) as f32).sqrt();
            let first = node.first_child as usize;
            let mut best = first;
            let mut best_score = f32::NEG_INFINITY;
            for (i, child) in self.nodes[node.children()].iter().enumerate() {
                let score = puct(child, explore);
                // `>=`: the last of equal scores, as `Iterator::max_by` picks.
                if score >= best_score {
                    best_score = score;
                    best = first + i;
                }
            }
            idx = best;
        }
    }

    /// Propagate `value` up to root, negating at each level (negamax).
    fn backup(&mut self, leaf_idx: usize, mut value: f32) {
        let mut idx = leaf_idx;
        loop {
            self.nodes[idx].n += 1;
            self.nodes[idx].w += f64::from(value);
            value = -value;
            match self.nodes[idx].parent() {
                Some(parent) => idx = parent,
                None => break,
            }
        }
    }
}

/// PUCT score of `child` as seen **from the parent**; `explore` is
/// `C_PUCT * sqrt(parent visits)`.
#[inline]
fn puct(child: &Node, explore: f32) -> f32 {
    // In-flight simulations count as visits that the parent lost, which makes
    // the child less attractive until their results arrive.
    let visits = (child.n + child.in_flight) as f32;
    // Q from the parent's side: the parent wins when the child loses.
    let q = if visits == 0.0 { 0.0 } else { (-child.w as f32 - child.in_flight as f32) / visits };
    q + child.p * explore / (1.0 + visits)
}
