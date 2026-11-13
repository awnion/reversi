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
    w: f32,
    /// Simulations that passed through this node and are still waiting for
    /// their evaluation. Always 0 in a single-threaded search.
    in_flight: u32,
    parent: Option<usize>,
    children: Vec<usize>,
    expanded: bool,
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
        let root = Node {
            board,
            is_black,
            move_mask: 0,
            p: 1.0,
            n: 0,
            w: 0.0,
            in_flight: 0,
            parent: None,
            children: Vec::new(),
            expanded: false,
        };
        MctsSearch { nodes: vec![root] }
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
                node = self.nodes[i].parent;
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
            let child_idx = self.nodes.len();
            self.nodes.push(Node {
                board,
                is_black: request.is_black,
                move_mask: 0,
                p: 1.0,
                n: 0,
                w: 0.0,
                in_flight: 0,
                parent: Some(idx),
                children: Vec::new(),
                expanded: false,
            });
            self.nodes[idx].children.push(child_idx);
            self.nodes[idx].expanded = true;
            // Value from the passing player's perspective = -opponent's value.
            self.backup(idx, -value);
            return;
        }

        let is_black = request.is_black;
        let mut legal = request.legal;
        while legal != 0 {
            let i = legal.trailing_zeros() as usize;
            let move_mask = 1u64 << i;
            legal &= legal - 1;
            let child_idx = self.nodes.len();
            self.nodes.push(Node {
                board: board.apply_move(is_black, move_mask),
                is_black: !is_black,
                move_mask,
                p: policy[i],
                n: 0,
                w: 0.0,
                in_flight: 0,
                parent: Some(idx),
                children: Vec::new(),
                expanded: false,
            });
            self.nodes[idx].children.push(child_idx);
        }
        self.nodes[idx].expanded = true;
        self.backup(idx, value);
    }

    /// The search below the root's child reached by `move_mask` (`0` = the
    /// pass), as a search of its own: playing a move keeps the visits already
    /// spent on the position that follows. `None` when that child was never
    /// visited.
    pub fn into_child(self, move_mask: u64) -> Option<MctsSearch> {
        let child =
            *self.nodes[0].children.iter().find(|&&c| self.nodes[c].move_mask == move_mask)?;
        if self.nodes[child].n == 0 {
            return None;
        }
        let mut old = self.nodes;
        // Copy the subtree breadth-first; `remap[old index] = new index`.
        let mut nodes: Vec<Node> = Vec::new();
        let mut remap = vec![usize::MAX; old.len()];
        let mut queue = std::collections::VecDeque::from([child]);
        while let Some(i) = queue.pop_front() {
            remap[i] = nodes.len();
            queue.extend(old[i].children.iter().copied());
            let node = std::mem::replace(
                &mut old[i],
                Node {
                    board: Board::new(0, 0),
                    is_black: false,
                    move_mask: 0,
                    p: 0.0,
                    n: 0,
                    w: 0.0,
                    in_flight: 0,
                    parent: None,
                    children: Vec::new(),
                    expanded: false,
                },
            );
            nodes.push(node);
        }
        for node in &mut nodes {
            node.parent = node.parent.map(|p| remap[p]).filter(|&p| p != usize::MAX);
            for c in &mut node.children {
                *c = remap[*c];
            }
        }
        nodes[0].parent = None;
        nodes[0].move_mask = 0;
        nodes[0].p = 1.0;
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
        self.nodes[0]
            .children
            .iter()
            .max_by_key(|&&c| self.nodes[c].n)
            .map(|&c| self.nodes[c].move_mask)
    }

    /// Normalised visit-count distribution over all 64 squares.
    /// Used as the policy target when building training records.
    pub fn policy_target(&self) -> [f32; 64] {
        let children = &self.nodes[0].children;
        let total: u32 = children.iter().map(|&c| self.nodes[c].n).sum();
        let mut policy = [0.0_f32; 64];
        if total > 0 {
            for &c in children {
                let child = &self.nodes[c];
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
        let children = self.nodes[0].children.clone();
        if children.is_empty() || children.len() != noise.len() {
            return;
        }
        for (child_idx, &n) in children.iter().zip(noise.iter()) {
            let child = &mut self.nodes[*child_idx];
            child.p = (1.0 - epsilon) * child.p + epsilon * n;
        }
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// Walk down the tree from `start` using PUCT until reaching an unexpanded node.
    /// Expected result for the side to move at the root, from −1 (a loss) to
    /// +1 (a win): the value of its most visited move, the one it would play.
    /// Before any move has been visited, the root's own average.
    pub fn root_value(&self) -> f32 {
        let root = &self.nodes[0];
        let best = root.children.iter().map(|&c| &self.nodes[c]).max_by_key(|c| c.n);
        match best {
            // A child's value is from the opponent's side.
            Some(child) if child.n > 0 => -child.w / child.n as f32,
            _ if root.n > 0 => root.w / root.n as f32,
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
            node = self.nodes[i].parent;
        }
    }

    fn select(&self, mut idx: usize) -> usize {
        loop {
            let node = &self.nodes[idx];
            if !node.expanded || node.children.is_empty() {
                return idx;
            }
            let parent_n = node.n + node.in_flight;
            idx = *node
                .children
                .iter()
                .max_by(|&&a, &&b| {
                    let pa = self.puct(a, parent_n);
                    let pb = self.puct(b, parent_n);
                    pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
        }
    }

    /// PUCT score of `child_idx` as seen **from the parent**.
    fn puct(&self, child_idx: usize, parent_n: u32) -> f32 {
        let c = &self.nodes[child_idx];
        // In-flight simulations count as visits that the parent lost, which
        // makes the child less attractive until their results arrive.
        let visits = (c.n + c.in_flight) as f32;
        // Q from parent's perspective: parent wins when child loses → negate child's w/n.
        let q = if visits == 0.0 { 0.0 } else { (-c.w - c.in_flight as f32) / visits };
        let u = C_PUCT * c.p * (parent_n as f32).sqrt() / (1.0 + visits);
        q + u
    }

    /// Propagate `value` up to root, negating at each level (negamax).
    fn backup(&mut self, leaf_idx: usize, mut value: f32) {
        let mut idx = leaf_idx;
        loop {
            self.nodes[idx].n += 1;
            self.nodes[idx].w += value;
            value = -value;
            match self.nodes[idx].parent {
                Some(parent) => idx = parent,
                None => break,
            }
        }
    }
}
