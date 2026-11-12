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
        let idx = self.select(0);
        let board = self.nodes[idx].board;
        let is_black = self.nodes[idx].is_black;

        let legal = board.legal_moves(is_black);
        if legal != 0 {
            return Some(EvalRequest { node: idx, board, is_black, legal, pass: false });
        }

        let opp_legal = board.legal_moves(!is_black);
        if opp_legal != 0 {
            // Forced pass: the position is evaluated from the opponent's side.
            return Some(EvalRequest {
                node: idx,
                board,
                is_black: !is_black,
                legal: opp_legal,
                pass: true,
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
                parent: Some(idx),
                children: Vec::new(),
                expanded: false,
            });
            self.nodes[idx].children.push(child_idx);
        }
        self.nodes[idx].expanded = true;
        self.backup(idx, value);
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
    fn select(&self, mut idx: usize) -> usize {
        loop {
            let node = &self.nodes[idx];
            if !node.expanded || node.children.is_empty() {
                return idx;
            }
            let parent_n = node.n;
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
        // Q from parent's perspective: parent wins when child loses → negate child's w/n.
        let q = if c.n == 0 { 0.0 } else { -(c.w / c.n as f32) };
        let u = C_PUCT * c.p * (parent_n as f32).sqrt() / (1.0 + c.n as f32);
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
