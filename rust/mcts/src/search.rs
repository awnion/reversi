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
            for i in 0..64_usize {
                if (legal >> i) & 1 == 1 {
                    policy[i] = 1.0 / n_legal;
                }
            }
        }

        (policy, value)
    }
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
        let leaf = self.select(0);
        let value = self.expand_and_eval(leaf, eval);
        self.backup(leaf, value);
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
                    self.puct(a, parent_n).partial_cmp(&self.puct(b, parent_n)).unwrap()
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

    /// Expand `idx`, evaluate with `eval`, return value from that node's perspective.
    fn expand_and_eval(&mut self, idx: usize, eval: &dyn EvalFn) -> f32 {
        let board = self.nodes[idx].board;
        let is_black = self.nodes[idx].is_black;

        let legal = board.legal_moves(is_black);

        // ── no legal moves: pass or terminal ─────────────────────────────────
        if legal == 0 {
            let opp_legal = board.legal_moves(!is_black);
            if opp_legal == 0 {
                // Terminal position: count discs
                let b = board.black.count_ones() as i32;
                let w = board.white.count_ones() as i32;
                let outcome: f32 = match b.cmp(&w) {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Less => -1.0,
                    std::cmp::Ordering::Equal => 0.0,
                };
                self.nodes[idx].expanded = true;
                return if is_black { outcome } else { -outcome };
            }

            // Forced pass: evaluate from opponent's perspective, add pass child.
            let (_, opp_value) = eval.evaluate(board, !is_black, opp_legal);
            let child_idx = self.nodes.len();
            self.nodes.push(Node {
                board,
                is_black: !is_black,
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
            // Value from current player's perspective = -opponent's value
            return -opp_value;
        }

        // ── normal expansion ──────────────────────────────────────────────────
        let (policy, value) = eval.evaluate(board, is_black, legal);

        // Collect moves before mutating self.nodes to satisfy the borrow checker.
        let mut moves: Vec<(u64, Board, f32)> = Vec::new();
        for i in 0..64_usize {
            if (legal >> i) & 1 == 1 {
                let m = 1u64 << i;
                moves.push((m, board.apply_move(is_black, m), policy[i]));
            }
        }

        let base = self.nodes.len();
        for (offset, (move_mask, next_board, prior)) in moves.iter().enumerate() {
            let child_idx = base + offset;
            self.nodes.push(Node {
                board: *next_board,
                is_black: !is_black,
                move_mask: *move_mask,
                p: *prior,
                n: 0,
                w: 0.0,
                parent: Some(idx),
                children: Vec::new(),
                expanded: false,
            });
            self.nodes[idx].children.push(child_idx);
        }
        self.nodes[idx].expanded = true;

        value
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
