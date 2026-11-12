"""Training hyper-parameters shared by all backends."""

BASE_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
# Policy targets are mixed with a uniform distribution to avoid memorising
# peaked MCTS visit counts.
LABEL_SMOOTHING = 0.03
# Weight of the auxiliary loss on the final disc margin (from this player's
# side, divided by 64). It gives the value head a richer signal than the bare
# win/loss.
MARGIN_LOSS_WEIGHT = 1.0
