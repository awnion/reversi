"""Training hyper-parameters shared by all backends."""

BASE_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
# Policy targets are mixed with a uniform distribution to avoid memorising
# peaked MCTS visit counts.
LABEL_SMOOTHING = 0.03
