# alphazero

"AlphaZero" is only the name this run was given. It is not DeepMind's
AlphaZero and has nothing to do with its weights: it is this project's own
small network (4 residual blocks of 32 channels), trained by self-play in the
AlphaZero style.

The weights date from March 2026 (`champion.json`: 25 March 2026, MLX), when
the first single-threaded prototype of the engine and the training loop was
written. ZeroRev and the later runs replaced it.

It is kept for history. `rust/alphazero` and `packages/bot-alphazero` still
embed and wrap `champion.bin`, but the web app no longer offers the bot: the
name suggested something it is not.
