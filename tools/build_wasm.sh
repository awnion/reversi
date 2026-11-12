#!/usr/bin/env bash
# Builds the WASM bots with Bazel and copies the wasm-bindgen output into
# packages/bot-<bot>/wasm/, where the TypeScript wrappers import it from.
set -euo pipefail
cd "$(dirname "$0")/.."

bots=(minimax alphazero zerorev)

targets=()
for bot in "${bots[@]}"; do targets+=("//rust/$bot:wasm"); done
bazel build "${targets[@]}"

bin="$(bazel info bazel-bin)"
for bot in "${bots[@]}"; do
  out="packages/bot-$bot/wasm"
  rm -rf "$out"
  mkdir -p "$out"
  cp -R "$bin/rust/$bot/wasm/." "$out/"
  # Bazel outputs are read-only and executable; normalise the copies.
  chmod -R u+w "$out"
  find "$out" -type f -exec chmod 644 {} +
done
