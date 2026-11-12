#!/usr/bin/env bash
# Builds the WASM bots and puts the wasm-bindgen output where the TypeScript
# wrappers import it from:
#
#   packages/bot-<bot>/wasm/          single-threaded, built by Bazel
#   packages/bot-<bot>/wasm-threads/  multi-threaded search
#
# The threaded build shares one search tree between Web Workers. That needs
# the standard library rebuilt with atomics, which Bazel's rules_rust cannot
# do, so this part goes through cargo with a nightly toolchain:
#
#   rustup toolchain install nightly --component rust-src \
#       --target wasm32-unknown-unknown
#
# REVERSI_NIGHTLY selects the toolchain (default: nightly); CI pins a date.
set -euo pipefail
cd "$(dirname "$0")/.."

bots=(minimax alphazero zerorev)
threaded_bots=(minimax alphazero zerorev)

# ── single-threaded: Bazel ──────────────────────────────────────────────────
targets=(@rules_rust_wasm_bindgen//3rdparty:wasm_bindgen_cli)
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

# ── multi-threaded: cargo nightly + the same wasm-bindgen CLI ───────────────
# The CLI must match the wasm-bindgen crate version exactly, so the one Bazel
# built is reused.
wasm_bindgen="$PWD/$(bazel cquery --output=files \
  @rules_rust_wasm_bindgen//3rdparty:wasm_bindgen_cli 2>/dev/null | head -1)"
target_dir="$PWD/.cargo-target/wasm-threads"
build_date="$(date '+%Y-%m-%d')"

# Shared, imported memory and the thread-local-storage symbols that
# wasm-bindgen needs to start threads.
threads_flags=(
  -C target-feature=+atomics,+bulk-memory,+simd128
  -C link-arg=--shared-memory
  -C link-arg=--import-memory
  -C link-arg=--max-memory=1073741824
  -C link-arg=--export=__wasm_init_tls
  -C link-arg=--export=__tls_size
  -C link-arg=--export=__tls_align
  -C link-arg=--export=__tls_base
  -C link-arg=--export=__heap_base
)

packages=()
for bot in "${threaded_bots[@]}"; do packages+=(-p "reversi_$bot"); done
BUILD_DATE="$build_date" RUSTFLAGS="${threads_flags[*]}" \
  cargo "+${REVERSI_NIGHTLY:-nightly}" build --release "${packages[@]}" --features threads \
  --target wasm32-unknown-unknown -Z build-std=panic_abort,std \
  --target-dir "$target_dir"

for bot in "${threaded_bots[@]}"; do
  out="packages/bot-$bot/wasm-threads"
  rm -rf "$out"
  mkdir -p "$out"
  "$wasm_bindgen" \
    "$target_dir/wasm32-unknown-unknown/release/reversi_$bot.wasm" \
    --target web --out-dir "$out" --out-name "reversi_$bot"
  # The thread helper imports "the package" (`../../..`), which only resolves
  # in a wasm-pack layout; point it at the generated module instead.
  helper="$(find "$out/snippets" -name workerHelpers.js)"
  sed -i.bak "s#import('../../..')#import('../../../reversi_$bot.js')#" "$helper"
  rm -f "$helper.bak"
  grep -q "reversi_$bot.js" "$helper"
  # Under cross-origin isolation Safari refuses a worker whose script it
  # serves from its cache, so every thread after the first would fail to
  # start. Give each thread its own script URL; the name stays `Worker` so
  # Vite still recognises `new Worker(new URL(...))` and bundles the helper.
  cat >>"$helper" <<'JS'

const Worker = class extends globalThis.Worker {
  constructor(url, options) {
    const unique = new URL(url);
    unique.searchParams.set('thread', crypto.randomUUID());
    super(unique, options);
  }
};
JS
done
