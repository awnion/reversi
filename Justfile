_weights_base := justfile_directory() / "weights"
_bots := "alphazero zerorev"

# Install JS deps
install:
    bun install

# Download the Playwright browsers used by the e2e tests
install-browsers:
    bun run install:browsers

# Build the WASM bots with Bazel and copy them into packages/bot-*/wasm
wasm:
    bun run build:wasm

# Vite dev server with HMR (rebuilds the WASM bots first)
dev:
    bun run dev

# Production bundle -> apps/web/dist
build:
    bun run build

# Rust tests (Bazel) + TS unit tests (bun)
test:
    bazel test //...
    bun run test:unit

# Everything, including e2e tests (needs `just install-browsers`)
test-all:
    bazel test //...
    bun run test:all

# Run all formatters with auto-fix
fmt:
    uvx ruff format .
    cargo fmt --all
    bunx biome check --write
    uvx taplo fmt

# Run all linters (no changes)
lint:
    uvx ruff check .
    bazel build --config=clippy //rust/...
    bunx biome check
    bunx tsc -b --pretty

# Run all linters with auto-fix
lint-fix:
    uvx ruff check --fix .
    cargo clippy --all-targets --fix --allow-dirty
    bunx biome check --write
    bunx tsc -b --pretty

# Start training in background: just train <botname>
train botname:
    #!/usr/bin/env bash
    mkdir -p {{ _weights_base / botname }}
    nohup uv run --directory python python -m train.loop --run-name {{ botname }} --selfplay-processes 1 \
        >> {{ _weights_base / botname / "train.log" }} 2>&1 &
    echo $! > {{ _weights_base / botname / "nohup.pid" }}
    echo -e "\033[32mTraining '{{ botname }}' started (pid $!)\033[0m"
    echo -e "  logs: \033[36mjust logs {{ botname }}\033[0m"

# Kill training immediately (SIGKILL): just kill <botname>
kill botname:
    #!/usr/bin/env bash
    pid_file="{{ _weights_base / botname / "train.pid" }}"
    nohup_pid="{{ _weights_base / botname / "nohup.pid" }}"
    if [ -f "$pid_file" ]; then
        kill -9 "$(cat "$pid_file")" 2>/dev/null && echo "Killed learner ($(cat $pid_file))" || true
    fi
    if [ -f "$nohup_pid" ]; then
        kill -9 "$(cat "$nohup_pid")" 2>/dev/null || true
        rm -f "$nohup_pid"
    fi
    pkill -9 -f "train.loop --run-name {{ botname }}" || true

# Stop background training: just stop <botname>
stop botname:
    #!/usr/bin/env bash
    pid_file="{{ _weights_base / botname / "train.pid" }}"
    nohup_pid="{{ _weights_base / botname / "nohup.pid" }}"
    if [ -f "$pid_file" ]; then
        kill "$(cat "$pid_file")" 2>/dev/null && echo "Sent SIGTERM to learner ($(cat $pid_file))" || true
    fi
    if [ -f "$nohup_pid" ]; then
        kill "$(cat "$nohup_pid")" 2>/dev/null || true
        rm -f "$nohup_pid"
    fi
    pkill -f "train.loop --run-name {{ botname }}" || true

# Follow training logs: just logs <botname>
logs botname:
    tail -f {{ _weights_base / botname / "train.log" }}

# Run a tournament for a bot: just tournament <botname>
tournament botname:
    uv run --directory python python -m train.tournament --run-name {{ botname }}

# --- per-bot shortcuts (for tab completion) ---
train-alphazero:  (train "alphazero")
train-zerorev:    (train "zerorev")
stop-alphazero:   (stop "alphazero")
stop-zerorev:     (stop "zerorev")
kill-alphazero:   (kill "alphazero")
kill-zerorev:     (kill "zerorev")
logs-alphazero:   (logs "alphazero")
logs-zerorev:     (logs "zerorev")
tournament-alphazero: (tournament "alphazero")
tournament-zerorev:   (tournament "zerorev")
