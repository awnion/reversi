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
train botname processes="":
    #!/usr/bin/env bash
    mkdir -p {{ _weights_base / botname }}
    # Self-play processes: 4 on CUDA, 1 on Apple Silicon unless given.
    nohup uv run --directory python python -m train.loop --run-name {{ botname }} \
        {{ if processes != "" { "--selfplay-processes " + processes } else { "" } }} \
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
    # The Rust self-play process of this run only (REVERSI_SELFPLAY=rust).
    rs_pid_file="{{ _weights_base / botname / "selfplay-rs.pid" }}"
    rs_pid="$(cat "$rs_pid_file" 2>/dev/null)"
    if [ -n "$rs_pid" ] && ps -p "$rs_pid" -o args= | grep -q reversi-selfplay; then
        kill -9 "$rs_pid" 2>/dev/null || true
    fi
    rm -f "$rs_pid_file"
    pkill -9 -f "reversi-selfplay selfplay --run-dir {{ _weights_base / botname }} " || true

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
    # The learner saves its replay buffer and stops its Rust self-play child
    # (REVERSI_SELFPLAY=rust) on the way out; wait for that, so that a
    # following `just train` does not find it still running.
    rs_pattern="reversi-selfplay selfplay --run-dir {{ _weights_base / botname }} "
    # Saving a full replay buffer can take more than a minute.
    for _ in $(seq 300); do
        pgrep -f "train.loop --run-name {{ botname }}" >/dev/null || break
        sleep 1
    done
    if pgrep -f "train.loop --run-name {{ botname }}" >/dev/null; then
        echo "Learner still running after 300 s" >&2
        exit 1
    fi
    # Only then an orphaned Rust process, with a single signal: a second
    # SIGTERM makes it exit at once without flushing its last shard.
    rs_pid="$(cat "{{ _weights_base / botname / "selfplay-rs.pid" }}" 2>/dev/null)"
    if [ -n "$rs_pid" ] && ps -p "$rs_pid" -o args= | grep -q reversi-selfplay; then
        kill "$rs_pid" 2>/dev/null || true
    else
        pkill -f "$rs_pattern" || true
    fi
    for _ in $(seq 10); do
        pgrep -f "$rs_pattern" >/dev/null || break
        sleep 1
    done

# Links the libtorch of python/.venv (CUDA machine; torch must match tch, see
# rust/trainer/Cargo.toml). Niced and on 4 jobs, so that it can build next to a
# running training.
# Build the Rust self-play binary (REVERSI_SELFPLAY=rust): just build-selfplay
build-selfplay:
    cd python && LIBTORCH_USE_PYTORCH=1 nice -n 19 uv run --no-sync cargo build --release -j 4 \
        --manifest-path ../rust/trainer/Cargo.toml --target-dir ../rust/trainer/target

# Games are discarded; extra args go to the binary, e.g. --workers 8.
# Rust self-play throughput: just bench-selfplay <weights.bin> [seconds] [args]
bench-selfplay weights seconds="60" *args="":
    rust/trainer/target/release/reversi-selfplay bench --weights {{ weights }} --seconds {{ seconds }} {{ args }}

# Follow training logs: just logs <botname>
logs botname:
    tail -f {{ _weights_base / botname / "train.log" }}

# Run a tournament for a bot: just tournament <botname>
tournament botname:
    uv run --directory python python -m train.tournament --run-name {{ botname }}

# --- training on another machine -------------------------------------------
# Runs are interchangeable between machines: checkpoints are framework-neutral
# (MLX on Apple Silicon, PyTorch on Linux/CUDA).

# The ssh host of the training machine. Away from its network use a host that
# jumps through a relay, e.g. `REVERSI_TRAIN_HOST=ku-jump just snapshot`.
_train_host := env_var_or_default("REVERSI_TRAIN_HOST", "ku")
_remote_dir := "gh/awnion/reversi"
_remote_path := "export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Copy the code (not the weights) to a remote checkout: just sync-code <host>
sync-code host:
    rsync -az --exclude .git --exclude node_modules --exclude 'bazel-*' \
        --exclude .venv --exclude target --exclude __pycache__ --exclude .ruff_cache \
        --exclude 'python/dist' --exclude 'packages/*/wasm' --exclude 'apps/web/dist' \
        --exclude '*.tsbuildinfo' --exclude weights \
        ./ {{ host }}:{{ _remote_dir }}/

# Install deps and build the Rust extension on the remote: just setup-remote <host>
setup-remote host: (sync-code host)
    ssh {{ host }} '{{ _remote_path }}; cd {{ _remote_dir }}/python && uv sync && uv run maturin develop --release --features python'
    ssh {{ host }} '{{ _remote_path }}; cd {{ _remote_dir }} && just build-selfplay' \
        || echo "reversi-selfplay not built; Python self-play still works"

# Send a run's state (champion, checkpoints, replay buffer): just push-run <host> <botname>
push-run host botname:
    rsync -az --exclude replay_spool --exclude '*.pid' --exclude train.log --exclude old_checkpoints \
        {{ _weights_base / botname }}/ {{ host }}:{{ _remote_dir }}/weights/{{ botname }}/

# Fetch a run's state back: just pull-run <host> <botname>
pull-run host botname:
    rsync -az --exclude replay_spool --exclude '*.pid' --exclude train.log --exclude old_checkpoints \
        {{ host }}:{{ _remote_dir }}/weights/{{ botname }}/ {{ _weights_base / botname }}/

# Start / stop / follow training on the remote: just train-remote <host> <botname>
train-remote host botname: (sync-code host)
    ssh {{ host }} '{{ _remote_path }}; cd {{ _remote_dir }} && just train {{ botname }}'

stop-remote host botname:
    ssh {{ host }} '{{ _remote_path }}; cd {{ _remote_dir }} && just stop {{ botname }}'

logs-remote host botname:
    ssh {{ host }} 'tail -f {{ _remote_dir }}/weights/{{ botname }}/train.log'

# Download the newest champion from the training machine into a browser bot
# and rebuild the WASM. With no arguments: zerorev-v2 from the training host
# into ZeroRev.
#   just snapshot [host] [run] [bot] [champion|latest]
# "champion" is the last tournament winner, "latest" the newest trained weights.
snapshot host=_train_host run="zerorev-v2" bot="zerorev" which="champion":
    #!/usr/bin/env bash
    set -euo pipefail
    remote={{ host }}:{{ _remote_dir }}/weights/{{ run }}
    local={{ _weights_base / run }}
    target={{ _weights_base / bot }}
    mkdir -p "$local"
    if [ "{{ which }}" = "latest" ]; then
        rsync -az "$remote/selfplay_latest.bin" "$local/selfplay_latest.bin"
        cp "$local/selfplay_latest.bin" "$target/champion.bin"
        # The newest weights have not been through a tournament yet.
        printf '{\n  "run": "%s",\n  "date": "%s",\n  "loss": null,\n  "win_rate": null\n}\n' \
            "{{ run }}" "$(date +%F)" > "$target/champion.json"
    else
        rsync -az "$remote/champion.bin" "$remote/champion.json" "$local/"
        cp "$local/champion.bin" "$local/champion.json" "$target/"
    fi
    bun run build:wasm >/dev/null 2>&1
    echo -e "\033[32m{{ bot }} now plays the {{ which }} weights of '{{ run }}' from {{ host }}:\033[0m"
    tr -d '\n' < "$target/champion.json"; echo
    echo "Reload the page to play it; \`just vs-minimax\` checks it against Minimax."

# Play ZeroRev against Minimax with each colour: just vs-minimax [ms per move]
vs-minimax ms="300":
    bun apps/web/scripts/vs-minimax.ts {{ ms }}

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
