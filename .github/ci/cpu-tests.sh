#!/bin/bash
# CPU-only tests: shell syntax, the host C++ tests, the Python tests and the Rust crates.
# Linux with g++ (C++20, ASan), Python 3 with numpy, and cargo. Engine cargo uses --manifest-path from
# the repo root, so rustup reads rust-toolchain.toml there. The front end cds into its crate and reads that pin.
# Keep those channels equal to Dockerfile ARG RUST_VERSION.
#   .github/ci/cpu-tests.sh [frontend|engine|all]     from any directory; all by default. The CI's cpu-frontend and
#                                                    cpu-engine jobs run one half each.
# frontend: the front end, the shim, the launcher and the pseudo engine, with the small engine-side checkers their tests
# drive; with GR8_PSEUDO_TOKENIZER (.github/ci/fetch-tokenizer.sh DIR) also the smoke test, the front end's API tests
# and ./gr8 pseudo up end to end, against the pseudo engine. It runs on Linux and macOS. engine: everything else.
set -euo pipefail
PART=${1:-all}
case $PART in frontend|engine|all) ;; *) echo "usage: cpu-tests.sh [frontend|engine|all]" >&2; exit 2 ;; esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
step() { echo; echo "== $*"; }

step "bash -n"
git ls-files -z '*.sh' | xargs -0 -n1 bash -n

# Print compiler versions up front (serially) so rustup never races auto-installing the pinned toolchain in parallel cargo jobs.
step "Rust compiler"
rustc -V
cargo -V

if [ "$PART" != frontend ]; then
  step "scripts/host_tests.sh"
  scripts/host_tests.sh

  step "Python tests (engine)"
  # Artifact tools and the C++ loader must agree on the same materialized profiles.
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I engine/runtime \
    engine/runtime/tests/steering_profile_check.cpp engine/loader/g6pack.cpp -o "$T/steering_profile_check"
  # Run independent Python engine test suites in parallel.
  PYL="$T/py_engine"; mkdir -p "$PYL"
  PY_NAMES=()
  py_job() {
    local name=$1; shift
    PY_NAMES+=("$name")
    ( "$@" ) > "$PYL/$name.log" 2>&1 || touch "$PYL/$name.failed" &
  }
  py_job steering   env STEERING_PROFILE_CHECK="$T/steering_profile_check" python3 -m unittest discover -s tools/steering/tests -v
  py_job convert    python3 -m unittest tools.convert.test_v2
  py_job ref        python3 -m unittest discover -s engine/ref -p 'test_*.py'
  py_job health     python3 -m unittest discover -s tools/health
  py_job bench      python3 -m unittest discover -s bench/tests
  py_job ngram      python3 -m unittest discover -s tools/ngram/tests
  py_job query      python3 -m unittest discover -s tools/query
  py_job eng_py     bash -c 'cd engine/tests && python3 -m unittest test_prefix_reuse test_cp_pf_ix_order'
  wait
  for name in "${PY_NAMES[@]}"; do
    cat "$PYL/$name.log"
  done
  if ls "$PYL"/*.failed > /dev/null 2>&1; then
    echo "Python tests (engine) failed: $(cd "$PYL" && ls -- *.failed | sed 's/\.failed$//' | tr '\n' ' ')" >&2
    exit 1
  fi
  # the container's compose file and entrypoint
  if docker compose version > /dev/null 2>&1; then docker compose -f docker/compose.yaml config -q
  else echo "(no docker compose here: compose.yaml not checked)"; fi
  bash -n docker/entrypoint.sh
  # the Helm chart: it renders with the model named, and refuses to without it
  if command -v helm > /dev/null; then
    model=(--set options.GR8_MODEL=glm53 --set options.MODEL=https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4)
    helm lint --quiet deploy/k8s "${model[@]}" && helm template gr8 deploy/k8s "${model[@]}" > /dev/null
    ! helm template gr8 deploy/k8s > /dev/null 2>&1 || { echo "deploy/k8s rendered without GR8_MODEL"; exit 1; }
  else echo "(no helm here: deploy/k8s not checked)"; fi

  step "Rust: engine/loader-rs"
  cargo test --locked --manifest-path engine/loader-rs/Cargo.toml
  step "Rust: engine/knobs-rs"
  cargo test --locked --manifest-path engine/knobs-rs/Cargo.toml
  step "Rust: the engine workspace (engine/kvman-rs, engine/scheduler-rs, engine/hostcore-rs)"
  cargo test --locked --workspace --manifest-path engine/Cargo.toml
fi

[ "$PART" = engine ] && exit 0

step "bin/g6load and host/knobs libs (for the launch and S2 shim tests)"
cargo build --release --locked --quiet --manifest-path engine/loader-rs/Cargo.toml & P1=$!
cargo build --release --locked --quiet --manifest-path engine/Cargo.toml & P2=$!
(
  cargo build --release --locked --quiet --features capi --manifest-path engine/knobs-rs/Cargo.toml
  cargo rustc --release --locked --quiet --features capi --lib --crate-type cdylib --manifest-path engine/knobs-rs/Cargo.toml
) & P3=$!
wait "$P1"; wait "$P2"; wait "$P3"
KNOBS_LIB="$REPO/engine/knobs-rs/target/release/libgr8knobs.so"
[ "$(uname -s)" = Darwin ] && KNOBS_LIB="$REPO/engine/knobs-rs/target/release/libgr8knobs.dylib"
# Export GR8_KNOBS_LIB so none of the hundreds of launch.py subprocesses re-check mtime or invoke cargo.
export GR8_KNOBS_LIB="$KNOBS_LIB"

if [ -n "${GR8_PSEUDO_TOKENIZER:-}" ]; then
  step "the pseudo engine: the smoke test and the front end's API tests (release builds, for the launcher's tests too)"
  frontend/pseudo-engine/smoke.sh "$GR8_PSEUDO_TOKENIZER"
  frontend/tests/api.sh "$GR8_PSEUDO_TOKENIZER"
else
  echo "(no GR8_PSEUDO_TOKENIZER: no pseudo engine smoke or API tests)"
fi

step "Python tests (front end and launcher)"
KNOBS=(engine/knobs-rs/target/release/libgr8knobs.a engine/target/release/libg6host.a -lpthread -ldl -lm)
g++ -std=c++17 -O2 -Wall -Wextra -I engine/runtime \
  engine/runtime/s2/tests/steering_transport_test.cpp -o "$T/steering_transport_test" "${KNOBS[@]}" & C1=$!
g++ -std=c++17 -O2 -pthread -I engine/runtime engine/runtime/tests/steering_boot_check.cpp engine/loader/g6pack.cpp -o "$T/steering_boot_check" \
  "${KNOBS[@]}" & C2=$!
python3 -m unittest discover -s tools/switch/tests -v
wait "$C1"; wait "$C2"
(cd frontend && STEERING_TRANSPORT_CHECK="$T/steering_transport_test" python3 -m unittest tests.test_s2shim tests.test_steering)
# the launch: options, model sets, the handshake, mentat, and four supervisors end to end on 127.0.0.1-4.
# Modules that spawn g6m4/g6pseudo or run `launch.py up` scan /proc for running engines, so run that group serially
# on one dedicated lane while the remaining independent launch modules run in parallel beside it.
STEERING_BOOT_CHECK="$T/steering_boot_check" python3 - <<'PY'
import concurrent.futures, glob, os, subprocess, sys, time

all_mods = sorted(os.path.basename(p)[:-3] for p in glob.glob('tools/launch/test_*.py'))
# Serial lane: modules that start live engine/supervisor processes (running_engines() scans /proc).
serial_order = ['test_scale', 'test_e2e', 'test_pseudo_e2e', 'test_kvnode']
serial_mods = [m for m in serial_order if m in all_mods]
parallel_mods = [m for m in all_mods if m not in serial_mods]
# Schedule heaviest parallel modules first (LPT).
priority = ['test_hf', 'test_options', 'test_supervisor', 'test_handshake', 'test_models', 'test_mentat']
parallel_mods.sort(key=lambda m: (priority.index(m) if m in priority else len(priority), m))
workers = max(2, min(4, os.cpu_count() or 1))

def run_mod(mod):
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, '-m', 'unittest', mod], cwd='tools/launch',
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return mod, r.returncode, time.monotonic() - t0, r.stdout

def run_serial(mods):
    return [run_mod(m) for m in mods]

t_all = time.monotonic()
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
    serial_fut = ex.submit(run_serial, serial_mods)
    par_futs = [ex.submit(run_mod, m) for m in parallel_mods]
    for fut in par_futs:
        results.append(fut.result())
    results.extend(serial_fut.result())

failed = []
for mod, rc, dt, out in results:
    summary = next((l for l in reversed(out.splitlines()) if l.strip()), '')
    print(f'tools/launch/{mod}.py ({dt:.1f}s): {summary}', flush=True)
    if rc != 0:
        failed.append((mod, out))
print(f'tools/launch: {len(all_mods)} modules in {time.monotonic() - t_all:.1f}s ({workers} workers, {len(serial_mods)} serial)', flush=True)
if failed:
    for mod, out in failed:
        print(f'\n=== FAILED: tools/launch/{mod}.py ===\n{out}', file=sys.stderr)
    sys.exit(1)
PY

step "Rust: frontend/g6front-rs"
(cd frontend/g6front-rs && cargo test --locked)
cargo build --release --locked --manifest-path frontend/g6front-rs/Cargo.toml
python3 -m unittest discover -s frontend/tests -p test_ds41_api.py
step "Rust: frontend/pseudo-engine (and its weights compile)"
(cd frontend/pseudo-engine && cargo test --locked && cargo run --locked --quiet -- --weights model/responses.toml --check)
