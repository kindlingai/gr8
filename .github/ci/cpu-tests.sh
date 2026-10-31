#!/bin/bash
# CPU-only tests: shell syntax, the host C++ tests, the Python tests and the Rust crates.
# Linux with g++ (C++20, ASan), Python 3 with numpy, and cargo (rustup reads frontend/g6front-rs/rust-toolchain.toml).
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

if [ "$PART" != frontend ]; then
  step "scripts/host_tests.sh"
  scripts/host_tests.sh

  step "Python tests (engine)"
  # Artifact tools and the C++ loader must agree on the same materialized profiles.
  g++ -std=c++17 -O2 -Wall -Wextra -pthread -I engine/runtime \
    engine/runtime/tests/steering_profile_check.cpp engine/loader/g6pack.cpp -o "$T/steering_profile_check"
  STEERING_PROFILE_CHECK="$T/steering_profile_check" python3 -m unittest discover -s tools/steering/tests -v
  python3 -m unittest tools.convert.test_v2
  python3 -m unittest discover -s engine/ref -p 'test_*.py'
  python3 -m unittest discover -s tools/health
  python3 -m unittest discover -s bench/gate/tests
  python3 -m unittest discover -s tools/ngram/tests
  python3 -m unittest discover -s tools/query
  (cd engine/tests && python3 -m unittest test_prefix_reuse)
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
  step "Rust: engine/kvman-rs"
  cargo test --locked --manifest-path engine/kvman-rs/Cargo.toml
  step "Rust: engine/knobs-rs"
  cargo test --locked --manifest-path engine/knobs-rs/Cargo.toml
fi

[ "$PART" = engine ] && exit 0

step "bin/g6load (the Hugging Face fetch's planner and checker, for the launch tests)"
cargo build --release --locked --quiet --manifest-path engine/loader-rs/Cargo.toml

if [ -n "${GR8_PSEUDO_TOKENIZER:-}" ]; then
  step "the pseudo engine: the smoke test and the front end's API tests (release builds, for the launcher's tests too)"
  frontend/pseudo-engine/smoke.sh "$GR8_PSEUDO_TOKENIZER"
  frontend/tests/api.sh "$GR8_PSEUDO_TOKENIZER"
else
  echo "(no GR8_PSEUDO_TOKENIZER: no pseudo engine smoke or API tests)"
fi

step "Python tests (front end and launcher)"
# the S2 headers read knobs: link the knob table's C API
cargo build --release --locked --quiet --features capi --manifest-path engine/knobs-rs/Cargo.toml
KNOBS=(engine/knobs-rs/target/release/libgr8knobs.a -lpthread -ldl -lm)
g++ -std=c++17 -O2 -Wall -Wextra -I engine/runtime \
  engine/runtime/s2/tests/steering_transport_test.cpp -o "$T/steering_transport_test" "${KNOBS[@]}"
(cd frontend && STEERING_TRANSPORT_CHECK="$T/steering_transport_test" python3 -m unittest tests.test_s2shim tests.test_steering)
# the launch: options, model sets, the handshake, mentat, and four supervisors end to end on 127.0.0.1-4
g++ -std=c++17 -O2 -pthread -I engine/runtime engine/runtime/tests/steering_boot_check.cpp engine/loader/g6pack.cpp -o "$T/steering_boot_check" \
  "${KNOBS[@]}"
STEERING_BOOT_CHECK="$T/steering_boot_check" python3 -m unittest discover -s tools/launch

step "Rust: frontend/g6front-rs"
(cd frontend/g6front-rs && cargo test --locked)
step "Rust: frontend/pseudo-engine (and its weights compile)"
(cd frontend/pseudo-engine && cargo test --locked && cargo run --locked --quiet -- --weights model/responses.toml --check)
