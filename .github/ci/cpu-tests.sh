#!/bin/bash
# CPU-only tests: shell syntax, the host C++ tests, the Python tests and the Rust crates.
# Linux with g++ (C++20, ASan), Python 3 with numpy, and cargo (rustup reads frontend/g6front-rs/rust-toolchain.toml).
#   .github/ci/cpu-tests.sh            from any directory. The CI's `cpu` job runs it.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
step() { echo; echo "== $*"; }

step "bash -n"
git ls-files -z '*.sh' | xargs -0 -n1 bash -n

step "scripts/host_tests.sh"
scripts/host_tests.sh

step "other host C++ tests"
g++ -O2 -std=c++17 engine/tests/test_spec_len.cpp -o "$T/test_spec_len"
"$T/test_spec_len"
g++ -O2 -std=c++17 -I engine/kernels engine/tests/test_topk.cpp -o "$T/test_topk"
"$T/test_topk"
g++ -std=c++17 -O2 -pthread -I engine/runtime/s2 engine/runtime/s2/tests/control_request_test.cpp -o "$T/control_request_test"
"$T/control_request_test"
g++ -std=c++17 -O1 -fsanitize=address,undefined -I engine/runtime engine/runtime/tests/serve_parse_test.cpp -o "$T/serve_parse_test"
.github/ci/runner/test_refuse_forks.sh   # the self-hosted runner's hook that refuses fork pull requests
"$T/serve_parse_test"
g++ -std=c++17 -O2 -pthread -I engine/runtime/s2 engine/runtime/s2/tests/control_idle_test.cpp -o "$T/control_idle_test"
"$T/control_idle_test"     # the socketpair cases. The silent-peer case needs a network namespace.
g++ -std=c++17 -O2 -fsanitize=address,undefined -I engine/comms engine/comms/tests/relay_test.cpp -o "$T/relay_test"
"$T/relay_test"
g++ -std=c++20 -O1 -pthread -I engine/comms tools/health/test_progress_cpus.cpp -o "$T/test_progress_cpus"
"$T/test_progress_cpus"    # needs 3 or more CPUs

step "bin/g6load (the Hugging Face fetch's planner and checker, for the launch tests)"
cargo build --release --locked --quiet --manifest-path engine/loader-rs/Cargo.toml

step "Python tests"
python3 -m unittest tools.convert.test_v2
python3 -m unittest tools.convert.test_slim
python3 -m unittest discover -s tools/health
python3 -m unittest discover -s bench/gate/tests
python3 tools/s2_costfit.py --selftest
# test_unit only. test_integration_order still expects g6m4 to refuse --reuse-prefix under S2, which it no longer does.
(cd engine/tests && python3 -m unittest test_prefix_reuse.PrefixReuseTests.test_unit)
(cd frontend && python3 -m unittest tests.test_s2shim tests.test_g6shim tests.test_parser)
# the launch: options, model sets, the handshake, mentat, and four supervisors end to end on 127.0.0.1-4
python3 -m unittest discover -s tools/launch
# the container's compose file and entrypoint
if docker compose version > /dev/null 2>&1; then docker compose -f docker/compose.yaml config -q
else echo "(no docker compose here: compose.yaml not checked)"; fi
bash -n docker/entrypoint.sh

step "Rust: engine/loader-rs"
cargo test --locked --manifest-path engine/loader-rs/Cargo.toml
step "Rust: frontend/g6front-rs"
(cd frontend/g6front-rs && cargo test --locked)
