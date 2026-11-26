# gr8 for Blackwell GPUs: GB10 (DGX Spark / ASUS GX10: linux/arm64, CUDA 13, sm_121a) and RTX 5090 / RTX PRO 6000
# (linux/amd64, CUDA 13, sm_120a). docker/README.md has how to run it.
#
#   docker buildx build --platform linux/arm64 -t gr8 .                         the arm64 runtime image (sm_121a)
#   docker buildx build --platform linux/amd64 -t gr8 .                         the amd64 runtime image (sm_120a)
#   docker buildx build --platform linux/arm64 --target dist -o type=local,dest=out .
#                                                                              only the binaries, into out/bin
#
# GR8_ARCH overrides the CUDA SM target (empty defaults to 121a on arm64 and 120a on amd64 via scripts/build.sh).
# Stages: toolchain (CUDA devel + rdma-core) -> rust -> loader (the Rust loader), kvman (the host Rust: page allocator,
# scheduler, planner), knobs and front (the Rust front end) -> rust-dist -> engine, engine-int4, engine-flash,
# engine-flash-int4, engine-ds41, engine-ink2, engine-qw38 and engine-tests-ship -> dist (the binaries only)
# -> runtime (CUDA runtime + binaries, scripts, shims).
ARG CUDA_VERSION=13.0.1
ARG UBUNTU=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS toolchain
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# binutils for build.sh's `strings` check, curl for rustup
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ binutils libibverbs-dev ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

FROM toolchain AS rust
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Pins: rust-toolchain.toml, engine/rust-toolchain.toml, frontend/g6front-rs/rust-toolchain.toml,
# frontend/pseudo-engine/rust-toolchain.toml. Keep RUST_VERSION equal to those pins.
ARG RUST_VERSION=1.94.1
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "$RUST_VERSION"
WORKDIR /src

# Each Rust build copies its output out and drops target/ in the same layer, so the cached layer holds one file.
FROM rust AS loader
COPY engine/knobs-rs engine/knobs-rs
COPY knobs knobs
COPY engine/loader-rs engine/loader-rs
RUN mkdir -p /out \
 && cargo build --release --locked --manifest-path engine/loader-rs/Cargo.toml \
 && cp engine/loader-rs/target/release/libg6load.a engine/loader-rs/target/release/g6load engine/loader-rs/target/release/g6layout /out/ \
 && rm -rf engine/loader-rs/target

FROM rust AS kvman
COPY engine/knobs-rs engine/knobs-rs
COPY knobs knobs
COPY engine/Cargo.toml engine/Cargo.lock engine/
COPY engine/hostcore-rs engine/hostcore-rs
COPY engine/kvman-rs engine/kvman-rs
COPY engine/scheduler-rs engine/scheduler-rs
RUN mkdir -p /out \
 && cargo build --release --locked --manifest-path engine/Cargo.toml \
 && cp engine/target/release/libg6host.a engine/target/release/kvman-remote engine/target/release/kvman-peer /out/ \
 && rm -rf engine/target

FROM rust AS knobs
COPY engine/knobs-rs engine/knobs-rs
COPY knobs knobs
RUN mkdir -p /out \
 && cargo build --release --locked --features capi --manifest-path engine/knobs-rs/Cargo.toml \
 && cargo rustc --release --locked --features capi --lib --crate-type cdylib --manifest-path engine/knobs-rs/Cargo.toml \
 && cp engine/knobs-rs/target/release/libgr8knobs.a engine/knobs-rs/target/release/libgr8knobs.so /out/ \
 && rm -rf engine/knobs-rs/target

# the slowest step (fat LTO); built once in rust-dist and reused across CI jobs
FROM rust AS front
# the xgrammar crate builds XGrammar's C++ (json_schema grammars) with CMake and clang, and binds it through libclang
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake clang libclang-dev \
 && rm -rf /var/lib/apt/lists/*
COPY engine/knobs-rs engine/knobs-rs
COPY knobs knobs
COPY frontend/g6front-rs frontend/g6front-rs
# from the crate's directory, so rustup reads its rust-toolchain.toml
WORKDIR /src/frontend/g6front-rs
RUN mkdir -p /out \
 && cargo build --release --locked \
 && cp target/release/g6front-rs /out/ \
 && rm -rf target

# rust-dist holds all Rust outputs: static libraries under /lib for the engine stages, and shipped binaries under /bin
# for dist / runtime. In CI the rust job builds rust-dist once and passes it as a named build context to every engine
# leg (avoiding 8x duplicate Rust builds), while local builds still build rust-dist directly.
FROM scratch AS rust-dist
COPY --from=loader /out/libg6load.a /lib/
COPY --from=kvman /out/libg6host.a /lib/
COPY --from=knobs /out/libgr8knobs.a /lib/
COPY --from=loader /out/g6load /out/g6layout /bin/
COPY --from=kvman /out/kvman-remote /out/kvman-peer /bin/
COPY --from=front /out/g6front-rs /bin/
# the knob table the launcher reads (tools/launch/knobs.py)
COPY --from=knobs /out/libgr8knobs.so /bin/lib/

# XGrammar's C++ sources, the copy the front end's xgrammar crate pins (Cargo.lock): the engine's request grammars build
# from it (engine/runtime/grammar.h). cargo fetch only, so the engine stages need not wait for the front end's build
FROM rust AS xgrammar-src
COPY engine/knobs-rs engine/knobs-rs
COPY knobs knobs
COPY frontend/g6front-rs/Cargo.toml frontend/g6front-rs/Cargo.lock frontend/g6front-rs/rust-toolchain.toml frontend/g6front-rs/
WORKDIR /src/frontend/g6front-rs
RUN mkdir -p src && touch src/main.rs && cargo fetch --locked \
 && cp -a "$(ls -d /opt/cargo/registry/src/*/xgrammar-*/thirdparty/xgrammar | head -1)" /out-xgrammar

# XGrammar's library, built once for the GLM and Flash engine stages; it changes only with the pin, so engine edits keep it
# cached
FROM toolchain AS xgrammar
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake \
 && rm -rf /var/lib/apt/lists/*
COPY engine/runtime/xgrammar_build.sh /xgrammar_build.sh
COPY --from=xgrammar-src /out-xgrammar /opt/xgrammar
RUN /xgrammar_build.sh /opt/xgrammar /xb && mkdir -p /out && cp /xb/libxgrammar.a /out/ && rm -rf /xb

# engine-src copies only the non-test C++/CUDA engine sources and Rust C headers, so edits to docs, scripts, the front
# end, Python refs, or engine test files keep the 6 model engine layers cached.
# This stage has no cargo: build.sh skips its cargo step and links the libg6load.a, libg6host.a and libgr8knobs.a
# copied from rust-dist.
# The compile cache (docs/BUILD-CACHE.md) is a cache mount in each engine stage, one id a stage, which CI keeps between
# runs (ci.yml). A layer that has to run again then compiles only what its edit reaches. After the build the stage
# deletes the entries the build did not read (a hit sets its entry's time), so the cache holds one build.
FROM toolchain AS engine-src
ENV GR8_NVCC_CACHE_DIR=/var/cache/gr8-nvcc
WORKDIR /src
COPY engine/*.h engine/
COPY engine/knobs-rs/*.h engine/knobs-rs/
COPY engine/loader-rs/*.h engine/loader-rs/
COPY engine/kvman-rs/*.h engine/kvman-rs/
COPY engine/scheduler-rs/*.h engine/scheduler-rs/
COPY engine/attn/*.cu engine/attn/*.cuh engine/attn/*.h engine/attn/
COPY engine/comms/*.cpp engine/comms/*.h engine/comms/
COPY engine/engram/*.cuh engine/engram/*.h engine/engram/
COPY engine/kda/*.cuh engine/kda/*.h engine/kda/
COPY engine/kernels/*.cuh engine/kernels/*.h engine/kernels/
COPY engine/loader/*.cpp engine/loader/*.h engine/loader/
COPY engine/mhc/*.cuh engine/mhc/*.h engine/mhc/
COPY engine/runtime/*.cc engine/runtime/*.cpp engine/runtime/*.cu engine/runtime/*.cuh engine/runtime/*.h engine/runtime/*.sh engine/runtime/
COPY engine/runtime/s2/*.cuh engine/runtime/s2/*.h engine/runtime/s2/
COPY scripts/build.sh scripts/build.sh
COPY --from=rust-dist /lib/libg6load.a engine/loader-rs/target/release/libg6load.a
COPY --from=rust-dist /lib/libg6host.a engine/target/release/libg6host.a
COPY --from=rust-dist /lib/libgr8knobs.a engine/knobs-rs/target/release/libgr8knobs.a
# Default CUDA SM target comes from scripts/build.sh (121a on arm64, 120a on amd64); GR8_ARCH overrides it.
ARG GR8_ARCH=
ENV ARCH=${GR8_ARCH}

# the GLM and Flash engine stages add XGrammar (and grammar_test.cc run by build_m4.sh); the tests build without it
FROM engine-src AS engine-src-glm
COPY engine/runtime/tests/grammar_test.cc engine/runtime/tests/grammar_test.cc
COPY --from=xgrammar-src /out-xgrammar /opt/xgrammar
COPY --from=xgrammar /out/libxgrammar.a /opt/xgrammar-lib/libxgrammar.a
ENV GR8_XGRAMMAR=/opt/xgrammar GR8_XGRAMMAR_LIB=/opt/xgrammar-lib/libxgrammar.a

# one g6m4 and one m4sim for every TP width; every launcher runs this build.
# Each engine stage keeps only the binaries its dist stage copies: a cached layer is reused whole or rebuilt whole, so
# its objects and build-only binaries are dead weight in the CI cache.
# GR8_WIDTHS builds a subset (build.sh): CI passes one on a pull request and
# leaves it empty, for all seven widths (1 2 3 4 5 6 8), on main and on tags.
FROM engine-src-glm AS engine
ARG TARGETARCH
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-fp8-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" GR8_TESTS=0 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4 ! -name m4sim -exec rm -rf {} +

# the two test binaries the image ships (tr3_e4m3_test and gpuprobe), on their own CI runner: GR8_TESTS=ship builds only
# them, so the image does not wait for the other test builds.
FROM engine-src AS engine-tests-ship
ARG TARGETARCH
COPY engine/kernels/tests/tr3_e4m3_test.cu engine/kernels/tests/tr3_e4m3_test.cu
COPY tools/health/gpuprobe.cu tools/health/gpuprobe.cu
RUN --mount=type=cache,id=gr8-nvcc-tests-ship-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_TESTS=ship scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name gpuprobe ! -name tr3_e4m3_test -exec rm -rf {} +

# every test binary build.sh makes after the GLM engine, a compile check on its own CI runner (nothing here ships).
# Test sources are copied only here so test edits never invalidate the model engine stages.
FROM engine-src AS engine-tests
ARG TARGETARCH
COPY engine/tests engine/tests
COPY engine/attn/tests engine/attn/tests
COPY engine/comms/tests engine/comms/tests
COPY engine/kernels/tests engine/kernels/tests
COPY engine/runtime/tests engine/runtime/tests
COPY tools/health/gpuprobe.cu tools/health/gpuprobe.cu
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-tests-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" GR8_TESTS=only scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && rm -rf /out/bin

# GLM-5.3-Flash: g6m4_flash and m4sim_flash (MODEL=flash), TP 1, 2, 3 and 4 (or GR8_WIDTHS subset on PRs)
FROM engine-src-glm AS engine-flash
ARG TARGETARCH
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-flash-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" MODEL=flash scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_flash ! -name m4sim_flash -exec rm -rf {} +

# GLM-5.3-Flash with the 4-bit KV latent (--kv-format int4, docs/FLASH.md): g6m4_flash-int4, TP 1, 2, 3 and 4
FROM engine-src-glm AS engine-flash-int4
ARG TARGETARCH
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-flash-int4-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" MODEL=flash KV_FORMAT=int4 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_flash-int4 -exec rm -rf {} +

# DeepSeek-V4.1-Flash-1017: g6m4_ds41 and m4sim_ds41 (MODEL=ds41), TP 4. build.sh builds it without request grammars, so
# it starts from engine-src and does not wait for XGrammar. Its build knobs (GR8_DS_TRUNK_BF16) come from
# ds41_build_flags.sh, which only this stage copies, so the other engine stages keep their cache.
FROM engine-src AS engine-ds41
ARG TARGETARCH
COPY scripts/ds41_build_flags.sh scripts/ds41_build_flags.sh
RUN --mount=type=cache,id=gr8-nvcc-ds41-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && MODEL=ds41 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_ds41 ! -name m4sim_ds41 -exec rm -rf {} +

# Inkling-2-Small: g6m4_ink2 and m4sim_ink2 (MODEL=ink2), TP 1, 4 and 6, without request grammars
FROM engine-src AS engine-ink2
ARG TARGETARCH
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-ink2-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" MODEL=ink2 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_ink2 ! -name m4sim_ink2 -exec rm -rf {} +

# Qwen3.8-Flash-Next: g6m4_qw38 and m4sim_qw38 (MODEL=qw38), TP 1, 2, 3 and 4, without request grammars
FROM engine-src AS engine-qw38
ARG TARGETARCH
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-qw38-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" MODEL=qw38 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_qw38 ! -name m4sim_qw38 -exec rm -rf {} +

# the 4-bit KV latent (--kv-format int4, docs/KV4.md): g6m4 alone, built with -DG6_KV4=1, as g6m4-int4 next to the fp8
# engine, for every TP width
FROM engine-src-glm AS engine-int4
ARG TARGETARCH
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-int4-${TARGETARCH},target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" KV_FORMAT=int4 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4-int4 -exec rm -rf {} +

# dist holds the shipped binaries in bin/: g6m4 (fp8 KV) and g6m4-int4 (--kv-format int4), both for every TP width,
# g6load (the Hugging Face fetch's planner and checker), and g6layout (the width-change layout planner for gr8 scale).
# build.sh's CPU test binaries and objects stay behind.
# tr3_e4m3_test is the GPU decode check build.sh cannot run without a GPU.
# dist-rust, dist-fp8, dist-int4, dist-flash, dist-flash-int4, dist-ds41, dist-ink2 and dist-tests are the parts of
# dist. CI builds rust-dist once and each engine leg in its own job, then unpacks the artifacts into one directory.
# dist stays the union, for a local build and for anything that wants every binary from one target.
FROM scratch AS dist-rust
COPY --from=rust-dist /bin /bin

FROM scratch AS dist-fp8
COPY --from=engine /out/bin/g6m4 /out/bin/m4sim /bin/

FROM scratch AS dist-int4
COPY --from=engine-int4 /out/bin/g6m4-int4 /bin/

FROM scratch AS dist-flash
COPY --from=engine-flash /out/bin/g6m4_flash /out/bin/m4sim_flash /bin/

FROM scratch AS dist-flash-int4
COPY --from=engine-flash-int4 /out/bin/g6m4_flash-int4 /bin/

FROM scratch AS dist-ds41
COPY --from=engine-ds41 /out/bin/g6m4_ds41 /out/bin/m4sim_ds41 /bin/

FROM scratch AS dist-ink2
COPY --from=engine-ink2 /out/bin/g6m4_ink2 /out/bin/m4sim_ink2 /bin/

FROM scratch AS dist-qw38
COPY --from=engine-qw38 /out/bin/g6m4_qw38 /out/bin/m4sim_qw38 /bin/

FROM scratch AS dist-tests
COPY --from=engine-tests-ship /out/bin/gpuprobe /out/bin/tr3_e4m3_test /bin/

FROM scratch AS dist
COPY --from=dist-rust / /
COPY --from=dist-fp8 / /
COPY --from=dist-int4 / /
COPY --from=dist-flash / /
COPY --from=dist-flash-int4 / /
COPY --from=dist-ds41 / /
COPY --from=dist-ink2 / /
COPY --from=dist-qw38 / /
COPY --from=dist-tests / /

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS runtime
# libibverbs1 + ibverbs-providers (mlx5) for the fabric, ibverbs-utils for ibv_devinfo, python3 for the shims and
# the launcher (python3-yaml for KV_STORE=@FILE), curl + jq for smoketest/run.sh (./gr8 glm53 smoke). Every engine here
# serves through S2, so nothing in the container uses ssh. The runtime base already has cuBLAS.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers ibverbs-utils python3 python3-yaml \
      curl jq \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/gr8
COPY . .
COPY --from=dist / .
# The container runs as uid 1000 (the base image's ubuntu account) on a read-only image. Its writable state is two
# volumes (docker/compose.yaml): the model cache and GR8_WORK_DIR, which docker fills from these empty, uid-1000 directories
# on first use.
RUN mkdir -p /var/lib/gr8/home /var/lib/gr8/cache \
 && chown 1000:1000 /var/lib/gr8/home /var/lib/gr8/cache
ENV PATH=/opt/gr8/bin:$PATH PYTHONPATH=/opt/gr8/frontend HOME=/tmp GR8_CONTAINER=1 \
    GR8_WORK_DIR=/var/lib/gr8/home MODEL_CACHE=/var/lib/gr8/cache PY=/usr/bin/python3
USER 1000:1000
# ./gr8 glm53 COMMAND; with no command, `up` in the foreground
ENTRYPOINT ["/opt/gr8/docker/entrypoint.sh"]
