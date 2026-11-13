# gr8 for GB10 (DGX Spark / ASUS GX10): arm64, CUDA 13, sm_121a. docker/README.md has how to run it.
#
#   docker buildx build --platform linux/arm64 -t gr8 .                         the runtime image
#   docker buildx build --platform linux/arm64 --target dist -o type=local,dest=out .
#                                                                              only the binaries, into out/bin
#
# Stages: toolchain (CUDA devel + rdma-core) -> rust -> loader (the Rust loader) and kvman (the host Rust: page allocator, scheduler, planner)
# -> engine, engine-int4, engine-flash, engine-flash-int4, engine-ds41 and engine-ink2, with front (the Rust front end) beside them -> dist (the binaries only)
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
 && cp engine/loader-rs/target/release/libg6load.a engine/loader-rs/target/release/g6load /out/ \
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
 && cp engine/target/release/libg6host.a engine/target/release/kvman-remote /out/ \
 && rm -rf engine/target

FROM rust AS knobs
COPY engine/knobs-rs engine/knobs-rs
COPY knobs knobs
RUN mkdir -p /out \
 && cargo build --release --locked --features capi --manifest-path engine/knobs-rs/Cargo.toml \
 && cargo rustc --release --locked --features capi --lib --crate-type cdylib --manifest-path engine/knobs-rs/Cargo.toml \
 && cp engine/knobs-rs/target/release/libgr8knobs.a engine/knobs-rs/target/release/libgr8knobs.so /out/ \
 && rm -rf engine/knobs-rs/target

# the slowest step (fat LTO); it runs beside the engine builds
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

# engine-src copies only the engine sources, so edits to docs, scripts or the front end keep the engine layers cached.
# This stage has no cargo: build.sh skips its cargo step and links the libg6load.a, libg6host.a and libgr8knobs.a
# built above.
# The compile cache (docs/BUILD-CACHE.md) is a cache mount in each engine stage, one id a stage, which CI keeps between
# runs (ci.yml). A layer that has to run again then compiles only what its edit reaches. After the build the stage
# deletes the entries the build did not read (a hit sets its entry's time), so the cache holds one build.
FROM toolchain AS engine-src
ENV GR8_NVCC_CACHE_DIR=/var/cache/gr8-nvcc
WORKDIR /src
COPY engine engine
COPY tools/health/gpuprobe.cu tools/health/gpuprobe.cu
COPY scripts/build.sh scripts/build.sh
COPY --from=loader /out/libg6load.a engine/loader-rs/target/release/libg6load.a
COPY --from=kvman /out/libg6host.a engine/target/release/libg6host.a
COPY --from=knobs /out/libgr8knobs.a engine/knobs-rs/target/release/libgr8knobs.a

# the GLM and Flash engine stages add XGrammar; the tests build without it, so they do not wait for its stage
FROM engine-src AS engine-src-glm
COPY --from=xgrammar-src /out-xgrammar /opt/xgrammar
COPY --from=xgrammar /out/libxgrammar.a /opt/xgrammar-lib/libxgrammar.a
ENV GR8_XGRAMMAR=/opt/xgrammar GR8_XGRAMMAR_LIB=/opt/xgrammar-lib/libxgrammar.a

# one g6m4 and one m4sim for every TP width; every launcher runs this build.
# Each engine stage keeps only the binaries its dist stage copies: a cached layer is reused whole or rebuilt whole, so
# its objects and build-only binaries are dead weight in the CI cache.
# GR8_WIDTHS builds a subset (build.sh): CI passes one on a pull request and
# leaves it empty, for all seven widths (1 2 3 4 5 6 8), on main and on tags.
FROM engine-src-glm AS engine
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-fp8,target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" GR8_TESTS=0 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4 ! -name m4sim -exec rm -rf {} +

# the test binaries build.sh makes after the GLM engine (gpuprobe and tr3_e4m3_test ship), on their own CI runner
FROM engine-src AS engine-tests
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-tests,target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" GR8_TESTS=only scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name gpuprobe ! -name tr3_e4m3_test -exec rm -rf {} +

# GLM-5.3-Flash: g6m4_flash and m4sim_flash (MODEL=flash), TP 1, 2, 3 and 4
FROM engine-src-glm AS engine-flash
RUN --mount=type=cache,id=gr8-nvcc-flash,target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && MODEL=flash scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_flash ! -name m4sim_flash -exec rm -rf {} +

# GLM-5.3-Flash with the 4-bit KV latent (--kv-format int4, docs/FLASH.md): g6m4_flash-int4, TP 1, 2, 3 and 4
FROM engine-src-glm AS engine-flash-int4
RUN --mount=type=cache,id=gr8-nvcc-flash-int4,target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && MODEL=flash KV_FORMAT=int4 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_flash-int4 -exec rm -rf {} +

# DeepSeek-V4.1-Flash-1017: g6m4_ds41 and m4sim_ds41 (MODEL=ds41), TP 4. build.sh builds it without request grammars, so
# it starts from engine-src and does not wait for XGrammar. Its build knobs (GR8_DS_TRUNK_BF16) come from
# ds41_build_flags.sh, which only this stage copies, so the other engine stages keep their cache.
FROM engine-src AS engine-ds41
COPY scripts/ds41_build_flags.sh scripts/ds41_build_flags.sh
RUN --mount=type=cache,id=gr8-nvcc-ds41,target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && MODEL=ds41 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_ds41 ! -name m4sim_ds41 -exec rm -rf {} +

# Inkling-2-Small: g6m4_ink2 and m4sim_ink2 (MODEL=ink2), TP 1, 4 and 6, without request grammars
FROM engine-src AS engine-ink2
RUN --mount=type=cache,id=gr8-nvcc-ink2,target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && MODEL=ink2 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4_ink2 ! -name m4sim_ink2 -exec rm -rf {} +

# the 4-bit KV latent (--kv-format int4, docs/KV4.md): g6m4 alone, built with -DG6_KV4=1, as g6m4-int4 next to the fp8
# engine, for every TP width
FROM engine-src-glm AS engine-int4
ARG GR8_WIDTHS=
RUN --mount=type=cache,id=gr8-nvcc-int4,target=/var/cache/gr8-nvcc \
    touch "$GR8_NVCC_CACHE_DIR/.start" \
 && GR8_WIDTHS="$GR8_WIDTHS" KV_FORMAT=int4 scripts/build.sh /out/bin \
 && find "$GR8_NVCC_CACHE_DIR" -type f ! -newer "$GR8_NVCC_CACHE_DIR/.start" -delete \
 && find /out/bin -mindepth 1 -maxdepth 1 ! -name g6m4-int4 -exec rm -rf {} +

# rust-dist holds the Rust binaries only: CI builds them on a hosted arm runner, and its engine job links libg6load.a,
# libg6host.a and libgr8knobs.a
FROM scratch AS rust-dist
COPY --from=loader /out/libg6load.a /out/g6load /
COPY --from=kvman /out/libg6host.a /
COPY --from=knobs /out/libgr8knobs.a /
COPY --from=front /out/g6front-rs /

# dist holds the shipped binaries in bin/: g6m4 (fp8 KV) and g6m4-int4 (--kv-format int4), both for every TP width,
# and g6load (the Hugging Face fetch's planner and checker). build.sh's CPU test binaries and objects stay behind.
# tr3_e4m3_test is the GPU decode check build.sh cannot run without a GPU.
# dist-fp8, dist-int4, dist-flash, dist-flash-int4, dist-ds41, dist-ink2 and dist-tests are the seven parts of dist. Their
# stages are the whole cost of a build and they share nothing, so CI builds each
# in its own job and unpacks the artifacts into one directory. dist stays the
# union, for a local build and for anything that wants every binary from one target.
FROM scratch AS dist-fp8
COPY --from=engine /out/bin/g6m4 /out/bin/m4sim /bin/
COPY --from=front /out/g6front-rs /bin/
COPY --from=loader /out/g6load /bin/
COPY --from=kvman /out/kvman-remote /bin/
# the knob table the launcher reads (tools/launch/knobs.py)
COPY --from=knobs /out/libgr8knobs.so /bin/lib/

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

FROM scratch AS dist-tests
COPY --from=engine-tests /out/bin/gpuprobe /out/bin/tr3_e4m3_test /bin/

FROM scratch AS dist
COPY --from=dist-fp8 / /
COPY --from=dist-int4 / /
COPY --from=dist-flash / /
COPY --from=dist-flash-int4 / /
COPY --from=dist-ds41 / /
COPY --from=dist-ink2 / /
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
