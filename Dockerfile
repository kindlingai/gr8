# gr8 for GB10 (DGX Spark / ASUS GX10): arm64, CUDA 13, sm_121a. docker/README.md has how to run it.
#
#   docker buildx build --platform linux/arm64 -t gr8 .                         the runtime image
#   docker buildx build --platform linux/arm64 --target dist -o type=local,dest=out .
#                                                                              only the binaries, into out/bin
#
# Stages: toolchain (CUDA devel + rdma-core) -> rust -> loader (the Rust loader) and kvman (the Rust S2 page allocator)
# -> engine and engine-int4, with front (the Rust front end) beside them -> dist (the binaries only)
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
# The front end pins its compiler in frontend/g6front-rs/rust-toolchain.toml. Keep RUST_VERSION equal to that pin.
ARG RUST_VERSION=1.94.1
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "$RUST_VERSION"
WORKDIR /src

# Each Rust build copies its output out and drops target/ in the same layer, so the cached layer holds one file.
FROM rust AS loader
COPY engine/loader-rs engine/loader-rs
RUN mkdir -p /out \
 && cargo build --release --locked --manifest-path engine/loader-rs/Cargo.toml \
 && cp engine/loader-rs/target/release/libg6load.a engine/loader-rs/target/release/g6load /out/ \
 && rm -rf engine/loader-rs/target

FROM rust AS kvman
COPY engine/kvman-rs engine/kvman-rs
RUN mkdir -p /out \
 && cargo build --release --locked --manifest-path engine/kvman-rs/Cargo.toml \
 && cp engine/kvman-rs/target/release/libg6kvman.a engine/kvman-rs/target/release/kvman-remote /out/ \
 && rm -rf engine/kvman-rs/target

# the slowest step (fat LTO); it runs beside the engine builds
FROM rust AS front
COPY frontend/g6front-rs frontend/g6front-rs
# from the crate's directory, so rustup reads its rust-toolchain.toml
WORKDIR /src/frontend/g6front-rs
RUN mkdir -p /out \
 && cargo build --release --locked \
 && cp target/release/g6front-rs /out/ \
 && rm -rf target

# engine-src copies only the engine sources, so edits to docs, scripts or the front end keep the engine layers cached.
# This stage has no cargo: build.sh skips its cargo step and links the libg6load.a and libg6kvman.a built above.
FROM toolchain AS engine-src
WORKDIR /src
COPY engine engine
COPY tools/health/gpuprobe.cu tools/health/gpuprobe.cu
COPY scripts/build.sh scripts/build.sh
COPY --from=loader /out/libg6load.a engine/loader-rs/target/release/libg6load.a
COPY --from=kvman /out/libg6kvman.a engine/kvman-rs/target/release/libg6kvman.a

# one g6m4 and one m4sim for every TP width; every launcher runs this build
FROM engine-src AS engine
RUN scripts/build.sh /out/bin

# the 4-bit KV latent (--kv-format int4, docs/KV4.md): g6m4 alone, built with -DG6_KV4=1, as g6m4-int4 next to the fp8
# engine, for every TP width
FROM engine-src AS engine-int4
RUN KV_FORMAT=int4 scripts/build.sh /out/bin

# rust-dist holds the Rust binaries only: CI builds them on a hosted arm runner, and its engine job links libg6load.a
# and libg6kvman.a
FROM scratch AS rust-dist
COPY --from=loader /out/libg6load.a /out/g6load /
COPY --from=kvman /out/libg6kvman.a /
COPY --from=front /out/g6front-rs /

# dist holds the shipped binaries in bin/: g6m4 (fp8 KV) and g6m4-int4 (--kv-format int4), both for every TP width,
# and g6load (the Hugging Face fetch's planner and checker). build.sh's CPU test binaries and objects stay behind.
# tr3_e4m3_test is the GPU decode check build.sh cannot run without a GPU.
FROM scratch AS dist
COPY --from=engine /out/bin/g6m4 /out/bin/m4sim /out/bin/gpuprobe /out/bin/tr3_e4m3_test /bin/
COPY --from=engine-int4 /out/bin/g6m4-int4 /bin/
COPY --from=front /out/g6front-rs /bin/
COPY --from=loader /out/g6load /bin/
COPY --from=kvman /out/kvman-remote /bin/

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS runtime
# libibverbs1 + ibverbs-providers (mlx5) for the fabric, ibverbs-utils for ibv_devinfo, python3 for the shims,
# python3-numpy + libxxhash0 for tools/convert, curl + jq for smoketest/run.sh (./gr8 glm53 smoke). Every engine here
# serves through S2, so nothing in the container uses ssh. The runtime base already has cuBLAS.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers ibverbs-utils python3 python3-numpy \
      libxxhash0 curl jq \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/gr8
COPY . .
COPY --from=dist / .
# The container runs as uid 1000 (the base image's ubuntu account) on a read-only image. Its writable state is two
# volumes (docker/compose.yaml): the model cache and G6_HOME, which docker fills from these empty, uid-1000 directories
# on first use. bench/pbench.py appends results.jsonl next to itself, so that file links into G6_HOME/out.
# bin3/ and bin6/ held the TP=3 and TP=6 builds, and docker/entrypoint.sh still links one of them: bin/ runs every width.
RUN mkdir -p /var/lib/gr8/home /var/lib/gr8/cache \
 && chown 1000:1000 /var/lib/gr8/home /var/lib/gr8/cache \
 && ln -s /var/lib/gr8/home/out/pbench-results.jsonl bench/results.jsonl \
 && ln -s bin bin3 && ln -s bin bin6
ENV PATH=/opt/gr8/bin:$PATH PYTHONPATH=/opt/gr8/frontend HOME=/tmp GR8_CONTAINER=1 \
    G6_HOME=/var/lib/gr8/home MODEL_CACHE=/var/lib/gr8/cache PY=/usr/bin/python3
USER 1000:1000
# ./gr8 glm53 COMMAND; with no command, `up` in the foreground
ENTRYPOINT ["/opt/gr8/docker/entrypoint.sh"]
