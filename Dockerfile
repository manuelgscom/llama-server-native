# syntax=docker/dockerfile:1.7
#
# llama-server compiled NATIVELY for this machine.
#
# Derived from llama.cpp's .devops/cuda.Dockerfile, but with its flags inverted:
# the official Dockerfile uses GGML_NATIVE=OFF + GGML_CPU_ALL_VARIANTS=ON +
# GGML_BACKEND_DL=ON to produce images that are portable to any machine.
# Here we want the exact opposite: a binary optimized for this CPU and this
# GPU, even if it won't run anywhere else.

# 12.9.2: latest stable release of the 12.x branch (13.0 comes next). Chosen
# because the host driver covers it natively and with headroom, without relying
# on minor version compatibility. See the README for details.
ARG CUDA_VERSION=12.9.2
ARG UBUNTU_VERSION=24.04
ARG GCC_VERSION=14
ARG NODE_VERSION=24

# ===========================================================================
# Stage 1: source  -- clone llama.cpp (so git is not needed on the host)
# ===========================================================================
FROM alpine/git AS source

ARG LLAMA_CPP_REF=v0.7.0

# Invalidates the clone layer cache when REF points to a moving branch
# (master) and there are new commits. With a fixed tag the JSON does not change
# and the layer is reused, so it is harmless.
ADD https://api.github.com/repos/ggml-org/llama.cpp/commits/${LLAMA_CPP_REF} /tmp/ref.json

RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp /src \
    && git -C /src rev-parse --short HEAD > /src/.commit-sha

# ===========================================================================
# Stage 2: web  -- build the Web UI (SvelteKit) embedded in the binary
# ===========================================================================
# Since v0.3.0 the UI is no longer prebuilt in the repo. scripts/ui-assets.cmake
# looks for it, in order, in: (1) an existing tools/ui/dist, (2) npm run build,
# (3) a download from a HuggingFace bucket. We use (1): build it here and put it
# in place before invoking cmake. Option (3) is explicitly disabled further down
# (LLAMA_USE_PREBUILT_UI=OFF) so that, if something fails here, the build
# breaks instead of silently shipping a UI from a different version.
# This stage is throwaway; it does not end up in the final image.
FROM node:${NODE_VERSION} AS web

WORKDIR /ui

# Manifests first, so `npm ci` is cached and not re-run every time the UI
# code changes.
COPY --from=source /src/tools/ui/package.json /src/tools/ui/package-lock.json ./
RUN npm ci

COPY --from=source /src/tools/ui/ ./
RUN npm run build && test -f dist/index.html

# ===========================================================================
# Stage 3: build  -- native compilation
# ===========================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG GCC_VERSION
ARG CUDA_VERSION
ARG LLAMA_CPP_REF

# The GPU is not visible during `docker build`, so the compute capability is
# set by hand in .env. The entrypoint checks at startup that it matches the
# actual GPU, so a wrong value is detected rather than slipping through.
ARG CUDA_ARCH=120a-real

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} g++-${GCC_VERSION} \
        build-essential cmake git libssl-dev libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VERSION} \
    CXX=g++-${GCC_VERSION} \
    CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /app
COPY --from=source /src ./
COPY --from=web /ui/dist ./tools/ui/dist

# Record what GCC resolved -march=native to. On an Arrow Lake CPU it should be
# arrowlake-s; a generic fallback (x86-64) would mean GCC does not recognize
# the CPU and we would be silently losing performance.
RUN set -eu; \
    march=$(${CXX} -march=native -Q --help=target 2>/dev/null \
            | sed -n 's/^[[:space:]]*-march=[[:space:]]*//p' | head -n1); \
    echo "${march:-unknown}" > /tmp/march; \
    echo "=============================================="; \
    echo " -march=native  ->  ${march:-unknown}"; \
    echo " CUDA arch      ->  ${CUDA_ARCH}"; \
    echo "=============================================="

# No cache mount for the build directory, on purpose: .o files compiled with
# -march=native carry no record of which CPU they were built for, so reusing
# them on another machine would yield a silently incorrect binary.
#
# --allow-shlib-undefined: libggml-cuda.so uses the driver API (cuMemCreate,
# cuMemMap...) which lives in libcuda.so.1, and that library is NOT in the build
# image: the container toolkit injects it at runtime. Without this flag,
# linking llama-server fails with "undefined reference to cuMemCreate".
RUN set -eu; \
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_SUBPROCESS=ON \
        -DLLAMA_OPENSSL=ON \
        -DLLAMA_BUILD_UI=ON \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" \
        --target llama-server llama-bench \
    && mkdir -p /opt/llama/lib \
    && cp build/bin/llama-server build/bin/llama-bench /opt/llama/ \
    && find build -name '*.so*' -exec cp -P {} /opt/llama/lib/ \;

# Hardware fingerprint: what the entrypoint compares on every start.
# The CPU is detected here (the build sees the host's real CPU); for the GPU,
# CUDA_ARCH is enough, since it already holds the target compute capability.
COPY scripts/hw-fingerprint.sh /opt/llama/hw-fingerprint.sh
COPY scripts/entrypoint.sh     /opt/llama/entrypoint.sh
COPY scripts/mcp-websearch.py  /opt/llama/mcp-websearch.py

RUN set -eu; \
    { sh /opt/llama/hw-fingerprint.sh; \
      echo "CPU_MARCH=$(cat /tmp/march)"; \
      echo "CUDA_ARCH=${CUDA_ARCH}"; \
      echo "CUDA_VERSION=${CUDA_VERSION}"; \
      echo "LLAMA_CPP_REF=${LLAMA_CPP_REF}"; \
      echo "LLAMA_COMMIT=$(cat .commit-sha)"; \
      echo "BUILD_DATE=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"; \
    } > /opt/llama/build-info.env; \
    chmod +x /opt/llama/hw-fingerprint.sh /opt/llama/entrypoint.sh /opt/llama/mcp-websearch.py; \
    cat /opt/llama/build-info.env

# ===========================================================================
# Stage 4: runtime
# ===========================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# python3 is here only for the MCP servers: llama-server launches them as child
# processes INSIDE this container, so they need an interpreter.
# It is python3-minimal (~35 MB) and the search server uses only the stdlib,
# no pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates python3-minimal python3-urllib3 \
    && rm -rf /var/lib/apt/lists/*

# nvidia/cuda images ship /usr/local/cuda-*/compat: a user-space driver NEWER
# than the host's (e.g. 610.43.02 vs. a host 595.84), plus a file in
# /etc/ld.so.conf.d that puts it ahead of the real driver. This is CUDA
# "forward compatibility", which NVIDIA only supports on datacenter GPUs
# (Tesla), not GeForce. It also hides which driver is in use: nvidia-smi inside
# the container may even report a CUDA version different from the host's.
#
# We remove it so the host's real driver is always used, which is the supported
# path (minor version compatibility: any 13.x toolkit runs on a 13.x driver).
# See https://github.com/ggml-org/llama.cpp/issues/23111
RUN rm -rf /usr/local/cuda*/compat /etc/ld.so.conf.d/*compat*.conf && ldconfig

COPY --from=build /opt/llama /opt/llama

ENV LD_LIBRARY_PATH=/opt/llama/lib:${LD_LIBRARY_PATH} \
    PATH=/opt/llama:${PATH} \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080

WORKDIR /opt/llama
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/opt/llama/entrypoint.sh"]
