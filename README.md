# llama-server-native

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-v0.7.0-informational)](https://github.com/ggml-org/llama.cpp)
[![CUDA](https://img.shields.io/badge/CUDA-12.9.2-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](compose.yaml)

[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` and its built-in Web UI,
compiled **specifically for the hardware of the machine it runs on** and served from Docker.

- **Native build.** `-march=native` and the exact CUDA compute capability of your GPU, instead of
  the portable "runs everywhere" binaries of the official images.
- **Hardware guard.** The image remembers which CPU and GPU it was built for and refuses to start
  on different hardware, with a message saying what changed.
- **Multi-model router.** Drop `.gguf` files in a folder and pick them from the Web UI; models are
  loaded and unloaded on demand.
- **Private web search.** The model can search the web and read pages through a local
  [SearXNG](https://github.com/searxng/searxng) instance exposed as an MCP server: no API keys,
  no quotas, no third-party tracking.
- **Nothing to install on the host** besides Docker, the NVIDIA driver and the NVIDIA Container
  Toolkit. The whole toolchain (CUDA, CMake, GCC, Node) lives inside the build.

```bash
cp env.template .env          # once
docker compose build          # compile for this machine (~10-20 min)
docker compose up -d          # http://localhost:8080
```

> **Want to launch the server with other options?** → [LAUNCH.md](LAUNCH.md)

---

## Requirements

- Linux with an NVIDIA GPU and its proprietary driver
- Docker with Compose v2
- The **NVIDIA Container Toolkit**: it is what lets Docker see the GPU, so by definition it cannot
  live inside the container

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check that Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.2-base-ubuntu24.04 nvidia-smi -L
```

---

## Commands

### From any directory

The container is always named `llama-server`, so there is no need to look it up with `docker ps`:

```bash
docker stop llama-server        # stop
docker start llama-server       # start
docker restart llama-server     # restart
docker logs -f llama-server     # follow the logs
docker ps -f name=llama-server  # is it running?
```

### From the project directory

`docker compose` needs to see `compose.yaml`, so these run from the project directory (or with
`docker compose -f /path/to/llama-server-native/compose.yaml …` from anywhere):

```bash
docker compose build                 # build the native image (~10-20 min)
docker compose build --no-cache      # rebuild from scratch
docker compose up -d                 # create and start
docker compose down                  # stop and remove the container
docker compose logs -f               # logs
docker compose ps                    # status
```

### Which one to use

This is the most confusing part, so here it is explicitly:

| You want to… | Command | Why |
|---|---|---|
| Stop / start without changing anything | `docker stop` / `start llama-server` | Same container, same configuration |
| Make the router see a new model | `docker restart llama-server` | Models are scanned at startup |
| Apply a change to `.env` | `docker compose up -d` | **`docker restart` does NOT re-read `.env`** |
| Change version, CUDA or `CUDA_ARCH` | `docker compose build && docker compose up -d` | A rebuild is needed |

A container's environment is fixed when it is **created**. `docker restart` restarts the process
but keeps the variables it was created with, so after editing `.env` you need `up -d`, which
recreates the container if the configuration changed (and does nothing otherwise).

### Diagnostics

The image is named `llama-server-native:<LLAMA_CPP_REF>`, i.e. `llama-server-native:v0.7.0` with
the default configuration. If you change the version, change the tag below too.

```bash
# Which CPU, GPU and version the image was built for
docker run --rm --entrypoint cat llama-server-native:v0.7.0 /opt/llama/build-info.env

# Benchmark the native build
docker run --rm --gpus all -v "$PWD/models:/models" -v "$PWD/cache:/cache" \
  --entrypoint /opt/llama/llama-bench llama-server-native:v0.7.0 -m /models/your-model.gguf

# Every llama-server option
docker run --rm --gpus all llama-server-native:v0.7.0 --help
```

---

## Models

Two sources, and the server sees both.

**a) Download from Hugging Face.** llama.cpp has no `download` command: the way to put a model in
the cache is to start the server with `-hf`. It downloads the model, loads it and starts listening;
**once you see `listening on`, the download is done: press Ctrl+C**.

```bash
docker compose run --rm llama-server -hf ggml-org/gpt-oss-20b-GGUF --no-webui -ngl 0
```

`-ngl 0` keeps the VRAM untouched: loading is instant via mmap and does not fail even if the model
does not fit on the GPU. The model ends up in `./cache`. You will see a warning that `-ngl`
overrides the environment variable; that is intended.

`docker compose run` does not publish ports, so this works while the main server is running.

**b) Manually.** Any `.gguf` in `./models/`:

```bash
curl -L -o models/model.gguf https://huggingface.co/USER/REPO/resolve/main/FILE.gguf
```

If a model has several files (multimodal with its `mmproj-*.gguf`, or split into shards), put them
in their own subdirectory inside `models/`.

To list what you have:

```bash
find -L models cache -name '*.gguf' -type f -exec du -Lh {} +
```

After adding a model, run `docker restart llama-server` so the router lists it.

For private or *gated* Hugging Face repos, set `HF_TOKEN` in `.env`.

### Choosing a model

Open <http://localhost:8080>. The selector groups models into *Loaded* and *Available*; picking one
loads it automatically. `MODELS_MAX` in `.env` limits how many can be loaded at once (1 by
default; see below).

---

## Internet access

The model can search the web and read pages. **It is enabled by default**: `docker compose up -d`
also starts **SearXNG**, the local metasearch engine behind it.

SearXNG runs on your machine: no API keys, no quotas, and your searches are not tied to an
identity at any third-party service. It publishes no ports; only llama-server queries it, over the
compose internal network.

To disable it, set `WEB_SEARCH=` (empty) in `.env` and, if you don't want the search engine
running either, `docker compose stop searxng`.

The model gets two tools:

| Tool | What it does |
|---|---|
| `web_search` | Runs a query and returns the title, URL and snippet of each result |
| `fetch_url` | Downloads a page and converts it to readable text |

In the Web UI they appear in the conversation's tools panel. Through the API you pass them yourself
in the `tools` field and execute the calls against `POST /tools`: **the tool loop runs on the
client**, not on the server.

Details, and how to add other MCP servers: [LAUNCH.md](LAUNCH.md#12-tools-and-agents).

---

## Configuration

Everything lives in `.env` (`cp env.template .env` the first time). `docker compose` reads it
automatically.

**Require a rebuild** (`docker compose build && docker compose up -d`):

| Variable | Default | |
|---|---|---|
| `LLAMA_CPP_REF` | `v0.7.0` | llama.cpp version. Also the image tag |
| `CUDA_ARCH` | `120a-real` | Compute capability of your GPU |
| `CUDA_VERSION` | `12.9.2` | CUDA toolkit used to compile |

**Only require recreating the container** (`docker compose up -d`):

| Variable | Default | Why this value |
|---|---|---|
| `N_GPU_LAYERS` | `auto` | Splits layers between GPU and CPU automatically. **A fixed number breaks `--fit`** |
| `FIT_TARGET` | `1024` | MiB of VRAM left free (your desktop uses the GPU too) |
| `CTX_SIZE` | `16384` | ~12,000 words of history. **Do not set it to 0** |
| `CACHE_TYPE_K` | `q8_0` | 8-bit KV cache: half the VRAM of `f16` |
| `CACHE_TYPE_V` | `q8_0` | Same |
| `FLASH_ATTN` | `auto` | Required for the quantized KV cache |
| `CACHE_REUSE` | `256` | Reuses the prompt prefix between turns |
| `MODELS_MAX` | `1` | On 16 GB, two models steal VRAM from each other |
| `THREADS` | `-1` | Auto |
| `PORT` | `8080` | Published **on 127.0.0.1 only** |
| `WEB_SEARCH` | `/opt/llama/mcp.json` | Web search tools. Empty disables them |
| `EXTRA_ARGS` | empty | Any other `llama-server` flag |
| `HF_TOKEN` | empty | For private or *gated* repos |

Each one is documented in `env.template` with the reasoning behind it.

### The three values that cause the most trouble

These are the ones that broke when testing a 27B model, so here they are explicitly:

**`N_GPU_LAYERS` must be `auto`.** Setting a number disables `--fit`, the mechanism that splits
layers between GPU and CPU. Without `--fit`, a model that does not fit entirely **fails to load**
instead of being split:

```
common_fit_params: n_gpu_layers already set by user to 999, abort
cudaMalloc failed: out of memory
```

**`CTX_SIZE` must not be `0`.** `0` means "the model's native context", and some models have 256k:
that KV cache eats all the VRAM and the model ends up on the CPU. Measured with a 27B model:
5.7 tok/s with the native context versus 13.6 with `16384`.

**`MODELS_MAX` should be `1` on a 16 GB GPU.** With two models loaded they take memory from each
other. Measured: with a 27B also loaded, a 1.7B model dropped from 357 to 57 tok/s. Switching
models means reloading, which takes a few seconds and is well worth it.

### Finding your `CUDA_ARCH`

It is the only hardware detail you have to set by hand, because `docker build` cannot see the GPU:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# NVIDIA GeForce RTX 5070 Ti, 12.0   →   120   →   CUDA_ARCH=120a-real
```

Remove the dot from the capability. The `a` suffix is for Blackwell and newer (cc ≥ 12.0); older
cards go without it (an RTX 4090, cc 8.9, is `89-real`).

If you get it wrong, the hardware guard catches it at startup and tells you.

**Every launch option, explained: [LAUNCH.md](LAUNCH.md).**

### Changing the llama.cpp version

Edit `LLAMA_CPP_REF` in `.env` and rebuild:

- `v0.7.0`, `v0.7.1`… — stable releases (recommended)
- `b10786` — build of a specific commit
- `master` — latest commit; only rebuilt when there are new commits

---

## How it works

### The Dockerfile (4 stages)

```
source   alpine/git      clones llama.cpp at the requested version
   ↓
web      node:24         builds the Web UI (SvelteKit) → tools/ui/dist
   ↓
build    cuda:devel      cmake + gcc-14, -march=native, exact CUDA arch
   ↓
runtime  cuda:runtime    only the binary, its .so files and the entrypoint
```

Only the last stage ends up in the final image: neither the CUDA toolchain nor `node_modules` ship
with it.

### Native, not portable

The official llama.cpp images use `GGML_NATIVE=OFF` + `GGML_CPU_ALL_VARIANTS=ON` so they run on any
PC. This project does the opposite: `-march=native` (which resolves to `arrowlake-s` on the
reference machine) and the GPU's exact compute capability (`120a-real` for an RTX 5070 Ti). The
binary **does not work on another machine**, and that is the point.

Since `docker build` does not expose the GPU, the compute capability is set in `CUDA_ARCH` in
`.env`. If it is wrong, the guard below catches it at startup instead of letting it through.

### The hardware guard

`scripts/hw-fingerprint.sh` computes a hardware fingerprint at build time (stored in the image) and
again at every startup, and compares them. If they don't match, startup stops and says what
changed:

```
  GPU compute capability:
      built for    : 89-real
      this machine : NVIDIA GeForce RTX 5070 Ti (sm_12.0)
```

The GPU is compared by **compute capability**, not by name: what decides whether the binary works
are the SASS kernels inside it. Two different cards with the same capability (a 5070 Ti and a
5080, both sm_120) run the same binary without problems, and the guard won't make you rebuild for
nothing.

To bypass it once: `ALLOW_HW_MISMATCH=1 docker compose up -d`.

The CPU fingerprint filters the flags in `/proc/cpuinfo` through an allowlist of those that affect
`-march`: entries such as `ibrs` or `flush_l1d` change with microcode updates and would cause false
alarms.

---

## About CUDA

The build uses **CUDA 12.9.2**, the latest stable release of the 12.x branch. Current drivers
(tested with 595.84, which supports up to 13.2) cover it natively, so no forward-compatibility
rules are involved.

Three different things are easy to confuse:

| | What it is |
|---|---|
| **Driver** (what `nvidia-smi` shows) | The **highest** CUDA version your driver supports. Not what you use |
| **Toolkit** (`CUDA_VERSION`) | `nvcc` and the headers. Compiles the kernels and puts the libraries in the image |
| **Runtime** (`libcudart.so`) | Library shipped **inside** the image. It is not the driver |

**If you ever move to the 13.x branch, avoid 13.0 and 13.2.** 13.2 miscompiles sub-4-bit
quantizations ([#21255](https://github.com/ggml-org/llama.cpp/issues/21255)) and 13.0 generates
wrong code for Blackwell ([#19168](https://github.com/ggml-org/llama.cpp/issues/19168)). The first
clean release is 13.3.

The Dockerfile removes the *compat libs* that ship with the `nvidia/cuda` images
([#23111](https://github.com/ggml-org/llama.cpp/issues/23111)): they are a user-space driver that
takes precedence over the real one and makes `nvidia-smi` report a different version inside and
outside the container. Without them, your host driver is always used.

---

## Project structure

```
Dockerfile              4 stages: source → web → build → runtime
compose.yaml            services, GPU, ports, volumes
env.template            template → .env
mcp.json                MCP server config (web search) for llama-server
LAUNCH.md               every launch option, explained
scripts/
  hw-fingerprint.sh     CPU/GPU fingerprint (used at build time and at startup)
  entrypoint.sh         checks the fingerprint and launches llama-server
  mcp-websearch.py      MCP server: web_search and fetch_url on top of SearXNG
searxng/settings.yml    local SearXNG configuration
models/                 your .gguf files          → /models
cache/                  Hugging Face downloads    → /cache
```

---

## Notes

**Another UI.** The built-in Web UI is used because it is the only one that supports router mode
(model selector, automatic loading, LRU unloading). The server also exposes an OpenAI-compatible
`/v1` API, so adding Open WebUI or similar is just one more service in `compose.yaml` pointing at
`http://llama-server:8080/v1`.

**No authentication.** The port is published on `127.0.0.1` only. If you expose it to your LAN,
read the network section in [LAUNCH.md](LAUNCH.md#6-network-and-security).

**No Makefile or wrapper scripts.** Everything is plain `docker` and `docker compose`.

**Build time.** ~10-20 min. Most of it is `ggml-cuda`, which is compiled for a single architecture
(yours) instead of all of them.

---

## License

[MIT](LICENSE). llama.cpp and SearXNG are distributed under their own licenses (MIT and AGPL-3.0
respectively); SearXNG runs from its official, unmodified image.
