# Launching llama-server — options guide

Every option `llama-server` can be launched with in this project, grouped by topic and explained.
They correspond to **llama.cpp v0.7.0**; to regenerate the raw list:

```bash
docker run --rm --gpus all llama-server-native:v0.7.0 --help
```

---

## Contents

- [The three ways to pass options](#the-three-ways-to-pass-options)
- [1. Model and loading](#1-model-and-loading)
- [2. GPU and memory](#2-gpu-and-memory)
- [3. Context and KV cache](#3-context-and-kv-cache)
- [4. CPU and threads](#4-cpu-and-threads)
- [5. Multi-model router](#5-multi-model-router)
- [6. Network and security](#6-network-and-security)
- [7. Web UI](#7-web-ui)
- [8. Chat, templates and reasoning](#8-chat-templates-and-reasoning)
- [9. Sampling](#9-sampling)
- [10. Speculative decoding](#10-speculative-decoding)
- [11. Multimodal](#11-multimodal-image-audio-video-pdf)
- [12. Tools and agents](#12-tools-and-agents)
- [13. Embeddings and reranking](#13-embeddings-and-reranking)
- [14. Logging and diagnostics](#14-logging-and-diagnostics)
- [15. Model shortcuts](#15-model-shortcuts)
- [Recipes](#recipes)

---

## The three ways to pass options

### A) Named variables in `.env`

The everyday ones are already wired into `compose.yaml`. Edit `.env` and run `docker compose up -d`
(**not** `docker restart`: that keeps the environment the container was created with):

| `.env` | Equivalent flag | Default |
|---|---|---|
| `PORT` | `--port` | `8080` |
| `CTX_SIZE` | `--ctx-size` | `16384` |
| `N_GPU_LAYERS` | `--n-gpu-layers` | `auto` |
| `FIT_TARGET` | `--fit-target` | `1024` |
| `CACHE_TYPE_K` | `--cache-type-k` | `q8_0` |
| `CACHE_TYPE_V` | `--cache-type-v` | `q8_0` |
| `FLASH_ATTN` | `--flash-attn` | `auto` |
| `CACHE_REUSE` | `--cache-reuse` | `256` |
| `MODELS_MAX` | `--models-max` | `1` |
| `THREADS` | `--threads` | `-1` |

### B) `EXTRA_ARGS` in `.env`

Any other flag, verbatim, separated by spaces:

```
EXTRA_ARGS=--metrics --reasoning-budget 2048 --api-key my-key
```

### C) Presets file, for per-model flags

This is what you want when, for example, a 30B and a 4B model live side by side. Create
`models/models.ini` (that folder is already mounted) and enable it with
`EXTRA_ARGS=--models-preset /models/models.ini`:

```ini
version = 1

[*]                                       ; applies to all models
c = 8192

[ggml-org/gpt-oss-20b-GGUF:Q4_K_M]        ; this model only; overrides [*]
c = 32768
cache-type-k = q8_0
cache-type-v = q8_0
```

The key is the flag without dashes (`--n-gpu-layers 123` → `n-gpu-layers = 123`). Short forms
(`c`, `ngl`) and variable names (`LLAMA_ARG_N_GPU_LAYERS`) also work.

Besides the regular flags, presets accept three keys of their own:

| Key | What it does |
|---|---|
| `load-on-startup` | Loads the model when the server starts instead of waiting for the first request |
| `stop-timeout` | Seconds to wait before killing the process when unloading it (default 10) |
| `dedup-cache-models` | Hides the duplicate entry when a preset points to a model already in the cache |

**Precedence:** command line > model section > `[*]` section.

### How they combine

This project starts in **router mode**: a parent process that launches one child process per
model. Both environment variables and command-line flags **are inherited by the children**, so all
three methods work for model options.

The router manages `host`, `port`, `alias`, `hf-repo` and the children's API key on its own: it
overrides or removes them on load, so there is no point in putting them in a preset. `--api-key`
does go on the router (method A or B), since it is the one serving requests.

---

## 1. Model and loading

None of these are needed in router mode: models are discovered automatically in `/models` and
`/cache`. Use them if you ever want to pin a single model.

| Flag | What it does |
|---|---|
| `-m, --model FILE` | Path to the `.gguf`. Setting it **disables router mode** |
| `-hf, --hf-repo <user>/<repo>[:quant]` | Download from HuggingFace. Without a quant it uses `Q4_K_M` |
| `-hff, --hf-file FILE` | Specific file inside the repo; ignores the quant from `-hf` |
| `-hft, --hf-token TOKEN` | HF token for private repos. Better set via `HF_TOKEN` in `.env` |
| `-mu, --model-url URL` | Download from any URL |
| `-dr, --docker-repo <model>[:quant]` | Model from Docker Hub (`ai/` namespace by default) |
| `--offline` | Forbids any network access; only uses what is in the cache |
| `-cl, --cache-list` | Lists the cache contents and exits |
| `-a, --alias NAME` | Name under which the model appears in the API |
| `--tags TAG1,TAG2` | Informational tags; they do not affect routing |

**Memory loading mode:**

| Flag | What it does |
|---|---|
| `-lm, --load-mode MODE` | `auto` (default), `none`, `mmap`, `mlock`, `mmap+mlock`, `dio` |
| `-lzm, --lazy-mode MODE` | Reads certain large tensors (e.g. per-layer embeddings) from disk on demand instead of keeping them resident. `auto` (default: only those over 4 GiB), `on`, `off`. Requires mmap |
| `--check-tensors` | Validates tensors on load. Slow; only for diagnosing a suspicious GGUF |
| `--override-kv KEY=TYPE:VALUE` | Overrides model metadata |

`mmap` (the default) maps the file from disk and only loads what is used. `mlock` forces it to stay
in RAM without swapping: useful if the system is paging the model out. `dio` uses DirectIO.

> The standalone `--mlock`, `--mmap` and `--direct-io` flags **no longer exist**: if you pass them, llama-server will not start. Use `--load-mode`.

**Adapters:**

| Flag | What it does |
|---|---|
| `--lora FILE` | Loads a LoRA adapter (several, comma-separated) |
| `--lora-scaled FILE:SCALE` | LoRA with a scale factor |
| `--lora-init-without-apply` | Loads the LoRAs without applying them; they are enabled later via `POST /lora-adapters` |
| `--control-vector FILE` | Adds a control vector |
| `--control-vector-scaled FILE:SCALE` | Control vector with scale |
| `--control-vector-layer-range START END` | Layers to apply the control vectors to |

---

## 2. GPU and memory

The section you will tune most with 16 GB of VRAM.

| Flag | What it does |
|---|---|
| `-ngl, --n-gpu-layers N` | Layers in VRAM. Exact number, `auto` or `all`. Default `auto`. **Setting a number disables `--fit`** |
| `-dev, --device dev1,dev2` | Which devices to use. `none` = everything on CPU |
| `--list-devices` | Lists the available devices and exits |
| `-sm, --split-mode MODE` | With several GPUs: `none`, `layer` (default), `row`, `tensor` |
| `-ts, --tensor-split 3,1` | Share of the model on each GPU |
| `-mg, --main-gpu N` | Main GPU for intermediate results |
| `--op-offload / --no-op-offload` | Offload host tensor operations to the device |

**Automatic fitting — very useful here:**

| Flag | What it does |
|---|---|
| `-fit, --fit [on\|off]` | Adjusts only the parameters you did not set so the model fits in VRAM. **Enabled by default** |
| `-fitt, --fit-target MiB` | Margin to leave free per device (default 1024 MiB) |
| `-fitc, --fit-ctx N` | Minimum context `--fit` is allowed to set (default 4096) |

`--fit` is why models can be loaded without computing layers by hand: if you do not set `-ngl` and
`-c`, it lowers them until the model fits. **If you set `-ngl` to a number, `--fit` gives up**, and a
model that does not fit entirely fails to load instead of being split between GPU and CPU:

```
common_fit_params: failed to fit params to free device memory:
                   n_gpu_layers already set by user to 999, abort
ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory
```

That is why `N_GPU_LAYERS=auto` is the default in `.env`.

**MoE (Mixture of Experts) models — the way to fit large models in 16 GB:**

| Flag | What it does |
|---|---|
| `-cmoe, --cpu-moe` | Keeps **all** expert weights in RAM, the rest on GPU |
| `-ncmoe, --n-cpu-moe N` | Keeps the experts of the first N layers on CPU |
| `--moe-cache-mib N` | VRAM cache (MiB) for the experts that stay on CPU. With several GPUs it is split like the layers. `0` = disabled (default) |
| `-ot, --override-tensor pattern=buffer` | Fine-grained control over which tensor goes to which buffer |
| `-ncffn, --n-cpu-ffn N` | Dense models: keeps the FFN weights of the first N layers on CPU (for MoE use `--n-cpu-moe`) |

In an MoE model most of the weight is in the experts, and only a few are active for each token.
Sending them to RAM and keeping the rest in VRAM makes it possible to run models that would not fit
otherwise, at the cost of speed.

---

## 3. Context and KV cache

| Flag | What it does |
|---|---|
| `-c, --ctx-size N` | Context size. `0` = the model's own. **In `.env` as `CTX_SIZE`** |
| `-n, --predict N` | Maximum tokens to generate. `-1` = unlimited |
| `-b, --batch-size N` | Maximum logical batch (default 2048) |
| `-ub, --ubatch-size N` | Maximum physical batch (default 512) |
| `--keep N` | Tokens of the initial prompt that are never discarded |
| `-np, --parallel N` | Server slots = concurrent conversations. `-1` = auto |
| `-cb / -nocb` | Continuous (dynamic) batching. Enabled by default |

**Quantizing the KV cache — the most direct way to gain context:**

| Flag | What it does |
|---|---|
| `-ctk, --cache-type-k TYPE` | K cache type. `f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1` (default `f16`) |
| `-ctv, --cache-type-v TYPE` | Same for V |
| `-kvo / -nkvo` | Offload the KV cache to the GPU. Enabled |
| `-kvu, --kv-unified` | A single KV buffer shared across sequences |
| `--kv-unified-per-slot N` | Per-slot context limit. Without `-c`, the shared KV pool is sized to `parallel × N` |

With `-ctk q8_0 -ctv q8_0` the cache takes half the space with barely any quality loss. It is the
first thing to change if you run out of VRAM when increasing the context.

**Prompt cache (reusing work across requests):**

| Flag | What it does |
|---|---|
| `--cache-prompt / --no-cache-prompt` | Cache prompts. Enabled |
| `--cache-reuse N` | Minimum chunk to reuse via KV shifting (default 0 = disabled) |
| `-cram, --cache-ram N` | Cache cap in MiB (default 8192, `-1` unlimited, `0` disables) |
| `--cache-idle-slots` | Save idle slots to the prompt cache |
| `-ctxcp, --ctx-checkpoints N` | Context checkpoints per slot (default 32) |
| `-cms, --checkpoint-min-step N` | Minimum spacing between checkpoints, in tokens |
| `-sps, --slot-prompt-similarity N` | How similar a prompt must be to reuse a slot (default 0.10) |
| `--slot-save-path PATH` | Save the slots' KV cache to disk |
| `--context-shift` | Shift the context during infinite generation. **Disabled** by default |
| `--swa-full` | Full-size SWA cache |

`--cache-reuse 256` greatly speeds up long conversations: it reuses the common prefix instead of
reprocessing the whole history.

**Context scaling (RoPE / YaRN)** — to force more context than the model ships with:

`--rope-scaling {none,linear,yarn}`, `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`,
`--yarn-orig-ctx`, `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-slow`, `--yarn-beta-fast`.

Leave these alone unless you know the model supports it; they degrade quality.

---

## 4. CPU and threads

| Flag | What it does |
|---|---|
| `-t, --threads N` | Generation threads. `-1` = auto. **In `.env` as `THREADS`** |
| `-tb, --threads-batch N` | Prompt-processing threads (default: same as `--threads`) |
| `--threads-http N` | Threads serving HTTP |
| `-C, --cpu-mask M` | CPU affinity mask in hexadecimal |
| `-Cr, --cpu-range lo-hi` | CPU range for affinity |
| `--cpu-strict <0\|1>` | Strict placement on the given CPUs |
| `--prio N` | Process priority: `-1` low … `3` realtime |
| `--poll <0-100>` | Busy-wait level (default 50) |
| `--numa TYPE` | `distribute`, `isolate` or `numactl`. Irrelevant on a single NUMA node |
| `--repack / --no-repack` | Weight repacking for the CPU. Enabled |
| `--no-host` | Bypasses the host buffer, allowing extra buffers to be used |

The variants `-Cb`, `-Crb`, `--cpu-strict-batch`, `--prio-batch` and `--poll-batch` exist for the
batch phase separately.

With everything on the GPU these flags barely matter. They gain weight with `-cmoe`, because the
experts are then computed on the CPU cores (20 on the reference machine).

---

## 5. Multi-model router

The mode this project starts in: no `-m` or `-hf`.

| Flag | What it does |
|---|---|
| `--models-dir PATH` | Folder with the `.gguf` files. Here `/models`, already set |
| `--models-preset PATH` | INI with per-model configuration (see above) |
| `--models-max N` | Models loaded at once (default 4). **In `.env` as `MODELS_MAX`** |
| `--models-autoload / --no-models-autoload` | Load on selection, or require explicit loading |
| `--sleep-idle-seconds N` | Put the server to sleep after N idle seconds (`-1` = never) |

Expected layout in `models/`: one standalone `.gguf` per model, or a subfolder when there are
several files (multimodal with its `mmproj-*.gguf`, or models split into shards).

With 16 GB, `MODELS_MAX=2` prevents loading the second model from evicting the first for lack of VRAM.

---

## 6. Network and security

| Flag | What it does |
|---|---|
| `--host IP` | Listening interface. Here `0.0.0.0` inside the container, published only on `127.0.0.1` |
| `--port N` | Port. **In `.env` as `PORT`** |
| `--api-key KEY` | Key-based authentication; several, comma-separated |
| `--api-key-file FILE` | Keys from a file, one per line |
| `--ssl-key-file` / `--ssl-cert-file` | HTTPS with your own certificate |
| `--api-prefix PREFIX` | Serves everything under a prefix, e.g. `/llama` |
| `--cors-origins ORIGINS` | Allowed origins (default `*`). The value `localhost` is the safe mode |
| `--cors-methods` / `--cors-headers` / `--cors-credentials` | Rest of CORS |
| `-to, --timeout N` | Read/write timeout in seconds (default 3600) |
| `--sse-ping-interval N` | SSE streaming ping (default 30 s) |
| `--reuse-port` | Allows several sockets on the same port |
| `--path PATH` | Serve static files from a folder |

**The server has no authentication by default and CORS is set to `*`.** That is why the port is
published only on `127.0.0.1`. If you open it to the LAN, set at least:

```
EXTRA_ARGS=--api-key a-long-key --cors-origins localhost
```

---

## 7. Web UI

| Flag | What it does |
|---|---|
| `--ui / --no-ui` | Enable the Web UI. **Enabled** by default |
| `--ui-config JSON` | Default UI preferences, as JSON |
| `--ui-config-file PATH` | Same, from a file |
| `--ui-mcp-proxy` | CORS proxy so the UI can talk to remote MCP servers. Experimental |
| `--media-path PATH` | Folder from which the UI can load local files via `file://` |

`--ui-config` example:

```
EXTRA_ARGS=--ui-config {"pasteLongTextToFileLen":0,"renderUserContentAsMarkdown":true}
```

---

## 8. Chat, templates and reasoning

| Flag | What it does |
|---|---|
| `--jinja / --no-jinja` | Jinja engine for chat templates. **Enabled** |
| `--chat-template NAME` | Forces a specific template (`chatml`, `llama3`, `gemma`, `gpt-oss`…) |
| `--chat-template-file PATH` | Jinja template from a file |
| `--chat-template-kwargs JSON` | Extra parameters for the template |
| `--skip-chat-parsing` | Does not parse the output: everything goes verbatim into `content` |
| `--prefill-assistant / --no-prefill-assistant` | Continue an assistant message if it is the last one |

**Reasoning ("thinking" models):**

| Flag | What it does |
|---|---|
| `-rea, --reasoning [on\|off\|auto]` | Enable thinking. `auto` infers it from the template |
| `--reasoning-effort LEVEL` | `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `--reasoning-budget N` | Token cap for thinking. `-1` unlimited, `0` cuts off immediately |
| `--reasoning-budget-message TEXT` | Message injected when the budget is exhausted |
| `--reasoning-format FORMAT` | `none`, `deepseek` (in `reasoning_content`), `deepseek-legacy` |
| `--reasoning-preserve` | Keeps the reasoning across the whole history, not just the last message |

If a model keeps thinking indefinitely, `--reasoning-budget 2048` cuts it off. Without a budget,
Qwen3 models can return empty responses because they spend all their tokens thinking.

---

## 9. Sampling

Server defaults; the API and the Web UI can override them per request.

| Flag | Default | What it does |
|---|---|---|
| `--temp N` | 0.80 | Temperature. Lower = more deterministic |
| `--top-k N` | 40 | Only the K most likely tokens. `0` disables |
| `--top-p N` | 0.95 | Cumulative probability nucleus. `1.0` disables |
| `--min-p N` | 0.05 | Minimum probability relative to the best token |
| `--top-nsigma N` | -1.0 | Cutoff by standard deviations |
| `--typical N` | 1.0 | Locally typical sampling |
| `-s, --seed N` | -1 | Seed. `-1` = random |
| `--samplers LIST` | — | Sampler order, separated by `;` |
| `--sampler-seq SEQ` | `edskypmxt` | Same, in abbreviated form |
| `--ignore-eos` | — | Ignores the end-of-sequence token and keeps generating |

**Repetition penalties:**

| Flag | Default | |
|---|---|---|
| `--repeat-last-n N` | 64 | Token window to consider |
| `--repeat-penalty N` | 1.00 | Repetition penalty |
| `--presence-penalty N` | 0.00 | Presence penalty |
| `--frequency-penalty N` | 0.00 | Frequency penalty |
| `--dry-multiplier N` | 0.00 | DRY: multiplier (0 = disabled) |
| `--dry-base N` | 1.75 | DRY: base |
| `--dry-allowed-length N` | 2 | DRY: allowed length before penalizing |
| `--dry-penalty-last-n N` | 64 | DRY: window |
| `--dry-sequence-breaker STR` | — | Sequence breakers for DRY |

**Dynamic temperature and Mirostat:**

`--dynatemp-range`, `--dynatemp-exp`, `--mirostat {0,1,2}`, `--mirostat-lr`, `--mirostat-ent`,
`--adaptive-target`, `--adaptive-decay`, `--xtc-probability`, `--xtc-threshold`.

**Structured output:**

| Flag | What it does |
|---|---|
| `-j, --json-schema SCHEMA` | Forces the output to conform to a JSON Schema |
| `-jf, --json-schema-file FILE` | Same, from a file |
| `--grammar GRAMMAR` | BNF-like grammar |
| `--grammar-file FILE` | Grammar from a file |
| `-l, --logit-bias ID+/-N` | Raises or lowers the probability of a specific token |
| `-bs, --backend-sampling` | Sampling on the backend. Experimental |

---

## 10. Speculative decoding

A small ("draft") model proposes tokens and the large one validates them in bulk. It can nearly
double the speed if both models belong to the same family.

| Flag | What it does |
|---|---|
| `--spec-default` | Default speculative configuration, no tuning needed |
| `-md, --spec-draft-model FILE` | Draft model |
| `-hfd, --spec-draft-hf REPO` | Draft model from HuggingFace |
| `-ngld, --spec-draft-ngl N` | Draft model layers in VRAM |
| `--spec-draft-n-max N` | Tokens to propose per step (default 3) |
| `--spec-draft-n-min N` | Minimum number of draft tokens to use |
| `--spec-draft-p-split P` | Split probability (default 0.10) |
| `--spec-draft-p-min P` | Minimum probability to accept |
| `--spec-draft-sampling MODE` | `greedy` (default) takes the draft's argmax; `probabilistic` samples it and the large model verifies it via rejection sampling |
| `--spec-draft-n-ubatch N` | Draft physical batch. Lowering it saves VRAM (default = `--ubatch-size`) |
| `--spec-type TYPES` | `draft-simple`, `draft-eagle3`, `draft-mtp`, `ngram-simple`, `ngram-map-k`, `ngram-mod`, `ngram-cache`… |

The `ngram-*` types **do not need a second model**: they predict from repetitions in the text
itself. They work well for highly repetitive tasks (code, rewrites) and cost nothing to try.
They are tuned with `--spec-ngram-*-size-n`, `--spec-ngram-*-size-m` and `--spec-ngram-*-min-hits`.

The draft model also has its own set of CPU and cache flags: `-ctkd`, `-ctvd`, `-td`, `-tbd`,
`-Cd`, `-Crd`, `-cmoed`, `-ncmoed`, `-otd`, `-devd`.

`--spec-synth-len` and `--spec-synth-rates` simulate acceptance rates: they are only useful for benchmarks.

---

## 11. Multimodal (image, audio, video, PDF)

| Flag | What it does |
|---|---|
| `-mm, --mmproj FILE` | Multimodal projector. With `-hf` it is downloaded automatically |
| `-mmu, --mmproj-url URL` | Projector from a URL |
| `--no-mmproj` | Do not use a projector even if one exists |
| `--mmproj-offload / --no-mmproj-offload` | Projector on GPU. Enabled |
| `-mmdev, --mmproj-device DEVICE` | Device for the projector |
| `--image-min-tokens N` | Minimum tokens per image (dynamic-resolution models) |
| `--image-max-tokens N` | Maximum tokens per image |
| `--mtmd-batch-max-tokens N` | Image tokens per batch when encoding (default 1024) |
| `--video-fps N` | Frames per second extracted from video (default 4) |
| `--video-timestamp-interval N` | How often, in ms, a text timestamp is inserted (default 5000) |
| `--video-ffmpeg-dir DIR` | Folder containing `ffmpeg` and `ffprobe` (default: looked up in PATH) |

If you add a vision model to `models/`, put it in a subfolder together with its `mmproj-*.gguf`.

Video requires `ffmpeg` and `ffprobe`, which the image does not include: video does not currently work here.

---

## 12. Tools and agents

> All of this is **experimental** and gives the model access to your filesystem. The server
> automatically restricts CORS to localhost when it is enabled. Do not turn it on if the port is open.

| Flag | What it does |
|---|---|
| `--tools LIST` | Built-in tools: `read_file`, `file_glob_search`, `grep_search`, `exec_shell_command`, `write_file`, `edit_file`, `get_info`. `all` enables them all |
| `--tools-runtime OPTION` | Where they run: `docker:<image>`, `podman:<image>`, `ssh:<destination>` |
| `--mcp-servers-config PATH` | MCP servers from JSON (Cursor format) |
| `--mcp-servers-json JSON` | Same, inline |
| `-ag, --agent` | Shortcut: enables the CORS proxy and all tools |

`--tools-runtime docker:ubuntu` makes the tools run in a separate container instead of inside the
server. It is the reasonable minimum if you are going to enable `exec_shell_command`.

### Internet access (already set up)

The built-in tools **do not include web search**. That is what this project's MCP server is for,
**enabled by default**: a regular `docker compose up -d` brings up llama-server and SearXNG
together. Disable it by leaving `WEB_SEARCH=` empty in `.env`.

How the pieces fit together:

```
llama-server
  └─ launches as a child process (stdio, line-delimited JSON-RPC)
     scripts/mcp-websearch.py          ~300 lines, Python stdlib only
       ├─ web_search  → http://searxng:8080  (compose service)
       └─ fetch_url   → downloads and extracts the readable text
```

**Why a custom MCP server.** llama.cpp only supports the **stdio** transport: the MCP server is a
child process running *inside* the container. Typical MCP servers are `npx …`, and adding Node to
the image costs hundreds of MB. Since the protocol is line-delimited JSON-RPC, a Python script using
the standard library does the job — the only addition to the image was `python3-minimal`.

**Why SearXNG instead of a commercial API.** No keys, no quotas, and searches are not tied to an
identity. It fits a server that is compiled and run locally. The trade-off is that it returns links
and short snippets, not extracted text like Tavily or Exa; `fetch_url` covers that.

**The tool loop is driven by the client.** The server exposes `GET /tools` (list) and
`POST /tools` (execute), but does not chain calls on its own. The Web UI does this for you. If you
call `/v1/chat/completions` yourself, you must pass the definitions in `tools` and execute the calls
yourself — otherwise the model will not see them and will make up the answer.

**Adding other MCP servers:** edit `mcp.json`. Each entry accepts `command`, `args`, `env`,
`cwd` and `timeout_ms`. The child process runs with the same privileges as the server, so only
declare what you trust.

---

## 13. Embeddings and reranking

| Flag | What it does |
|---|---|
| `--embedding` | Embeddings-only mode. Use it only with embedding models |
| `--embd-normalize N` | Normalization: `-1` none, `2` Euclidean (default) |
| `--pooling TYPE` | `none`, `mean`, `cls`, `last`, `rank` |
| `--rerank` | Enables the reranking endpoint |

---

## 14. Logging and diagnostics

| Flag | What it does |
|---|---|
| `-lv, --verbosity N` | `0` generic, `1` error, `2` warning, `3` info (default), `4` trace, `5` debug |
| `-v, --verbose` | Everything |
| `--log-disable` | No logs |
| `--log-file FILE` | Log to a file |
| `--log-colors [on\|off\|auto]` | Color |
| `--log-prefix` / `--log-timestamps` | Prefixes and timestamps |
| `--log-jsonl / --no-log-jsonl` | JSONL logs (one JSON object per line) on stdout; disables color. Disabled |
| `--log-prompts-dir PATH` | Dumps prompts to disk. Debugging only |
| `--metrics` | Prometheus endpoint at `/metrics` |
| `--slots / --no-slots` | Slot monitoring endpoint. Enabled |
| `--props` | Allows changing global properties via `POST /props` |
| `--perf / --no-perf` | libllama internal performance metrics |
| `--warmup / --no-warmup` | Empty warm-up pass on load. Enabled |
| `--version` | Version and build information |

---

## 15. Model shortcuts

They load a well-known model, downloading it if needed. Use them on their own, without `-m` or `-hf`:

| Flag | Model |
|---|---|
| `--gpt-oss-20b-default` | gpt-oss 20B |
| `--gpt-oss-120b-default` | gpt-oss 120B (does not fit in 16 GB without `-cmoe`) |
| `--vision-gemma-4b-default` | Gemma 3 4B QAT, vision |
| `--vision-gemma-12b-default` | Gemma 3 12B QAT, vision |
| `--fim-qwen-1.5b-default` … `--fim-qwen-30b-default` | Qwen Coder, autocompletion |
| `--fim-qwen-7b-spec`, `--fim-qwen-14b-spec` | Qwen Coder + speculative draft |
| `--embd-gemma-default` | EmbeddingGemma |

Note: they pin a model, so they **disable router mode**.

---

## Recipes

All of these go in `.env`. To apply them: `docker compose up -d` from the project folder.


**Long context without running out of VRAM**

The KV cache is already `q8_0` by default, so just raise the context:
```
CTX_SIZE=65536
```

**A large MoE model that does not fit entirely**
```
EXTRA_ARGS=-cmoe --fit-target 512
```

**A DENSE model that does not fit entirely** (e.g. a 16 GB 27B Q4_K_M on 16 GB of VRAM)

`-cmoe` does not help here: a dense model has no experts to move out. What remains is to free up
VRAM by trimming the KV cache, so that more layers fit:

The `.env` defaults already do most of the work (`N_GPU_LAYERS=auto`, `CTX_SIZE=16384`,
KV cache in `q8_0`). To squeeze out a bit more:
```
FIT_TARGET=512
```

Measured with Qwen3.8-27B-UD-Q4_K_M on the reference machine (RTX 5070 Ti, 16 GB):

| | tok/s |
|---|---|
| `N_GPU_LAYERS=999` | fails to load |
| `auto`, native context (262144) | 5.7 |
| `auto`, `CTX_SIZE=16384`, KV in `q8_0` | 12.8 |
| + `--fit-target 512` | 13.7 |
| + `--fit-target 256` | 14.5 |

Lowering `--fit-target` below 512 MiB leaves very little headroom: if the GPU also drives the
desktop and Xorg requests memory, it can run out while the model is running.

Even so, that is 14 tok/s. With 16 GB of weights against 16 GB of VRAM some layers always stay on
the CPU. For a real jump, use a smaller quant that **fits entirely** (a Q3_K_M is around 13 GB) —
the difference between "almost everything on GPU" and "everything on GPU" is several-fold, not a
percentage.

**Keep the model from thinking forever**
```
EXTRA_ARGS=--reasoning-budget 2048
```

**Opening it to the home network, with minimal precautions**

In `compose.yaml` change the port to `"${PORT:-8080}:8080"` and in `.env`:
```
EXTRA_ARGS=--api-key a-long-key --cors-origins localhost
```

**Per-model configuration** — `models/models.ini`:
```ini
version = 1
[*]
c = 8192
cache-type-k = q8_0
cache-type-v = q8_0

[ggml-org/gpt-oss-20b-GGUF:Q4_K_M]
c = 32768
load-on-startup = true
```
```
EXTRA_ARGS=--models-preset /models/models.ini
```

**Metrics and debugging**
```
EXTRA_ARGS=--metrics -lv 4
```
