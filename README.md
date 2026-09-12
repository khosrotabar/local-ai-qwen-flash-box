# Qwen 3.8 Flash multi-VRAM bootstrap

This repository bootstraps the same Qwen 3.8 Flash Next `UD-Q3_K_XL` deployment for NVIDIA GPUs in 12 GB, 16 GB, 24 GB, 32 GB, and 48 GB VRAM classes. It keeps the pinned model and MoE-cache llama.cpp fork, derives the CUDA build architecture from the GPU's reported compute capability, and exposes an authenticated OpenAI-compatible API for DeepSeek Harness and other coding agents.

## Profile and context selection

On Ubuntu 24.04 x86_64, run as root with a working NVIDIA driver:

```bash
chmod +x bootstrap.sh
sudo ./bootstrap.sh auto auto
```

The first argument selects the VRAM profile. The optional second argument selects context: `auto`, `32`/`32k`, `64`/`64k`, `128`/`128k`, or `256`/`256k`. Omitting context is backward-compatible and means `auto`.

```bash
# 16 GB GPU, require exactly 64K context
sudo ./bootstrap.sh 16 64
sudo ./bootstrap.sh 16 64k

# 16 GB GPU, let the bootstrap choose context
sudo ./bootstrap.sh 16 auto

# 32 GB GPU, require exactly 128K or 256K
sudo ./bootstrap.sh 32 128
sudo ./bootstrap.sh 32 256

# 32 GB GPU, automatically find the highest safe context
sudo ./bootstrap.sh 32 auto

# Automatically detect the VRAM profile and context
sudo ./bootstrap.sh auto auto
```

Context aliases normalize to exact token counts:

| Context argument | Resolved request |
| --- | --- |
| `32` or `32k` | 32768 |
| `64` or `64k` | 65536 |
| `128` or `128k` | 131072 |
| `256` or `256k` | 262144 |

**An explicit context is never silently downgraded.** The bootstrap tunes only cache, allowed KV precision, and bounded batch settings around that exact context. If no candidate runs safely, it exits with suggested lower-context or auto commands.

An `auto` profile uses tolerant MiB thresholds matching real NVIDIA reports:

| Detected physical VRAM | Selected profile |
| --- | --- |
| 10752–14847 MiB | 12 GB |
| 14848–20991 MiB | 16 GB |
| 20992–29183 MiB | 24 GB |
| 29184–40959 MiB | 32 GB |
| 40960–57344 MiB | 48 GB |

VRAM outside those ranges—including 80 GB GPUs—fails instead of silently selecting an unsafe profile. A manual selection is checked against physical VRAM; `--force-profile` bypasses that check and the normal pre-launch threshold with a prominent warning. It is intended only for experts and cannot make an oversized configuration fit.

| Profile | Typical GPU | Auto-context order | KV | MoE cache | Status |
| --- | --- | --- | --- | --- | --- |
| 12 GB | RTX 5070 12 GB and similar | 64K, 32K, 24K, 16K, 12K, then existing 8K emergency fallback | Q4_0 | 48, 40, 32, 24, 16 | Experimental adaptive |
| 16 GB | RTX 5080 / RTX 5070 Ti / RTX 5060 Ti | 64K, 32K, 24K, 16K, 12K | Q8_0 first, then Q4_0 | 80, 72, 64, 56, 48, 40, 32 | Adaptive |
| 24 GB | RTX 3090, RTX 4090, L4, A30 | 128K, 64K, 32K | Q8_0 first, then Q4_0 | 144, 128, 112, 96, 80, 64 | Adaptive performance profile |
| 32 GB | RTX 5090 32 GB | 256K, 128K, 64K, 32K | Q8_0 first, then Q4_0 | 192, 160, 128, 96, 64 | Adaptive; locked fast path for known 128K setup |
| 48 GB | RTX A6000, RTX 6000 Ada, A40 | 256K, 128K, 64K, 32K | Q8_0 | 256, 240, 224, 208, 192, 176 | Adaptive high-VRAM profile |

Adaptive fitting iterates context first, then KV precision, then MoE cache. It therefore keeps a stable 64K configuration over a slightly faster 32K result. Each candidate must load, expose authenticated `/v1/models`, complete a real chat request with a 128-token output budget, remain free of CUDA OOMs, keep its process alive, and retain the profile's VRAM headroom. OOM signatures stop a candidate early, and the append-only tuning log records prompt/decode rates, VRAM use, and rejection reasons.

For the RTX 5080 16 GB case, `sudo ./bootstrap.sh 16 64 --retune` tests only `CTX_SIZE=65536`: Q8 KV before Q4, cache `80 72 64 56 48 40 32`, batch `2048`, and one bounded `1024` retry after CUDA OOM. It requires at least 1024 MiB free VRAM. If none pass, the command fails instead of trying 32K.

The 24 GB profile requires 1536 MiB free and uses batch 4096 with a bounded 2048 OOM retry. The 48 GB profile remains Q8, requires 2048 MiB free, and has the same bounded batch retry. All profiles use one parallel slot with MTP/speculation disabled.

The 24 GB and 48 GB strategies are adaptive and are not claimed to be universally benchmarked until tested on representative real GPUs. Capacity and speed are evaluated separately; no universal throughput cutoff rejects a legitimately slower GPU.

The validated 32 GB RTX 5090 configuration remains exactly context `131072`, MoE cache `192`, threads `8`, batch `4096`, ubatch `512`, one slot, Q8_0 K/V, all GPU layers, flash attention on, and MTP/speculation off. On matching RTX 5090 hardware, an explicit `32 128` request reuses those values without unnecessary fitting unless `--retune` is supplied. Requests such as `32 64`, `32 256`, and `32 auto` follow the requested fixed or auto-context strategy instead.

## Hardware and pinned inputs

- Ubuntu 24.04 x86_64
- A working NVIDIA driver from the 580+ family; the script does not install or replace the driver
- CUDA Toolkit 13.2 (installed toolkit-only if absent)
- A nominal 64 GB system RAM class (approximately 60 GiB Linux-visible RAM is accepted)
- 96 GB+ RAM strongly recommended for every adaptive profile
- At least 130 GiB deployment capacity, including verified reusable artifacts

This model places substantial pressure on system RAM even with 24 GB or 48 GB VRAM. Both RAM capacity and memory bandwidth can strongly affect load time and decode throughput. Hosts below the 96 GB recommendation are warned; genuinely lower-memory machines below the nominal 64 GB-class threshold fail. Swap is not created automatically.

Pinned artifacts:

- Model: `unsloth/Qwen3.8-Flash-Next-GGUF`
- Revision: `38bb39ee97821de2c9009abb7e93950eec396e66`
- Quant: `UD-Q3_K_XL` (the same three GGUF shards for every profile)
- llama.cpp fork: `https://github.com/GenerelSchwerz/llama.cpp.git`
- llama.cpp commit: `b46f7f7a436f990932d3da3ec53380e2b9effc89`

The script never substitutes a lower-quality quant. Downloads are pinned, resumable, and reused when their provenance stamps match. Architecture builds are reused when the source commit, CUDA version, CUDA architecture, and binary hash match. SM 12.0 keeps the validated `120a-real` strategy and existing `build-qwen38` location; other GPUs use conservative real builds such as `89-real` in `build-sm89` and `86-real` in `build-sm86`.

Bootstrap concurrency is protected by an atomic directory lock even on minimal images without `flock`. Once dependencies are present, the script also acquires the normal `flock` lock; users do not need to preinstall `util-linux` manually.

If Hugging Face authentication is required, export `HF_TOKEN` and use:

```bash
sudo --preserve-env=HF_TOKEN ./bootstrap.sh auto auto
```

Do not put the token in this repository.

## Fitted configuration and retuning

The resolved production configuration is stored at `/root/qwen38/profile.env`. It includes `CONTEXT_MODE` and normalized `REQUESTED_CONTEXT` alongside the resolved `CTX_SIZE`, selected profile, GPU identity, physical VRAM, compute capability, MoE cache, KV types, threads, batches, pinned revisions, CUDA build architecture, and binary/model paths.

Inspect it with:

```bash
sudo sed -n '1,240p' /root/qwen38/profile.env
sudo /root/qwen38/start-qwen38.sh status
```

Adaptive-fit details and rejection reasons are retained in `/root/qwen38/logs/qwen38-tuning.log`.

On rerun, a fit is reused only when the GPU name and UUID, VRAM, selected profile, context mode/request, compute capability, model revision, llama.cpp commit, and CUDA build architecture match. A cached `16 32` result cannot satisfy `16 64`. To repeat only fitting while preserving the downloaded model, verified build, and API key:

```bash
sudo ./bootstrap.sh 16 64 --retune
sudo ./bootstrap.sh 24 128 --retune
sudo ./bootstrap.sh 48 auto --retune
sudo ./bootstrap.sh auto auto --retune
```

## Dry run

Dry-run inspects local GPU/RAM information and prints the selected strategy without installing packages or CUDA, cloning/building llama.cpp, downloading the model, starting a server, acquiring deployment locks, or touching the API key:

```bash
sudo ./bootstrap.sh 16 64 --dry-run
sudo ./bootstrap.sh 32 256 --dry-run
sudo ./bootstrap.sh 32 auto --dry-run
sudo ./bootstrap.sh auto auto --dry-run
```

Dry-run shows the independently derived CMake CUDA architecture, context mode, normalized request, and exact context/KV/cache/batch/headroom strategy. It exits before locks, package or CUDA installation, downloads, builds, server startup, API-key handling, or persistent configuration writes.

## Operation, API, and recovery

The server uses systemd when it is genuinely available, otherwise a persistent tmux session named `qwen38-prod`. It survives SSH disconnects and uses one parallel slot for maximum single-agent performance.

```bash
sudo /root/qwen38/start-qwen38.sh start
sudo /root/qwen38/start-qwen38.sh stop
sudo /root/qwen38/start-qwen38.sh restart
sudo /root/qwen38/start-qwen38.sh status
sudo /root/qwen38/start-qwen38.sh logs
```

Before every tuning attempt and production start, the scripts stop only relevant Qwen llama-server processes, wait for their PIDs to leave NVIDIA's compute table, and verify that VRAM was released. They do not use a broad `pkill`, and they never run two fitting candidates simultaneously.

The API binds to `0.0.0.0:11434`, advertises `qwen3.8-flash`, and always requires the key stored at `/root/qwen38/api-key.txt` with mode `0600`. An existing key is never regenerated.

```bash
API_KEY="$(sudo head -n1 /root/qwen38/api-key.txt)"
curl http://127.0.0.1:11434/v1/models \
  -H "Authorization: Bearer ${API_KEY}"
```

Provider firewall and port-forwarding rules are intentionally not changed. Do not expose the API key publicly. If installation is interrupted, rerun the same bootstrap command: completed downloads, compatible builds, and the API key are preserved.
