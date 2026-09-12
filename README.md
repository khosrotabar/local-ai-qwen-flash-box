# Qwen 3.8 Flash multi-VRAM bootstrap

This repository bootstraps the same Qwen 3.8 Flash Next `UD-Q3_K_XL` deployment for NVIDIA GPUs in 12 GB, 16 GB, 24 GB, 32 GB, and 48 GB VRAM classes. It keeps the pinned model and MoE-cache llama.cpp fork, derives the CUDA build architecture from the GPU's reported compute capability, and exposes an authenticated OpenAI-compatible API for DeepSeek Harness and other coding agents.

## Quick start

On Ubuntu 24.04 x86_64, run as root with a working NVIDIA driver:

```bash
chmod +x bootstrap.sh
sudo ./bootstrap.sh auto
```

Or select the desired VRAM profile explicitly:

```bash
sudo ./bootstrap.sh 12
sudo ./bootstrap.sh 16
sudo ./bootstrap.sh 24
sudo ./bootstrap.sh 32
sudo ./bootstrap.sh 48
```

`auto` uses tolerant MiB thresholds matching real NVIDIA reports:

| Detected physical VRAM | Selected profile |
| --- | --- |
| 10752–14847 MiB | 12 GB |
| 14848–20991 MiB | 16 GB |
| 20992–29183 MiB | 24 GB |
| 29184–40959 MiB | 32 GB |
| 40960–57344 MiB | 48 GB |

VRAM outside those ranges—including 80 GB GPUs—fails instead of silently selecting an unsafe profile. A manual selection is checked against physical VRAM; `--force-profile` bypasses that check and the normal pre-launch threshold with a prominent warning. It is intended only for experts and cannot make an oversized configuration fit.

| Profile | Typical GPU | Context strategy | KV | MoE cache | Status |
| --- | --- | --- | --- | --- | --- |
| 12 GB | RTX 5070 12 GB and similar | Tries 16384, 12288, then 8192 | Q4_0 / Q4_0 | Tries 48, 40, 32, 24, 16 per context | Supported but experimental |
| 16 GB | RTX 5070 Ti / RTX 5060 Ti 16 GB | Tries 32768, 24576, 16384, then 12288 | Q4_0 / Q4_0 | Tries 80, 72, 64, 56, 48 per context | Adaptive; hardware-dependent |
| 24 GB | RTX 3090, RTX 4090, L4, A30 | Q8: 65536→24576; Q4 fallback: 65536→16384 | Q8_0 first, then Q4_0 | Tries 144, 128, 112, 96, 80, 64 | Adaptive performance profile |
| 32 GB | RTX 5090 32 GB | Locked at 131072 | Q8_0 / Q8_0 | Locked at 192 | Validated production configuration |
| 48 GB | RTX A6000, RTX 6000 Ada, A40 | Tries 262144, 196608, 131072, 98304, 65536 | Q8_0 / Q8_0 | Tries 256, 240, 224, 208, 192, 176 | Adaptive high-VRAM profile |

Adaptive fitting prioritizes context over small decode gains. Each candidate must load, expose authenticated `/v1/models`, complete a real chat request with a 128-token output budget, remain free of CUDA OOMs, and retain the profile's VRAM headroom. OOM signatures stop a candidate early, and the append-only tuning log records candidate boundaries and rejection reasons. The concise fit table records prompt and decode rates when reported plus used/free VRAM. It stops after the first safe context-priority result. The 24 GB profile requires 1536 MiB free and may retry an OOMed candidate at batch 2048 before sacrificing context; the 48 GB profile requires 2048 MiB free.

The 24 GB and 48 GB strategies are adaptive and are not claimed to be universally benchmarked until tested on representative real GPUs. Capacity and speed are evaluated separately; no universal throughput cutoff rejects a legitimately slower GPU.

The 32 GB profile is never auto-tuned. Its runtime remains exactly context `131072`, MoE cache `192`, threads `8`, batch `4096`, ubatch `512`, one slot, Q8_0 K/V, all GPU layers, flash attention on, and MTP/speculation off.

## Hardware and pinned inputs

- Ubuntu 24.04 x86_64
- A working NVIDIA driver from the 580+ family; the script does not install or replace the driver
- CUDA Toolkit 13.2 (installed toolkit-only if absent)
- At least nominal 64 GB system RAM
- 96 GB+ RAM strongly recommended for every adaptive profile
- At least 130 GiB deployment capacity, including verified reusable artifacts

This model places substantial pressure on system RAM even with 24 GB or 48 GB VRAM. Both RAM capacity and memory bandwidth can strongly affect load time and decode throughput. A 64–95 GB host is warned but not rejected for adaptive profiles; less than 64 GB fails. Swap is not created automatically.

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
sudo --preserve-env=HF_TOKEN ./bootstrap.sh auto
```

Do not put the token in this repository.

## Fitted configuration and retuning

The resolved production configuration is stored at `/root/qwen38/profile.env`. It includes the selected profile, GPU name and UUID, physical VRAM, compute capability, context, MoE cache, KV types, threads, batches, pinned revisions, CUDA build architecture, and binary/model paths.

Inspect it with:

```bash
sudo sed -n '1,240p' /root/qwen38/profile.env
sudo /root/qwen38/start-qwen38.sh status
```

Adaptive-fit details and rejection reasons are retained in `/root/qwen38/logs/qwen38-tuning.log`.

On rerun, an adaptive fit is reused only when the GPU name and UUID, VRAM, selected profile, compute capability, model revision, llama.cpp commit, and CUDA build architecture match. To repeat only adaptive fitting while preserving the downloaded model, verified build, and API key:

```bash
sudo ./bootstrap.sh 16 --retune
sudo ./bootstrap.sh 24 --retune
sudo ./bootstrap.sh 48 --retune
sudo ./bootstrap.sh auto --retune
```

## Dry run

Dry-run inspects local GPU/RAM information and prints the selected strategy without installing packages or CUDA, cloning/building llama.cpp, downloading the model, starting a server, acquiring deployment locks, or touching the API key:

```bash
sudo ./bootstrap.sh auto --dry-run
sudo ./bootstrap.sh 24 --dry-run
sudo ./bootstrap.sh 48 --dry-run
```

Dry-run also shows the independently derived CMake CUDA architecture and the exact context, KV, cache, batch, and headroom strategy for the selected profile.

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
