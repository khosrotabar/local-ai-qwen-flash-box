# Qwen 3.8 Flash multi-VRAM bootstrap

This repository bootstraps the same Qwen 3.8 Flash Next `UD-Q3_K_XL` deployment for NVIDIA GPUs in 12 GB, 16 GB, and 32 GB VRAM classes. It keeps the pinned model and MoE-cache llama.cpp fork, derives the CUDA build architecture from the GPU's reported compute capability, and exposes an authenticated OpenAI-compatible API for DeepSeek Harness and other coding agents.

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
sudo ./bootstrap.sh 32
```

`auto` maps 10.5–14.9 GiB to profile 12, 15–23.9 GiB to profile 16, and 30 GiB or more to profile 32. Unsupported gaps fail with an explanation. A manual selection is checked against physical VRAM; `--force-profile` bypasses that check and the normal 32 GB pre-launch threshold with a prominent warning. It is intended only for experts and cannot make an oversized configuration fit.

| Profile | Typical GPU | Context strategy | KV | MoE cache | Status |
| --- | --- | --- | --- | --- | --- |
| 32 GB | RTX 5090 32 GB | Locked at 131072 | Q8_0 / Q8_0 | Locked at 192 | Validated production configuration |
| 16 GB | RTX 5070 Ti / RTX 5060 Ti 16 GB | Tries 32768, 24576, 16384, then 12288 | Q4_0 / Q4_0 | Tries 80, 72, 64, 56, 48 per context | Adaptive; hardware-dependent |
| 12 GB | RTX 5070 12 GB and similar | Tries 16384, 12288, then 8192 | Q4_0 / Q4_0 | Tries 48, 40, 32, 24, 16 per context | Supported but experimental |

Adaptive fitting prioritizes context over small decode gains. Each candidate must load, expose authenticated `/v1/models`, complete a real chat request with a 128-token output budget, remain free of CUDA OOMs, and retain the profile's VRAM headroom. OOM signatures stop a candidate early, and the append-only tuning log records candidate boundaries and rejection reasons. The concise fit table records load, decode rate when reported, and used/free VRAM. It stops after the first safe context-priority result instead of sweeping every combination.

The 32 GB profile is never auto-tuned. Its runtime remains exactly context `131072`, MoE cache `192`, threads `8`, batch `4096`, ubatch `512`, one slot, Q8_0 K/V, all GPU layers, flash attention on, and MTP/speculation off.

## Hardware and pinned inputs

- Ubuntu 24.04 x86_64
- A working NVIDIA driver from the 580+ family; the script does not install or replace the driver
- CUDA Toolkit 13.2 (installed toolkit-only if absent)
- At least nominal 64 GB system RAM
- 96 GB+ RAM strongly recommended for 12/16 GB profiles
- At least 130 GiB deployment capacity, including verified reusable artifacts

Low-VRAM use moves substantial work and data pressure to system RAM. Both RAM capacity and memory bandwidth can strongly affect load time and decode throughput. A 64–95 GB host is warned but not rejected for 12/16 GB profiles; less than 64 GB fails. Swap is not created automatically.

Pinned artifacts:

- Model: `unsloth/Qwen3.8-Flash-Next-GGUF`
- Revision: `38bb39ee97821de2c9009abb7e93950eec396e66`
- Quant: `UD-Q3_K_XL` (the same three GGUF shards for every profile)
- llama.cpp fork: `https://github.com/GenerelSchwerz/llama.cpp.git`
- llama.cpp commit: `b46f7f7a436f990932d3da3ec53380e2b9effc89`

The script never substitutes a lower-quality quant. Downloads are pinned, resumable, and reused when their provenance stamps match. Architecture builds are reused when the source commit, CUDA version, CUDA architecture, and binary hash match. SM 12.0 keeps the validated `120a-real` strategy and existing `build-qwen38` location; other GPUs use conservative real builds such as `89-real` in `build-sm89` and `86-real` in `build-sm86`.

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

On rerun, a 12/16 GB fit is reused only when the GPU name and UUID, VRAM, selected profile, compute capability, model revision, llama.cpp commit, and CUDA build architecture match. To repeat only adaptive fitting while preserving the downloaded model, verified build, and API key:

```bash
sudo ./bootstrap.sh 16 --retune
sudo ./bootstrap.sh auto --retune
```

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
