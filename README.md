# Qwen 3.8 Flash RTX 5090 bootstrap

This repository contains a one-command bootstrap for the validated Qwen 3.8 Flash production configuration on a fresh Ubuntu 24.04 RTX 5090 server. It installs build dependencies and CUDA Toolkit 13.2 without installing a driver, checks out the pinned `moe-cache` llama.cpp fork, downloads the three pinned UD-Q3_K_XL shards, and starts an authenticated OpenAI-compatible API.

## Requirements

- Ubuntu 24.04 x86_64, running as root
- RTX 5090 / compatible SM 12.0 Blackwell GPU with a working NVIDIA driver >= 580
- Nominal 64 GB RAM or more
- At least 130 GiB deployment capacity on the filesystem containing `/root/qwen38`
- Internet access to NVIDIA, GitHub, PyPI, and Hugging Face

The script never installs or replaces the NVIDIA driver. CUDA 13.x requires a driver from the 580+ family for minor-version compatibility; if the provider image has an older driver, update it through the provider before running the bootstrap.

Pinned inputs are `unsloth/Qwen3.8-Flash-Next-GGUF` revision `38bb39ee97821de2c9009abb7e93950eec396e66` and `GenerelSchwerz/llama.cpp` commit `b46f7f7a436f990932d3da3ec53380e2b9effc89`.

## Install

```bash
chmod +x bootstrap.sh
sudo ./bootstrap.sh
```

The download is pinned and resumable. Rerunning `bootstrap.sh` preserves the API key, reuses complete model shards, checks the source commit, and skips the build when its verified build stamp matches.

If the Hugging Face repository ever requires authentication, export `HF_TOKEN` and run `sudo --preserve-env=HF_TOKEN ./bootstrap.sh` (or run from an authenticated root shell). Do not put the token in this repository.

## Operate and recover

The control script uses systemd when systemd is genuinely available, otherwise a persistent tmux session named `qwen38-prod`:

```bash
/root/qwen38/start-qwen38.sh start
/root/qwen38/start-qwen38.sh stop
/root/qwen38/start-qwen38.sh restart
/root/qwen38/start-qwen38.sh status
/root/qwen38/start-qwen38.sh logs
```

If installation is interrupted, rerun `sudo ./bootstrap.sh`. Do not launch `llama-server` by hand: the control script stops only the managed binary/model instance, waits for it to release GPU memory, and prevents duplicate instances before starting.

Important paths:

- API key: `/root/qwen38/api-key.txt` (mode `0600`; never regenerated automatically)
- Log: `/root/qwen38/logs/qwen38-production.log`
- Model: `/root/qwen38/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf`
- Server runner: `/root/qwen38/run-qwen38.sh`

## API

The service binds to `0.0.0.0:11434`, requires the API key, and advertises model alias `qwen3.8-flash`.

```bash
API_KEY="$(sudo head -n1 /root/qwen38/api-key.txt)"
curl http://127.0.0.1:11434/v1/models \
  -H "Authorization: Bearer ${API_KEY}"
```

Provider-side port forwarding and host/cloud firewall rules are intentionally not modified. Configure those in Clore, Vast, or the cloud provider after bootstrap, and avoid exposing the API key publicly.

## Locked production profile

The runner keeps the validated profile unchanged: context `131072`, MoE cache `192`, threads `8`, one parallel slot, Q8_0 K/V cache, all GPU layers, flash attention on, MTP/speculation explicitly off, and model alias `qwen3.8-flash` on port `11434`.
