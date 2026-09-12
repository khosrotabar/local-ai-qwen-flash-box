#!/usr/bin/env bash
# Portable bootstrap for Qwen3.8-Flash-Next UD-Q3_K_XL on 12/16/24/32/48 GB NVIDIA GPUs.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly QWEN_ROOT="/root/qwen38"
readonly LLAMA_DIR="${QWEN_ROOT}/llama.cpp"
readonly MODEL_ROOT="${QWEN_ROOT}/models/Qwen3.8-Flash-Next-GGUF"
readonly MODEL_DIR="${MODEL_ROOT}/UD-Q3_K_XL"
readonly MODEL_NAME="Qwen3.8-Flash-Next-UD-Q3_K_XL"
readonly MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}-00001-of-00003.gguf"
readonly API_KEY_FILE="${QWEN_ROOT}/api-key.txt"
readonly PROFILE_FILE="${QWEN_ROOT}/profile.env"
readonly LOG_DIR="${QWEN_ROOT}/logs"
readonly LOG_FILE="${LOG_DIR}/qwen38-production.log"
readonly TUNE_LOG="${LOG_DIR}/qwen38-tuning.log"
readonly RUNNER="${QWEN_ROOT}/run-qwen38.sh"
readonly TMUX_RUNNER="${QWEN_ROOT}/run-qwen38-tmux.sh"
readonly CONTROL_SCRIPT="${QWEN_ROOT}/start-qwen38.sh"
readonly MODE_FILE="${QWEN_ROOT}/supervisor-mode"
readonly MODEL_STAMP="${MODEL_ROOT}/.qwen38-model-revision"
readonly ACTIVE_DOWNLOAD_STAMP="${MODEL_ROOT}/.qwen38-active-download"
readonly SERVICE_NAME="qwen38.service"
readonly SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
readonly TMUX_SESSION="qwen38-prod"

readonly LLAMA_REPO="https://github.com/GenerelSchwerz/llama.cpp.git"
readonly LLAMA_COMMIT="b46f7f7a436f990932d3da3ec53380e2b9effc89"
readonly HF_REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
readonly HF_REVISION="38bb39ee97821de2c9009abb7e93950eec396e66"
readonly -a SHARDS=(
    "UD-Q3_K_XL/${MODEL_NAME}-00001-of-00003.gguf"
    "UD-Q3_K_XL/${MODEL_NAME}-00002-of-00003.gguf"
    "UD-Q3_K_XL/${MODEL_NAME}-00003-of-00003.gguf"
)
readonly CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
readonly MIN_DRIVER_VERSION="580.0"
readonly MIN_RAM_KIB=$((64000000000 / 1024))
readonly RECOMMENDED_RAM_KIB=$((96000000000 / 1024))
readonly MIN_CAPACITY_KIB=$((130 * 1024 * 1024))
readonly HEALTH_TIMEOUT_SECONDS=1800
readonly BOOTSTRAP_LOCK_DIR="/var/lock/qwen38-bootstrap.lock.d"

STAGE="initialization"
TEMP_FILES=()
TUNING_PID=""
PRODUCTION_STARTED=false
DEPLOYMENT_COMPLETE=false
REQUESTED_PROFILE=""
RETUNE=false
FORCE_PROFILE=false
DRY_RUN=false
BOOTSTRAP_LOCK_HELD=false
FLOCK_HELD=false

usage() {
    cat <<'USAGE'
Usage: sudo ./bootstrap.sh {12|16|24|32|48|auto} [--retune] [--force-profile] [--dry-run]

VRAM profiles:
  12       Experimental low-VRAM adaptive profile
  16       Balanced adaptive profile
  24       Q8-first adaptive performance profile
  32       Locked, validated RTX 5090-class production profile
  48       Q8 adaptive high-VRAM profile
  auto     Select from tolerant physical-VRAM bands (10.5 through 56 GiB)

Options:
  --retune        Refit an adaptive profile; model, build, and API key are reused
  --force-profile Override the manual profile VRAM safety check (expert use only)
  --dry-run        Inspect hardware and print strategy without making changes
  -h, --help      Show this help
USAGE
}

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { printf 'BOOTSTRAP_FATAL_ERROR [%s]: %s\n' "${STAGE}" "$*" >&2; exit 1; }

cleanup() {
    local path
    if [[ -n "${TUNING_PID}" ]] && kill -0 "${TUNING_PID}" 2>/dev/null; then
        kill -TERM "${TUNING_PID}" 2>/dev/null || true
        wait "${TUNING_PID}" 2>/dev/null || true
    fi
    if [[ "${PRODUCTION_STARTED}" == true && "${DEPLOYMENT_COMPLETE}" != true && -x "${CONTROL_SCRIPT}" ]]; then
        PRODUCTION_STARTED=false
        "${CONTROL_SCRIPT}" stop >/dev/null 2>&1 || true
    fi
    if [[ "${BOOTSTRAP_LOCK_HELD}" == true ]]; then
        rm -f -- "${BOOTSTRAP_LOCK_DIR}/pid"
        rmdir -- "${BOOTSTRAP_LOCK_DIR}" 2>/dev/null || true
    fi
    for path in "${TEMP_FILES[@]:-}"; do
        if [[ -n "${path}" && "${path}" == /tmp/qwen38.* ]]; then
            rm -f -- "${path}"
        fi
    done
}

on_error() {
    local exit_code="$1" line="$2" command="$3"
    printf '\nBOOTSTRAP FAILED\n  stage: %s\n  line: %s\n  command: %s\n  exit: %s\n' \
        "${STAGE}" "${line}" "${command}" "${exit_code}" >&2
    printf 'Correct the reported problem and rerun; verified downloads and builds are reusable.\n' >&2
}

trap cleanup EXIT
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

stage() { STAGE="$2"; printf '\n[%s/10] %s\n' "$1" "$2"; }
require_root() { [[ "${EUID}" -eq 0 ]] || die "Run this script as root (sudo ./bootstrap.sh ...)."; }
version_ge() { dpkg --compare-versions "$1" ge "$2"; }
systemd_usable() { [[ -d /run/systemd/system ]] && systemctl show-environment >/dev/null 2>&1; }

atomic_write() {
    local destination="$1" mode="$2" temporary
    temporary="$(mktemp /tmp/qwen38.XXXXXX)"
    TEMP_FILES+=("${temporary}")
    cat >"${temporary}"
    install -o root -g root -m "${mode}" "${temporary}" "${destination}"
}

acquire_bootstrap_lock() {
    local owner="unknown"
    if ! mkdir -- "${BOOTSTRAP_LOCK_DIR}" 2>/dev/null; then
        [[ -r "${BOOTSTRAP_LOCK_DIR}/pid" ]] && owner="$(<"${BOOTSTRAP_LOCK_DIR}/pid")"
        die "Another qwen38 bootstrap may be running (bootstrap lock owner: ${owner}). If that process no longer exists, remove ${BOOTSTRAP_LOCK_DIR} and rerun."
    fi
    BOOTSTRAP_LOCK_HELD=true
    printf '%s\n' "$$" >"${BOOTSTRAP_LOCK_DIR}/pid"
    if command -v flock >/dev/null 2>&1; then
        exec 9>/var/lock/qwen38-bootstrap.lock
        flock -n 9 || die "Another qwen38 bootstrap holds the flock lock."
        FLOCK_HELD=true
    fi
}

acquire_flock_after_dependencies() {
    [[ "${FLOCK_HELD}" == false ]] || return 0
    command -v flock >/dev/null 2>&1 || die "util-linux installation completed without flock."
    exec 9>/var/lock/qwen38-bootstrap.lock
    flock -n 9 || die "Another qwen38 bootstrap holds the flock lock."
    FLOCK_HELD=true
}

print_array_values() {
    local value first=true
    for value in "$@"; do
        [[ "${first}" == true ]] || printf ' '
        printf '%s' "${value}"
        first=false
    done
    printf '\n'
}

print_dry_run_strategy() {
    printf '\nProfile strategy (%s GB):\n' "${PROFILE}"
    if [[ "${PROFILE}" == 32 ]]; then
        printf '  Mode: locked validated configuration (no adaptive fitting)\n'
        printf '  Context: %s\n  MoE cache: %s\n  KV: %s / %s\n' "${CTX_SIZE}" "${MOE_CACHE}" "${KV_K^^}" "${KV_V^^}"
        printf '  Threads/batch/ubatch: %s / %s / %s\n' "${THREADS}" "${BATCH}" "${UBATCH}"
    else
        if [[ "${PROFILE}" == 24 ]]; then
            printf '  Q8 context candidates: '; print_array_values "${Q8_CONTEXT_CANDIDATES[@]}"
            printf '  Q4 fallback contexts:  '; print_array_values "${Q4_CONTEXT_CANDIDATES[@]}"
            printf '  KV strategy: Q8_0 first, Q4_0 fallback\n'
        else
            printf '  Context candidates: '; print_array_values "${CONTEXT_CANDIDATES[@]}"
            printf '  KV strategy: '; print_array_values "${KV_CANDIDATES[@]}"
        fi
        printf '  MoE cache candidates: '; print_array_values "${CACHE_CANDIDATES[@]}"
        printf '  Batch/ubatch: %s / %s\n' "${BATCH}" "${UBATCH}"
        (( BATCH_FALLBACK > 0 )) && printf '  OOM batch fallback: %s\n' "${BATCH_FALLBACK}"
        printf '  Required VRAM headroom: %s MiB\n' "${HEADROOM_MIB}"
    fi
    printf '\nNo changes made.\n'
}

parse_cli() {
    local argument
    (($#)) || { usage >&2; exit 64; }
    for argument in "$@"; do
        case "${argument}" in
            12|16|24|32|48|auto)
                [[ -z "${REQUESTED_PROFILE}" ]] || die "Specify exactly one VRAM profile."
                REQUESTED_PROFILE="${argument}"
                ;;
            --retune) RETUNE=true ;;
            --force-profile) FORCE_PROFILE=true ;;
            --dry-run) DRY_RUN=true ;;
            -h|--help) usage; exit 0 ;;
            *) usage >&2; die "Unknown argument: ${argument}" ;;
        esac
    done
    [[ -n "${REQUESTED_PROFILE}" ]] || die "A VRAM profile (12, 16, 24, 32, 48, or auto) is required."
    [[ "${REQUESTED_PROFILE}" != auto || "${FORCE_PROFILE}" == false ]] || \
        die "--force-profile is only meaningful with an explicit profile."
}

choose_profile() {
    local total_mib="$1"
    if [[ "${REQUESTED_PROFILE}" != auto ]]; then PROFILE="${REQUESTED_PROFILE}"; return; fi
    if (( total_mib >= 10752 && total_mib < 14848 )); then
        PROFILE=12
    elif (( total_mib >= 14848 && total_mib < 20992 )); then
        PROFILE=16
    elif (( total_mib >= 20992 && total_mib < 29184 )); then
        PROFILE=24
    elif (( total_mib >= 29184 && total_mib < 40960 )); then
        PROFILE=32
    elif (( total_mib >= 40960 && total_mib <= 57344 )); then
        PROFILE=48
    else
        die "Auto-selection does not support ${total_mib} MiB VRAM. Supported bands are 10752-14847 MiB (12), 14848-20991 (16), 20992-29183 (24), 29184-40959 (32), and 40960-57344 (48)."
    fi
}

verify_manual_profile_capacity() {
    local minimum_mib
    [[ "${REQUESTED_PROFILE}" != auto ]] || return 0
    case "${PROFILE}" in
        12) minimum_mib=10752 ;;
        16) minimum_mib=14848 ;;
        24) minimum_mib=20992 ;;
        32) minimum_mib=29184 ;;
        48) minimum_mib=40960 ;;
    esac
    if (( GPU_TOTAL_MIB < minimum_mib )); then
        if [[ "${FORCE_PROFILE}" == true ]]; then
            printf '\n*** DANGER: FORCING %s GB PROFILE ON A %s MiB GPU ***\n' "${PROFILE}" "${GPU_TOTAL_MIB}" >&2
            printf 'This can cause CUDA OOM, severe instability, or an unusable deployment.\n\n' >&2
        else
            die "Profile ${PROFILE} requires at least ${minimum_mib} MiB physical VRAM; GPU 0 reports ${GPU_TOTAL_MIB} MiB. Use --force-profile only for an expert override."
        fi
    fi
}

derive_cuda_architecture() {
    local compact major minor
    [[ "${GPU_CC}" =~ ^([0-9]+)\.([0-9]+)$ ]] || \
        die "Cannot derive a safe CUDA architecture from compute capability '${GPU_CC:-unknown}'."
    major="${GPU_CC%%.*}"
    minor="${GPU_CC#*.}"
    compact="${major}${minor}"
    CUDA_ARCH_CODE="${compact}"
    if [[ "${compact}" == 120 ]]; then
        CMAKE_CUDA_ARCH="120a-real"
        BUILD_DIR="${LLAMA_DIR}/build-qwen38"
    else
        CMAKE_CUDA_ARCH="${compact}-real"
        BUILD_DIR="${LLAMA_DIR}/build-sm${compact}"
    fi
    SERVER_BIN="${BUILD_DIR}/bin/llama-server"
    BUILD_STAMP="${BUILD_DIR}/.qwen38-build-stamp"
}

derive_threads() {
    local cpu_count="$(nproc)"
    if (( cpu_count >= 8 )); then THREADS=8; else THREADS="${cpu_count}"; fi
    (( THREADS >= 1 )) || THREADS=1
}

load_profile_defaults() {
    CONTEXT_CANDIDATES=(); Q8_CONTEXT_CANDIDATES=(); Q4_CONTEXT_CANDIDATES=()
    CACHE_CANDIDATES=(); KV_CANDIDATES=(); BATCH_FALLBACK=0
    PARALLEL=1; SPEC_TYPE=none
    FORCED_PROFILE="${FORCE_PROFILE}"
    case "${PROFILE}" in
        32)
            CTX_SIZE=131072; MOE_CACHE=192; KV_K=q8_0; KV_V=q8_0
            THREADS=8; BATCH=4096; UBATCH=512; HEADROOM_MIB=1024
            KV_CANDIDATES=(q8_0)
            ;;
        16)
            CONTEXT_CANDIDATES=(32768 24576 16384 12288)
            CACHE_CANDIDATES=(80 72 64 56 48)
            KV_CANDIDATES=(q4_0)
            KV_K=q4_0; KV_V=q4_0; BATCH=2048; UBATCH=512; HEADROOM_MIB=1024
            derive_threads
            ;;
        24)
            Q8_CONTEXT_CANDIDATES=(65536 49152 32768 24576)
            Q4_CONTEXT_CANDIDATES=(65536 49152 32768 24576 16384)
            CONTEXT_CANDIDATES=("${Q8_CONTEXT_CANDIDATES[@]}")
            CACHE_CANDIDATES=(144 128 112 96 80 64)
            KV_CANDIDATES=(q8_0 q4_0)
            KV_K=q8_0; KV_V=q8_0; BATCH=4096; BATCH_FALLBACK=2048; UBATCH=512; HEADROOM_MIB=1536
            derive_threads
            ;;
        12)
            CONTEXT_CANDIDATES=(16384 12288 8192)
            CACHE_CANDIDATES=(48 40 32 24 16)
            KV_CANDIDATES=(q4_0)
            KV_K=q4_0; KV_V=q4_0; BATCH=1024; UBATCH=256; HEADROOM_MIB=768
            derive_threads
            ;;
        48)
            CONTEXT_CANDIDATES=(262144 196608 131072 98304 65536)
            # The pinned fork documents rejection of negative sizes but no
            # fixed positive maximum. Treat 256 as an unproven upper candidate;
            # model-specific rejection remains an expected adaptive miss.
            CACHE_CANDIDATES=(256 240 224 208 192 176)
            KV_CANDIDATES=(q8_0)
            KV_K=q8_0; KV_V=q8_0; BATCH=4096; UBATCH=512; HEADROOM_MIB=2048
            derive_threads
            ;;
    esac
}

relevant_pids() {
    local proc pid executable commandline process_name
    for proc in /proc/[0-9]*; do
        pid="${proc#/proc/}"
        [[ "${pid}" != "$$" && -r "${proc}/cmdline" ]] || continue
        executable="$(readlink -f "${proc}/exe" 2>/dev/null || true)"
        commandline="$(tr '\0' ' ' <"${proc}/cmdline" 2>/dev/null || true)"
        process_name="$(cat "${proc}/comm" 2>/dev/null || true)"
        [[ "${process_name}" == llama-server || "${executable}" == */llama-server* ]] || continue
        if [[ "${commandline}" == *"${MODEL_PATH}"* || "${commandline}" == *qwen3.8-flash* || \
              "${commandline}" == *"--port 11434"* || "${commandline}" == *"--port=11434"* ]]; then
            printf '%s\n' "${pid}"
        fi
    done
}

stop_installed_managers() {
    if systemd_usable && systemctl cat "${SERVICE_NAME}" >/dev/null 2>&1; then systemctl stop "${SERVICE_NAME}" || true; fi
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then tmux kill-session -t "${TMUX_SESSION}"; fi
}

wait_for_gpu_pid_release() {
    local target_pid="$1" deadline=$((SECONDS + 30)) gpu_pid
    local -a gpu_pids=()
    [[ -n "${target_pid}" ]] || return 0
    while (( SECONDS < deadline )); do
        mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
        for gpu_pid in "${gpu_pids[@]}"; do
            [[ "${target_pid}" == "${gpu_pid//[[:space:]]/}" ]] && { sleep 1; continue 2; }
        done
        return 0
    done
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true
    return 1
}

stop_relevant_servers() {
    local deadline pid gpu_pid still_owned=false
    local -a pids=() original_pids=() gpu_pids=()
    mapfile -t pids < <(relevant_pids)
    ((${#pids[@]})) || return 0
    original_pids=("${pids[@]}")
    log "Stopping relevant llama-server PID(s): ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
        mapfile -t pids < <(relevant_pids); ((${#pids[@]} == 0)) && break; sleep 1
    done
    mapfile -t pids < <(relevant_pids)
    if ((${#pids[@]})); then
        log "Graceful stop timed out; force-stopping resolved PID(s): ${pids[*]}"
        kill -KILL "${pids[@]}" 2>/dev/null || true; sleep 2
    fi
    mapfile -t pids < <(relevant_pids)
    ((${#pids[@]} == 0)) || die "Relevant llama-server PID(s) survived shutdown: ${pids[*]}"
    deadline=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
        still_owned=false
        for pid in "${original_pids[@]}"; do
            for gpu_pid in "${gpu_pids[@]}"; do
                [[ "${pid}" == "${gpu_pid//[[:space:]]/}" ]] && { still_owned=true; break 2; }
            done
        done
        [[ "${still_owned}" == false ]] && return 0
        sleep 1
    done
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true
    die "A stopped llama-server still owns GPU memory after 30 seconds."
}

prepare_clean_gpu() {
    local free_mib minimum_free port_owner
    local -a stale=()
    stop_installed_managers
    stop_relevant_servers
    mapfile -t stale < <(relevant_pids)
    ((${#stale[@]} == 0)) || die "A relevant llama-server remains before launch: ${stale[*]}"
    port_owner="$(ss -H -ltnp 'sport = :11434' 2>/dev/null || true)"
    [[ -z "${port_owner}" ]] || die "TCP port 11434 is owned by an unrelated listener: ${port_owner}"
    if [[ "${PROFILE}" == 32 && "${FORCED_PROFILE}" != true ]]; then
        minimum_free=30000
        (( GPU_TOTAL_MIB - 2048 < minimum_free )) && minimum_free=$((GPU_TOTAL_MIB - 2048))
    else
        minimum_free=$((GPU_TOTAL_MIB - 2048))
    fi
    (( minimum_free < 0 )) && minimum_free=0
    free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | xargs)"
    if (( free_mib < minimum_free )); then
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true
        die "Only ${free_mib} MiB of ${GPU_TOTAL_MIB} MiB VRAM is free before launch. Stop unrelated GPU workloads first."
    fi
}

server_arguments() {
    local context="$1" cache="$2" kv_k="$3" kv_v="$4" threads="$5" batch="$6" ubatch="$7"
    SERVER_ARGS=(
        --offline --model "${MODEL_PATH}" --spec-type none
        -c "${context}" -b "${batch}" -ub "${ubatch}" -np 1 -t "${threads}"
        -ngl all -fa on -fit off --load-mode none --lazy-mode on
        --moe-expert-cache-size "${cache}" -ctk "${kv_k}" -ctv "${kv_v}" -kvo
        --cache-ram 0 --jinja --no-warmup --experimental-logs
        --alias qwen3.8-flash --api-key-file "${API_KEY_FILE}" --host 0.0.0.0 --port 11434
    )
}

log_slice_has_oom() {
    local file="$1" start_line="$2"
    tail -n "+${start_line}" "${file}" | \
        grep -Ei 'CUDA.*out of memory|out of memory.*CUDA|CUDA error.*memory' >/dev/null
}

profile_value() {
    local key="$1" file="$2"
    /usr/bin/env -i PATH=/usr/bin:/bin /bin/bash -c 'source "$1"; printf "%s" "${!2-}"' bash "${file}" "${key}"
}

cached_profile_matches() {
    [[ "${RETUNE}" == false && -f "${PROFILE_FILE}" && ! -L "${PROFILE_FILE}" ]] || return 1
    [[ "$(profile_value PROFILE "${PROFILE_FILE}")" == "${PROFILE}" ]] || return 1
    [[ "$(profile_value GPU_NAME "${PROFILE_FILE}")" == "${GPU_NAME}" ]] || return 1
    [[ "$(profile_value GPU_UUID "${PROFILE_FILE}")" == "${GPU_UUID}" ]] || return 1
    [[ "$(profile_value COMPUTE_CAPABILITY "${PROFILE_FILE}")" == "${GPU_CC}" ]] || return 1
    [[ "$(profile_value VRAM_MB "${PROFILE_FILE}")" == "${GPU_TOTAL_MIB}" ]] || return 1
    [[ "$(profile_value LLAMA_COMMIT "${PROFILE_FILE}")" == "${LLAMA_COMMIT}" ]] || return 1
    [[ "$(profile_value MODEL_REVISION "${PROFILE_FILE}")" == "${HF_REVISION}" ]] || return 1
    [[ "$(profile_value CMAKE_CUDA_ARCHITECTURE "${PROFILE_FILE}")" == "${CMAKE_CUDA_ARCH}" ]] || return 1
}

load_cached_profile() {
    CTX_SIZE="$(profile_value CTX_SIZE "${PROFILE_FILE}")"
    MOE_CACHE="$(profile_value MOE_CACHE "${PROFILE_FILE}")"
    KV_K="$(profile_value KV_K "${PROFILE_FILE}")"; KV_V="$(profile_value KV_V "${PROFILE_FILE}")"
    THREADS="$(profile_value THREADS "${PROFILE_FILE}")"
    BATCH="$(profile_value BATCH "${PROFILE_FILE}")"; UBATCH="$(profile_value UBATCH "${PROFILE_FILE}")"
    for numeric in CTX_SIZE MOE_CACHE THREADS BATCH UBATCH; do
        [[ "${!numeric}" =~ ^[0-9]+$ ]] || die "Cached profile has invalid ${numeric}; rerun with --retune."
    done
    [[ "${KV_K}" =~ ^q[48]_0$ && "${KV_V}" =~ ^q[48]_0$ ]] || die "Cached profile has invalid KV types; rerun with --retune."
}

write_profile() {
    local temporary
    if [[ -L "${PROFILE_FILE}" || ( -e "${PROFILE_FILE}" && ! -f "${PROFILE_FILE}" ) ]]; then
        die "Refusing to replace non-regular profile path ${PROFILE_FILE}."
    fi
    temporary="$(mktemp /tmp/qwen38.profile.XXXXXX)"
    TEMP_FILES+=("${temporary}")
    {
        printf '# Generated by the Qwen 3.8 Flash bootstrap.\n'
        printf 'PROFILE=%q\n' "${PROFILE}"; printf 'GPU_NAME=%q\n' "${GPU_NAME}"
        printf 'GPU_UUID=%q\n' "${GPU_UUID}"; printf 'COMPUTE_CAPABILITY=%q\n' "${GPU_CC}"
        printf 'VRAM_MB=%q\n' "${GPU_TOTAL_MIB}"; printf 'CTX_SIZE=%q\n' "${CTX_SIZE}"
        printf 'MOE_CACHE=%q\n' "${MOE_CACHE}"; printf 'KV_K=%q\n' "${KV_K}"; printf 'KV_V=%q\n' "${KV_V}"
        printf 'THREADS=%q\n' "${THREADS}"; printf 'BATCH=%q\n' "${BATCH}"; printf 'UBATCH=%q\n' "${UBATCH}"
        printf 'PARALLEL=%q\n' "${PARALLEL}"; printf 'SPEC_TYPE=%q\n' "${SPEC_TYPE}"
        printf 'FORCED_PROFILE=%q\n' "${FORCED_PROFILE}"
        printf 'HEADROOM_MIB=%q\n' "${HEADROOM_MIB}"; printf 'LLAMA_COMMIT=%q\n' "${LLAMA_COMMIT}"
        printf 'MODEL_REVISION=%q\n' "${HF_REVISION}"; printf 'CMAKE_CUDA_ARCHITECTURE=%q\n' "${CMAKE_CUDA_ARCH}"
        printf 'SERVER_BIN=%q\n' "${SERVER_BIN}"; printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    } >"${temporary}"
    install -o root -g root -m 600 "${temporary}" "${PROFILE_FILE}"
}

wait_for_candidate_health() {
    local response_file="$1" auth_file="$2" log_start_line="$3" deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        if log_slice_has_oom "${TUNE_LOG}" "${log_start_line}"; then
            CANDIDATE_REASON=CUDA_OOM
            return 1
        fi
        if ! kill -0 "${TUNING_PID}" 2>/dev/null; then
            CANDIDATE_REASON=LOAD_FAILED
            return 1
        fi
        if curl --silent --show-error --fail --max-time 10 --header @"${auth_file}" \
            --output "${response_file}" http://127.0.0.1:11434/v1/models 2>/dev/null && \
           jq -e '.data[]? | select(.id == "qwen3.8-flash")' "${response_file}" >/dev/null 2>&1; then return 0; fi
        sleep 5
    done
    CANDIDATE_REASON=HEALTH_TIMEOUT
    return 1
}

stop_tuning_candidate() {
    local deadline candidate_pid="${TUNING_PID}"
    if [[ -n "${TUNING_PID}" ]] && kill -0 "${TUNING_PID}" 2>/dev/null; then
        kill -TERM "${TUNING_PID}" 2>/dev/null || true; deadline=$((SECONDS + 60))
        while kill -0 "${TUNING_PID}" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
        kill -KILL "${TUNING_PID}" 2>/dev/null || true
    fi
    [[ -z "${TUNING_PID}" ]] || wait "${TUNING_PID}" 2>/dev/null || true
    TUNING_PID=""
    wait_for_gpu_pid_release "${candidate_pid}" || die "Tuning candidate PID ${candidate_pid} still owns GPU memory after shutdown."
    stop_relevant_servers
}

try_candidate() {
    local context="$1" cache="$2" kv="$3" candidate_batch="$4" auth_file="$5" models_file="$6" chat_file="$7"
    local free_mib=- used_mib=- prompt_rate=- decode=- load=FAIL result=EXPECTED_PROFILE_MISS candidate_log_start_line
    local payload='{"model":"qwen3.8-flash","messages":[{"role":"user","content":"Reply with exactly the word READY."}],"temperature":0,"max_tokens":128}'
    prepare_clean_gpu
    printf '\n[%s] candidate profile=%s context=%s cache=%s kv=%s batch=%s\n' \
        "$(date --iso-8601=seconds)" "${PROFILE}" "${context}" "${cache}" "${kv}" "${candidate_batch}" >>"${TUNE_LOG}"
    candidate_log_start_line=$(( $(wc -l <"${TUNE_LOG}") + 1 ))
    CANDIDATE_REASON=STARTUP_FAILED
    server_arguments "${context}" "${cache}" "${kv}" "${kv}" "${THREADS}" "${candidate_batch}" "${UBATCH}"
    CUDA_VISIBLE_DEVICES=0 LLAMA_ATTN_ROT_DISABLE=1 COMPILE_OFF=1 \
        "${SERVER_BIN}" "${SERVER_ARGS[@]}" >>"${TUNE_LOG}" 2>&1 &
    TUNING_PID=$!
    if wait_for_candidate_health "${models_file}" "${auth_file}" "${candidate_log_start_line}"; then
        load=OK
        CANDIDATE_REASON=INFERENCE_FAILED
        if curl --silent --show-error --fail --max-time 600 --header @"${auth_file}" --header 'Content-Type: application/json' \
            --data "${payload}" --output "${chat_file}" http://127.0.0.1:11434/v1/chat/completions 2>/dev/null && \
           jq -e '(.choices | length > 0) and (((.choices[0].message.content // "") | length > 0) or ((.choices[0].message.reasoning_content // "") | length > 0))' "${chat_file}" >/dev/null 2>&1 && \
           kill -0 "${TUNING_PID}" 2>/dev/null && \
           ! log_slice_has_oom "${TUNE_LOG}" "${candidate_log_start_line}"; then
            prompt_rate="$(jq -r '.timings.prompt_per_second // empty' "${chat_file}" 2>/dev/null || true)"; [[ -n "${prompt_rate}" ]] || prompt_rate=-
            decode="$(jq -r '.timings.predicted_per_second // empty' "${chat_file}" 2>/dev/null || true)"; [[ -n "${decode}" ]] || decode=OK
            free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | xargs)"
            used_mib="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | xargs)"
            if (( free_mib >= HEADROOM_MIB )); then
                result=OK
                CANDIDATE_REASON=OK
            else
                CANDIDATE_REASON="LOW_HEADROOM_${free_mib}_MIB"
            fi
        elif log_slice_has_oom "${TUNE_LOG}" "${candidate_log_start_line}"; then
            CANDIDATE_REASON=CUDA_OOM
        fi
    fi
    printf '%-8s %-7s %-7s %-7s %-8s %-10s %-10s %-13s %-10s\n' \
        "${context}" "${cache}" "${kv^^}" "${candidate_batch}" "${load}" "${prompt_rate}" "${decode}" "${used_mib}" "${free_mib}"
    printf '[%s] RESULT context=%s cache=%s kv=%s batch=%s load=%s prompt_tps=%s decode_tps=%s vram_used=%s vram_free=%s reason=%s\n' \
        "$(date --iso-8601=seconds)" "${context}" "${cache}" "${kv}" "${candidate_batch}" "${load}" \
        "${prompt_rate}" "${decode}" "${used_mib}" "${free_mib}" "${CANDIDATE_REASON}" >>"${TUNE_LOG}"
    stop_tuning_candidate
    [[ "${result}" == OK ]]
}

fit_adaptive_profile() {
    local context cache kv candidate_batch primary_batch auth_file models_file chat_file
    local -a active_contexts=()
    auth_file="$(mktemp /tmp/qwen38.auth.XXXXXX)"; models_file="$(mktemp /tmp/qwen38.models.XXXXXX.json)"; chat_file="$(mktemp /tmp/qwen38.chat.XXXXXX.json)"
    TEMP_FILES+=("${auth_file}" "${models_file}" "${chat_file}"); chmod 600 "${auth_file}" "${models_file}" "${chat_file}"
    printf 'Authorization: Bearer %s\n' "${API_KEY_VALUE}" >"${auth_file}"
    printf '\n[%s] starting adaptive fit for profile %s on %s (%s)\n' \
        "$(date --iso-8601=seconds)" "${PROFILE}" "${GPU_NAME}" "${GPU_UUID}" >>"${TUNE_LOG}"
    printf '\nAdaptive fit (first context-priority candidate with >= %s MiB free wins):\n' "${HEADROOM_MIB}"
    printf '%-8s %-7s %-7s %-7s %-8s %-10s %-10s %-13s %-10s\n' \
        CTX CACHE KV BATCH LOAD PROMPT_TPS DECODE_TPS VRAM_USED VRAM_FREE
    primary_batch="${BATCH}"
    for kv in "${KV_CANDIDATES[@]}"; do
        if [[ "${PROFILE}" == 24 && "${kv}" == q8_0 ]]; then
            active_contexts=("${Q8_CONTEXT_CANDIDATES[@]}")
        elif [[ "${PROFILE}" == 24 && "${kv}" == q4_0 ]]; then
            active_contexts=("${Q4_CONTEXT_CANDIDATES[@]}")
        else
            active_contexts=("${CONTEXT_CANDIDATES[@]}")
        fi
        for context in "${active_contexts[@]}"; do
            for cache in "${CACHE_CANDIDATES[@]}"; do
                candidate_batch="${primary_batch}"
                if try_candidate "${context}" "${cache}" "${kv}" "${candidate_batch}" "${auth_file}" "${models_file}" "${chat_file}"; then
                    CTX_SIZE="${context}"; MOE_CACHE="${cache}"; KV_K="${kv}"; KV_V="${kv}"; BATCH="${candidate_batch}"
                    log "Selected context ${CTX_SIZE}, MoE cache ${MOE_CACHE}, KV ${KV_K^^}, batch ${BATCH}."
                    printf '[%s] SELECTED context=%s cache=%s kv=%s batch=%s\n' \
                        "$(date --iso-8601=seconds)" "${CTX_SIZE}" "${MOE_CACHE}" "${KV_K}" "${BATCH}" >>"${TUNE_LOG}"
                    return 0
                fi
                log "EXPECTED_PROFILE_MISS: context ${context}, cache ${cache}, KV ${kv}, batch ${candidate_batch}, reason ${CANDIDATE_REASON}."
                printf '[%s] EXPECTED_PROFILE_MISS context=%s cache=%s kv=%s batch=%s reason=%s\n' \
                    "$(date --iso-8601=seconds)" "${context}" "${cache}" "${kv}" "${candidate_batch}" "${CANDIDATE_REASON}" >>"${TUNE_LOG}"
                if [[ "${PROFILE}" == 24 && "${CANDIDATE_REASON}" == CUDA_OOM && "${BATCH_FALLBACK}" -gt 0 ]]; then
                    candidate_batch="${BATCH_FALLBACK}"
                    log "Retrying the same 24 GB candidate with reduced batch ${candidate_batch} after CUDA OOM."
                    if try_candidate "${context}" "${cache}" "${kv}" "${candidate_batch}" "${auth_file}" "${models_file}" "${chat_file}"; then
                        CTX_SIZE="${context}"; MOE_CACHE="${cache}"; KV_K="${kv}"; KV_V="${kv}"; BATCH="${candidate_batch}"
                        log "Selected context ${CTX_SIZE}, MoE cache ${MOE_CACHE}, KV ${KV_K^^}, batch ${BATCH}."
                        printf '[%s] SELECTED context=%s cache=%s kv=%s batch=%s\n' \
                            "$(date --iso-8601=seconds)" "${CTX_SIZE}" "${MOE_CACHE}" "${KV_K}" "${BATCH}" >>"${TUNE_LOG}"
                        return 0
                    fi
                    log "EXPECTED_PROFILE_MISS: reduced-batch retry reason ${CANDIDATE_REASON}."
                    printf '[%s] EXPECTED_PROFILE_MISS context=%s cache=%s kv=%s batch=%s reason=%s\n' \
                        "$(date --iso-8601=seconds)" "${context}" "${cache}" "${kv}" "${candidate_batch}" "${CANDIDATE_REASON}" >>"${TUNE_LOG}"
                fi
            done
        done
    done
    case "${PROFILE}" in
        12) die "UD-Q3_K_XL could not pass validation at >=8192 context on this GPU; no worse quant was downloaded." ;;
        16) die "UD-Q3_K_XL could not find a safe 16 GB profile through 12288 context; no worse quant was downloaded." ;;
        24) die "UD-Q3_K_XL could not find a safe 24 GB Q8/Q4 profile through the 16384 fallback; no worse quant was downloaded." ;;
        48) die "UD-Q3_K_XL could not find a safe 48 GB Q8 profile through 65536 context; inspect ${TUNE_LOG}." ;;
    esac
}

parse_cli "$@"
stage 1 "Hardware and profile validation"
[[ "$(uname -m)" == x86_64 ]] || die "This bootstrap supports Ubuntu 24.04 x86_64 only."
[[ -r /etc/os-release ]] || die "Cannot identify the operating system."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 ]] || die "Ubuntu 24.04 is required; detected ${PRETTY_NAME:-unknown OS}."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable; install or attach a working NVIDIA driver first."
nvidia-smi >/dev/null 2>&1 || die "The NVIDIA driver cannot communicate with a GPU."
GPU_NAME="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_UUID="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -n1 | xargs || true)"
GPU_UUID="${GPU_UUID:-unavailable}"
GPU_CC="$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | xargs || true)"
GPU_TOTAL_MIB="$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
DRIVER_VERSION="$(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader | head -n1 | xargs)"
[[ "${GPU_TOTAL_MIB}" =~ ^[0-9]+$ ]] || die "Could not read physical VRAM from GPU 0."
choose_profile "${GPU_TOTAL_MIB}"; verify_manual_profile_capacity; derive_cuda_architecture; load_profile_defaults
TOTAL_RAM_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"; RAM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
printf '\nDetected deployment target:\n'
printf '  GPU model:                %s\n' "${GPU_NAME}"
printf '  Physical VRAM:            %s MiB (%s GiB)\n' "${GPU_TOTAL_MIB}" "$((GPU_TOTAL_MIB / 1024))"
printf '  Compute capability:       %s\n' "${GPU_CC}"
printf '  Selected VRAM profile:    %s GB\n' "${PROFILE}"
printf '  CMake CUDA architecture:  %s\n' "${CMAKE_CUDA_ARCH}"
printf '  NVIDIA driver:            %s\n' "${DRIVER_VERSION}"
printf '  System RAM total/free:    %s GiB / %s GiB\n' "$((TOTAL_RAM_KIB / 1024 / 1024))" "$((RAM_AVAILABLE_KIB / 1024 / 1024))"
if [[ "${DRY_RUN}" == true ]]; then
    if (( TOTAL_RAM_KIB < MIN_RAM_KIB )); then
        printf 'WARNING: a real deployment would fail because at least nominal 64 GB RAM is required.\n' >&2
    elif [[ "${PROFILE}" != 32 && "${TOTAL_RAM_KIB}" -lt "${RECOMMENDED_RAM_KIB}" ]]; then
        printf 'WARNING: adaptive profiles strongly recommend 96 GB+ RAM.\n' >&2
    fi
    print_dry_run_strategy
    exit 0
fi

require_root
acquire_bootstrap_lock
version_ge "${DRIVER_VERSION}" "${MIN_DRIVER_VERSION}" || die "NVIDIA driver ${DRIVER_VERSION} is too old for CUDA 13.x (need >= ${MIN_DRIVER_VERSION})."
(( TOTAL_RAM_KIB >= MIN_RAM_KIB )) || die "At least nominal 64 GB RAM is required; detected $((TOTAL_RAM_KIB / 1024 / 1024)) GiB."
if [[ "${PROFILE}" != 32 && "${TOTAL_RAM_KIB}" -lt "${RECOMMENDED_RAM_KIB}" ]]; then
    printf 'WARNING: adaptive profiles strongly recommend 96 GB+ RAM; this host has %s GiB.\n' "$((TOTAL_RAM_KIB / 1024 / 1024))" >&2
fi

mkdir -p "${QWEN_ROOT}"
ROOT_FREE_KIB="$(df -Pk "${QWEN_ROOT}" | awk 'NR==2 {print $4}')"; ROOT_DEVICE="$(stat -c '%d' "${QWEN_ROOT}")"; REUSABLE_KIB=0
for shard in "${SHARDS[@]}"; do
    shard_path="${MODEL_ROOT}/${shard}"; shard_stamp="${shard_path}.qwen38-revision"
    if [[ -s "${shard_path}" && -r "${shard_stamp}" ]]; then
        shard_bytes="$(stat -c '%s' "${shard_path}")"
        [[ "$(<"${shard_stamp}")" == "repo=${HF_REPO};revision=${HF_REVISION};size=${shard_bytes}" ]] && REUSABLE_KIB=$((REUSABLE_KIB + $(du -k "${shard_path}" | awk '{print $1}')))
    fi
done
if [[ -r "${ACTIVE_DOWNLOAD_STAMP}" ]]; then
    ACTIVE_DOWNLOAD="$(<"${ACTIVE_DOWNLOAD_STAMP}")"
    for shard in "${SHARDS[@]}"; do
        if [[ "${ACTIVE_DOWNLOAD}" == "repo=${HF_REPO};revision=${HF_REVISION};shard=${shard}" ]]; then
            INCOMPLETE_KIB="$(find "${MODEL_ROOT}" -type f -name '*.incomplete' -printf '%k\n' 2>/dev/null | awk '{sum += $1} END {print sum + 0}')"
            REUSABLE_KIB=$((REUSABLE_KIB + INCOMPLETE_KIB))
            if [[ -s "${MODEL_ROOT}/${shard}" && ! -r "${MODEL_ROOT}/${shard}.qwen38-revision" ]]; then
                REUSABLE_KIB=$((REUSABLE_KIB + $(du -k "${MODEL_ROOT}/${shard}" | awk '{print $1}')))
            fi
            break
        fi
    done
fi
EARLY_NVCC_VERSION=""
if [[ -x /usr/local/cuda-13.2/bin/nvcc ]] && \
   /usr/local/cuda-13.2/bin/nvcc --version 2>/dev/null | grep -Eq 'release 13\.2([,[:space:]]|$)'; then
    EARLY_NVCC_VERSION="$(/usr/local/cuda-13.2/bin/nvcc --version 2>/dev/null | tail -n1 || true)"
fi
if command -v git >/dev/null 2>&1 && [[ -n "${EARLY_NVCC_VERSION}" ]] && \
   [[ -d "${LLAMA_DIR}/.git" && -x "${SERVER_BIN}" && -r "${BUILD_STAMP}" ]] && \
   [[ "$(git -C "${LLAMA_DIR}" rev-parse HEAD 2>/dev/null || true)" == "${LLAMA_COMMIT}" ]] && \
   [[ "$(sed -n '1p' "${BUILD_STAMP}")" == "commit=${LLAMA_COMMIT};cuda=${EARLY_NVCC_VERSION};arch=${CMAKE_CUDA_ARCH};gcc=13;profile=qwen38-v1" ]] && \
   [[ "$(sed -n '2p' "${BUILD_STAMP}")" == "$(sha256sum "${SERVER_BIN}" | awk '{print $1}')" ]]; then
    REUSABLE_KIB=$((REUSABLE_KIB + $(du -sk "${LLAMA_DIR}" | awk '{print $1}')))
fi
if [[ -n "${EARLY_NVCC_VERSION}" && "$(stat -c '%d' /usr/local/cuda-13.2)" == "${ROOT_DEVICE}" ]]; then
    REUSABLE_KIB=$((REUSABLE_KIB + $(du -sk /usr/local/cuda-13.2 | awk '{print $1}')))
fi
AVAILABLE_CAPACITY_KIB=$((ROOT_FREE_KIB + REUSABLE_KIB))
(( AVAILABLE_CAPACITY_KIB >= MIN_CAPACITY_KIB )) || die "The deployment filesystem needs 130 GiB free plus verified reusable artifacts; found $((AVAILABLE_CAPACITY_KIB / 1024 / 1024)) GiB."
log "Deployment capacity: $((AVAILABLE_CAPACITY_KIB / 1024 / 1024)) GiB."

stage 2 "System dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update
packages=(build-essential ca-certificates ccache cmake curl g++-13 gcc-13 git iproute2 jq logrotate ninja-build openssl procps python3 python3-pip python3-venv tmux util-linux wget)
missing_packages=()
for package in "${packages[@]}"; do dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q 'ok installed' || missing_packages+=("${package}"); done
if ((${#missing_packages[@]})); then apt-get install -y --no-install-recommends "${missing_packages[@]}"; else log "All required Ubuntu packages are installed."; fi
acquire_flock_after_dependencies

stage 3 "CUDA Toolkit"
CUDA_HOME=""
if [[ -x /usr/local/cuda-13.2/bin/nvcc ]] && /usr/local/cuda-13.2/bin/nvcc --version | grep -Eq 'release 13\.2([,[:space:]]|$)'; then
    CUDA_HOME=/usr/local/cuda-13.2
elif command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -Eq 'release 13\.2([,[:space:]]|$)'; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
    log "CUDA Toolkit 13.2 is missing; installing toolkit-only packages (not a driver)."
    if ! dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null | grep -q 'ok installed'; then
        CUDA_KEYRING_DEB="$(mktemp /tmp/qwen38.cuda-keyring.XXXXXX.deb)"; TEMP_FILES+=("${CUDA_KEYRING_DEB}")
        curl --fail --location --retry 5 --retry-all-errors --output "${CUDA_KEYRING_DEB}" "${CUDA_KEYRING_URL}"; dpkg -i "${CUDA_KEYRING_DEB}"
    fi
    apt-get update; apt-cache show cuda-toolkit-13-2 >/dev/null 2>&1 || die "NVIDIA repository does not offer cuda-toolkit-13-2; no driver was changed."
    apt-get install -y --no-install-recommends cuda-toolkit-13-2; CUDA_HOME=/usr/local/cuda-13.2
fi
NVCC="${CUDA_HOME}/bin/nvcc"; [[ -x "${NVCC}" ]] || die "CUDA 13.2 installation completed without ${NVCC}."
export PATH="${CUDA_HOME}/bin:${PATH}"; export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
NVCC_VERSION="$("${NVCC}" --version | tail -n1)"; log "Using ${NVCC_VERSION} from ${CUDA_HOME}."

stage 4 "Pinned llama.cpp source"
mkdir -p "${QWEN_ROOT}" "${QWEN_ROOT}/models" "${LOG_DIR}"
if [[ ! -e "${LLAMA_DIR}" ]]; then git clone --filter=blob:none --no-checkout "${LLAMA_REPO}" "${LLAMA_DIR}"
elif [[ ! -d "${LLAMA_DIR}/.git" ]]; then die "${LLAMA_DIR} exists but is not a git checkout."
else
    EXISTING_ORIGIN="$(git -C "${LLAMA_DIR}" remote get-url origin 2>/dev/null || true)"
    case "${EXISTING_ORIGIN}" in "${LLAMA_REPO}"|"${LLAMA_REPO%.git}"|git@github.com:GenerelSchwerz/llama.cpp.git) ;; *) die "${LLAMA_DIR} has unexpected origin '${EXISTING_ORIGIN}'." ;; esac
fi
if ! git -C "${LLAMA_DIR}" cat-file -e "${LLAMA_COMMIT}^{commit}" 2>/dev/null; then git -C "${LLAMA_DIR}" fetch --no-tags --depth=1 origin "${LLAMA_COMMIT}"; fi
[[ -z "$(git -C "${LLAMA_DIR}" status --porcelain --untracked-files=no)" ]] || die "Tracked changes exist in ${LLAMA_DIR}."
git -C "${LLAMA_DIR}" checkout --detach "${LLAMA_COMMIT}"
[[ "$(git -C "${LLAMA_DIR}" rev-parse HEAD)" == "${LLAMA_COMMIT}" ]] || die "Pinned checkout verification failed."

stage 5 "Architecture-specific llama.cpp build"
BUILD_SIGNATURE="commit=${LLAMA_COMMIT};cuda=${NVCC_VERSION};arch=${CMAKE_CUDA_ARCH};gcc=13;profile=qwen38-v1"
EXISTING_SIGNATURE=""; EXISTING_BINARY_HASH=""; CURRENT_BINARY_HASH=""
if [[ -r "${BUILD_STAMP}" ]]; then EXISTING_SIGNATURE="$(sed -n '1p' "${BUILD_STAMP}")"; EXISTING_BINARY_HASH="$(sed -n '2p' "${BUILD_STAMP}")"; fi
[[ -x "${SERVER_BIN}" ]] && CURRENT_BINARY_HASH="$(sha256sum "${SERVER_BIN}" | awk '{print $1}')"
if [[ -x "${SERVER_BIN}" && "${EXISTING_SIGNATURE}" == "${BUILD_SIGNATURE}" && -n "${EXISTING_BINARY_HASH}" && "${CURRENT_BINARY_HASH}" == "${EXISTING_BINARY_HASH}" ]]; then
    log "Reusing verified ${CMAKE_CUDA_ARCH} llama-server build."
else
    cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
        -DCMAKE_CUDA_COMPILER="${NVCC}" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
        -DCMAKE_CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCH}" \
        -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_FORCE_MMQ=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF \
        -DGGML_BACKEND_DL=OFF -DGGML_BLAS=OFF -DGGML_CCACHE=ON
    cmake --build "${BUILD_DIR}" --target llama-server --parallel "$(nproc)"
    [[ -x "${SERVER_BIN}" ]] || die "Build finished without ${SERVER_BIN}."
    printf '%s\n%s\n' "${BUILD_SIGNATURE}" "$(sha256sum "${SERVER_BIN}" | awk '{print $1}')" >"${BUILD_STAMP}"; chmod 600 "${BUILD_STAMP}"
fi
SERVER_HELP="$("${SERVER_BIN}" --help 2>&1 || true)"
required_flags=(--offline --spec-type --ctx-size --batch-size --ubatch-size --parallel --threads --gpu-layers --flash-attn --fit --load-mode --lazy-mode --moe-expert-cache-size --cache-type-k --cache-type-v --kv-offload --cache-ram --jinja --no-warmup --experimental-logs --alias --api-key-file --host --port)
for required_flag in "${required_flags[@]}"; do grep -Fq -- "${required_flag}" <<<"${SERVER_HELP}" || die "Pinned server lacks required option ${required_flag}."; done

stage 6 "Pinned resumable model download"
VENV_DIR="${QWEN_ROOT}/.venv-huggingface"; [[ -x "${VENV_DIR}/bin/python" ]] || python3 -m venv "${VENV_DIR}"
if [[ ! -x "${VENV_DIR}/bin/hf" ]] || ! "${VENV_DIR}/bin/python" -c 'import huggingface_hub, hf_xet' >/dev/null 2>&1; then "${VENV_DIR}/bin/python" -m pip install --upgrade pip huggingface_hub hf_xet; fi
mkdir -p "${MODEL_DIR}"; all_shards_valid=true
for shard in "${SHARDS[@]}"; do [[ -s "${MODEL_ROOT}/${shard}" ]] && (( $(stat -c '%s' "${MODEL_ROOT}/${shard}" 2>/dev/null || printf 0) > 1048576 )) || all_shards_valid=false; done
MODEL_SIGNATURE=""
if [[ "${all_shards_valid}" == true ]]; then MODEL_SIGNATURE="repo=${HF_REPO};revision=${HF_REVISION}"; for shard in "${SHARDS[@]}"; do MODEL_SIGNATURE+=";${shard}=$(stat -c '%s' "${MODEL_ROOT}/${shard}")"; done; fi
EXISTING_MODEL_SIGNATURE=""; [[ -r "${MODEL_STAMP}" ]] && EXISTING_MODEL_SIGNATURE="$(<"${MODEL_STAMP}")"
global_model_valid=false; [[ "${all_shards_valid}" == true && "${EXISTING_MODEL_SIGNATURE}" == "${MODEL_SIGNATURE}" ]] && global_model_valid=true
export HF_HUB_DISABLE_XET=0 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_XET_HIGH_PERFORMANCE=1
for shard in "${SHARDS[@]}"; do
    shard_path="${MODEL_ROOT}/${shard}"; shard_stamp="${shard_path}.qwen38-revision"; shard_bytes="$(stat -c '%s' "${shard_path}" 2>/dev/null || printf 0)"
    expected_shard_stamp="repo=${HF_REPO};revision=${HF_REVISION};size=${shard_bytes}"
    if [[ "${global_model_valid}" == true ]] || [[ "${shard_bytes}" -gt 1048576 && -r "${shard_stamp}" && "$(<"${shard_stamp}")" == "${expected_shard_stamp}" ]]; then
        log "Verified model shard already present: ${shard##*/}"; atomic_write "${shard_stamp}" 600 <<<"${expected_shard_stamp}"; continue
    fi
    atomic_write "${ACTIVE_DOWNLOAD_STAMP}" 600 <<<"repo=${HF_REPO};revision=${HF_REVISION};shard=${shard}"; download_ok=false
    for attempt in 1 2 3; do
        log "Pinned download ${shard##*/}, attempt ${attempt}/3 (resumable)."
        if "${VENV_DIR}/bin/hf" download "${HF_REPO}" "${shard}" --revision "${HF_REVISION}" --local-dir "${MODEL_ROOT}"; then download_ok=true; break; fi
        (( attempt == 3 )) || sleep $((attempt * 10))
    done
    [[ "${download_ok}" == true ]] || die "Model shard download failed: ${shard}."
    shard_bytes="$(stat -c '%s' "${shard_path}")"; (( shard_bytes > 1048576 )) || die "Downloaded shard is implausibly small."
    atomic_write "${shard_stamp}" 600 <<<"repo=${HF_REPO};revision=${HF_REVISION};size=${shard_bytes}"; rm -f -- "${ACTIVE_DOWNLOAD_STAMP}"
done
MODEL_SIGNATURE="repo=${HF_REPO};revision=${HF_REVISION}"
for shard in "${SHARDS[@]}"; do [[ -s "${MODEL_ROOT}/${shard}" ]] || die "Expected model shard is missing: ${shard}."; MODEL_SIGNATURE+=";${shard}=$(stat -c '%s' "${MODEL_ROOT}/${shard}")"; done
atomic_write "${MODEL_STAMP}" 600 <<<"${MODEL_SIGNATURE}"; rm -f -- "${ACTIVE_DOWNLOAD_STAMP}"

stage 7 "Authentication and resolved profile"
if [[ -e "${API_KEY_FILE}" || -L "${API_KEY_FILE}" ]]; then
    [[ -f "${API_KEY_FILE}" && -s "${API_KEY_FILE}" && ! -L "${API_KEY_FILE}" ]] || die "Existing API key path is not a non-empty regular file."
    chmod 600 "${API_KEY_FILE}"; log "Preserving the existing API key."
else
    atomic_write "${API_KEY_FILE}" 600 <<<"$(openssl rand -hex 32)"; log "Generated a new 256-bit API key."
fi
API_KEY_VALUE="$(awk 'NF && $1 !~ /^#/ {print; exit}' "${API_KEY_FILE}")"; [[ -n "${API_KEY_VALUE}" ]] || die "The preserved API key contains no usable key."
touch "${LOG_FILE}" "${TUNE_LOG}"; chmod 600 "${LOG_FILE}" "${TUNE_LOG}"
if [[ "${PROFILE}" == 32 ]]; then
    log "Using the locked 32 GB profile without auto-tuning."; write_profile
elif cached_profile_matches; then
    load_cached_profile; log "Reusing hardware-specific fitted configuration from ${PROFILE_FILE}."
else
    [[ "${RETUNE}" == false ]] || log "--retune requested; model, build, and API key remain untouched."
    fit_adaptive_profile; write_profile
fi

stage 8 "Production scripts and supervision"
atomic_write "${RUNNER}" 755 <<'RUNNER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck disable=SC1091
source /root/qwen38/profile.env
RAM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
printf 'Qwen 3.8 Flash\nGPU: %s\nVRAM profile: %sGB\nVRAM detected: %s MiB\nContext: %s\nMoE cache: %s\nKV cache: %s / %s\nThreads: %s\nMTP: OFF\nAvailable RAM: %s MiB\n' \
    "${GPU_NAME}" "${PROFILE}" "${VRAM_MB}" "${CTX_SIZE}" "${MOE_CACHE}" "${KV_K^^}" "${KV_V^^}" "${THREADS}" "$((RAM_AVAILABLE_KIB / 1024))"
export CUDA_VISIBLE_DEVICES=0 LLAMA_ATTN_ROT_DISABLE=1 COMPILE_OFF=1
exec "${SERVER_BIN}" \
    --offline --model "${MODEL_PATH}" --spec-type none \
    -c "${CTX_SIZE}" -b "${BATCH}" -ub "${UBATCH}" -np 1 -t "${THREADS}" \
    -ngl all -fa on -fit off --load-mode none --lazy-mode on \
    --moe-expert-cache-size "${MOE_CACHE}" -ctk "${KV_K}" -ctv "${KV_V}" -kvo \
    --cache-ram 0 --jinja --no-warmup --experimental-logs \
    --alias qwen3.8-flash --api-key-file /root/qwen38/api-key.txt --host 0.0.0.0 --port 11434
RUNNER_EOF

atomic_write "${TMUX_RUNNER}" 755 <<'TMUX_RUNNER_EOF'
#!/usr/bin/env bash
set -uo pipefail
readonly LOG_FILE=/root/qwen38/logs/qwen38-production.log
while :; do
    printf '[%s] tmux supervisor starting llama-server\n' "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
    /root/qwen38/run-qwen38.sh >>"${LOG_FILE}" 2>&1; result=$?
    printf '[%s] llama-server exited with status %s; restarting in 10 seconds\n' "$(date --iso-8601=seconds)" "${result}" >>"${LOG_FILE}"
    sleep 10
done
TMUX_RUNNER_EOF

if systemd_usable; then
    SUPERVISOR_MODE=systemd
    atomic_write "${SERVICE_FILE}" 644 <<'SERVICE_EOF'
[Unit]
Description=Qwen 3.8 Flash OpenAI-compatible llama-server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/qwen38
ExecStart=/root/qwen38/run-qwen38.sh
Restart=on-failure
RestartSec=10
TimeoutStopSec=60
KillSignal=SIGTERM
LimitNOFILE=1048576
StandardOutput=append:/root/qwen38/logs/qwen38-production.log
StandardError=append:/root/qwen38/logs/qwen38-production.log

[Install]
WantedBy=multi-user.target
SERVICE_EOF
    systemctl daemon-reload
    # Do not enable an unvalidated configuration at boot.
    systemctl disable "${SERVICE_NAME}" >/dev/null 2>&1 || true
else
    SUPERVISOR_MODE=tmux
fi
atomic_write "${MODE_FILE}" 600 <<<"${SUPERVISOR_MODE}"

atomic_write "${CONTROL_SCRIPT}" 755 <<'CONTROL_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
readonly PROFILE_FILE=/root/qwen38/profile.env
readonly TMUX_RUNNER=/root/qwen38/run-qwen38-tmux.sh
readonly MODE_FILE=/root/qwen38/supervisor-mode
readonly LOG_FILE=/root/qwen38/logs/qwen38-production.log
readonly SERVICE=qwen38.service
readonly TMUX_SESSION=qwen38-prod
# shellcheck disable=SC1091
source "${PROFILE_FILE}"
systemd_usable() { [[ -d /run/systemd/system ]] && systemctl show-environment >/dev/null 2>&1; }
mode() { [[ -r "${MODE_FILE}" ]] && head -n1 "${MODE_FILE}" || printf 'tmux\n'; }
relevant_pids() {
    local proc pid executable commandline process_name
    for proc in /proc/[0-9]*; do
        pid="${proc#/proc/}"; [[ "${pid}" != "$$" && -r "${proc}/cmdline" ]] || continue
        executable="$(readlink -f "${proc}/exe" 2>/dev/null || true)"; commandline="$(tr '\0' ' ' <"${proc}/cmdline" 2>/dev/null || true)"; process_name="$(cat "${proc}/comm" 2>/dev/null || true)"
        [[ "${process_name}" == llama-server || "${executable}" == */llama-server* ]] || continue
        if [[ "${commandline}" == *"${MODEL_PATH}"* || "${commandline}" == *qwen3.8-flash* || "${commandline}" == *"--port 11434"* || "${commandline}" == *"--port=11434"* ]]; then printf '%s\n' "${pid}"; fi
    done
}
stop_managers() {
    if systemd_usable && systemctl cat "${SERVICE}" >/dev/null 2>&1; then systemctl stop "${SERVICE}" || true; fi
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then tmux kill-session -t "${TMUX_SESSION}"; fi
}
stop_relevant() {
    local deadline owned pid gpu_pid; local -a pids=() original=() gpu_pids=()
    mapfile -t pids < <(relevant_pids); ((${#pids[@]})) || return 0; original=("${pids[@]}")
    printf 'Stopping managed llama-server PID(s): %s\n' "${pids[*]}"; kill -TERM "${pids[@]}" 2>/dev/null || true; deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do mapfile -t pids < <(relevant_pids); ((${#pids[@]} == 0)) && break; sleep 1; done
    mapfile -t pids < <(relevant_pids); if ((${#pids[@]})); then kill -KILL "${pids[@]}" 2>/dev/null || true; sleep 2; fi
    mapfile -t pids < <(relevant_pids); ((${#pids[@]} == 0)) || { printf 'Managed process survived shutdown.\n' >&2; return 1; }
    deadline=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true); owned=false
        for pid in "${original[@]}"; do for gpu_pid in "${gpu_pids[@]}"; do [[ "${pid}" == "${gpu_pid//[[:space:]]/}" ]] && { owned=true; break 2; }; done; done
        [[ "${owned}" == false ]] && return 0; sleep 1
    done
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true; return 1
}
stop_all() { stop_managers; stop_relevant; }
print_config() {
    local available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    printf 'Qwen 3.8 Flash\nGPU: %s\nVRAM profile: %sGB\nVRAM detected: %s MiB\nContext: %s\nMoE cache: %s\nKV cache: %s / %s\nThreads: %s\nMTP: OFF\nAvailable RAM: %s MiB\n' \
        "${GPU_NAME}" "${PROFILE}" "${VRAM_MB}" "${CTX_SIZE}" "${MOE_CACHE}" "${KV_K^^}" "${KV_V^^}" "${THREADS}" "$((available_kib / 1024))"
}
wait_for_clean_vram() {
    local free_mib minimum deadline=$((SECONDS + 60))
    if [[ "${PROFILE}" == 32 && "${FORCED_PROFILE:-false}" != true ]]; then
        minimum=30000
        (( VRAM_MB - 2048 < minimum )) && minimum=$((VRAM_MB - 2048))
    else
        minimum=$((VRAM_MB - 2048))
    fi
    (( minimum < 0 )) && minimum=0
    while (( SECONDS < deadline )); do free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | xargs)"; (( free_mib >= minimum )) && return 0; sleep 2; done
    printf 'Only %s MiB VRAM is free; expected at least %s MiB before startup.\n' "${free_mib}" "${minimum}" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true; return 1
}
start_server() {
    local port_owner
    [[ -x "${SERVER_BIN}" && -s "${MODEL_PATH}" ]] || { printf 'Server binary or model is missing.\n' >&2; return 1; }
    stop_all
    port_owner="$(ss -H -ltnp 'sport = :11434' 2>/dev/null || true)"
    [[ -z "${port_owner}" ]] || { printf 'TCP port 11434 has an unrelated listener: %s\n' "${port_owner}" >&2; return 1; }
    wait_for_clean_vram; print_config
    if [[ "$(mode)" == systemd ]]; then systemd_usable || { printf 'systemd is configured but unavailable; rerun bootstrap.\n' >&2; return 1; }; systemctl start "${SERVICE}"; systemctl is-active --quiet "${SERVICE}"
    else
        command -v tmux >/dev/null 2>&1 || { printf 'tmux is unavailable.\n' >&2; return 1; }
        tmux has-session -t "${TMUX_SESSION}" 2>/dev/null && tmux kill-session -t "${TMUX_SESSION}"
        tmux new-session -d -s "${TMUX_SESSION}" "${TMUX_RUNNER}"; sleep 2; tmux has-session -t "${TMUX_SESSION}" 2>/dev/null
    fi
}
show_status() {
    local -a pids=(); print_config
    if [[ "$(mode)" == systemd ]] && systemd_usable; then systemctl --no-pager --full status "${SERVICE}" || true
    elif tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then printf 'tmux session %s is running.\n' "${TMUX_SESSION}"
    else printf 'tmux session %s is stopped.\n' "${TMUX_SESSION}"; fi
    mapfile -t pids < <(relevant_pids); ((${#pids[@]})) && { printf 'Managed llama-server PID(s): %s\n' "${pids[*]}"; return 0; }
    printf 'No managed llama-server process is running.\n'; return 3
}
case "${1:-status}" in
    start) start_server ;; stop) stop_all ;; restart) stop_all; start_server ;; status) show_status ;;
    logs) exec tail -n 200 -F "${LOG_FILE}" ;; pid) relevant_pids ;;
    *) printf 'Usage: %s {start|stop|restart|status|logs|pid}\n' "$0" >&2; exit 64 ;;
esac
CONTROL_EOF

atomic_write /etc/logrotate.d/qwen38 644 <<'LOGROTATE_EOF'
/root/qwen38/logs/qwen38-production.log /root/qwen38/logs/qwen38-tuning.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su root root
}
LOGROTATE_EOF

stage 9 "Production startup"
prepare_clean_gpu
PRODUCTION_LOG_START_LINE=$(( $(wc -l <"${LOG_FILE}") + 1 ))
PRODUCTION_STARTED=true
"${CONTROL_SCRIPT}" start
log "Server launched using ${SUPERVISOR_MODE}; waiting for model load."

stage 10 "Authenticated production verification"
production_fail() {
    local message="$1"
    "${CONTROL_SCRIPT}" stop || true
    PRODUCTION_STARTED=false
    die "${message}"
}
AUTH_HEADER="$(mktemp /tmp/qwen38.auth.XXXXXX)"; MODELS_RESPONSE="$(mktemp /tmp/qwen38.models.XXXXXX.json)"; CHAT_RESPONSE="$(mktemp /tmp/qwen38.chat.XXXXXX.json)"
TEMP_FILES+=("${AUTH_HEADER}" "${MODELS_RESPONSE}" "${CHAT_RESPONSE}"); chmod 600 "${AUTH_HEADER}" "${MODELS_RESPONSE}" "${CHAT_RESPONSE}"
printf 'Authorization: Bearer %s\n' "${API_KEY_VALUE}" >"${AUTH_HEADER}"
deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS)); healthy=false
while (( SECONDS < deadline )); do
    if curl --silent --show-error --fail --max-time 10 --header @"${AUTH_HEADER}" --output "${MODELS_RESPONSE}" http://127.0.0.1:11434/v1/models 2>/dev/null && jq -e '.data[]? | select(.id == "qwen3.8-flash")' "${MODELS_RESPONSE}" >/dev/null 2>&1; then healthy=true; break; fi
    sleep 5
done
if [[ "${healthy}" != true ]]; then "${CONTROL_SCRIPT}" status || true; tail -n 120 "${LOG_FILE}" >&2 || true; production_fail "Authenticated /v1/models did not become healthy."; fi
UNAUTHENTICATED_STATUS="$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 10 http://127.0.0.1:11434/v1/models || true)"
[[ "${UNAUTHENTICATED_STATUS}" != 200 ]] || production_fail "Security verification failed: unauthenticated /v1/models returned 200."
CHAT_PAYLOAD='{"model":"qwen3.8-flash","messages":[{"role":"user","content":"Reply with exactly the word READY."}],"temperature":0,"max_tokens":128}'
if ! curl --silent --show-error --fail --max-time 600 --header @"${AUTH_HEADER}" --header 'Content-Type: application/json' --data "${CHAT_PAYLOAD}" --output "${CHAT_RESPONSE}" http://127.0.0.1:11434/v1/chat/completions; then tail -n 120 "${LOG_FILE}" >&2 || true; production_fail "Production chat completion failed."; fi
jq -e '(.choices | length > 0) and (((.choices[0].message.content // "") | length > 0) or ((.choices[0].message.reasoning_content // "") | length > 0))' "${CHAT_RESPONSE}" >/dev/null || production_fail "Chat completion returned neither content nor reasoning_content."
if log_slice_has_oom "${LOG_FILE}" "${PRODUCTION_LOG_START_LINE}"; then
    production_fail "CUDA OOM is present in the current production launch log."
fi
GPU_USED_MIB="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | xargs)"; GPU_FREE_MIB="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | xargs)"
RAM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"; SERVER_PID="$("${CONTROL_SCRIPT}" pid | head -n1)"
[[ -n "${SERVER_PID}" ]] || production_fail "Verification succeeded but the managed server PID was not resolved."
if [[ "${PROFILE}" != 32 && "${GPU_FREE_MIB}" -lt "${HEADROOM_MIB}" ]]; then production_fail "Production headroom fell to ${GPU_FREE_MIB} MiB (required ${HEADROOM_MIB} MiB). Stop other GPU users and rerun with --retune."; fi
if [[ "${SUPERVISOR_MODE}" == systemd ]]; then systemctl enable "${SERVICE_NAME}"; fi
DEPLOYMENT_COMPLETE=true
printf '\nDEPLOYMENT COMPLETE\n'
printf '  GPU model:        %s\n' "${GPU_NAME}"; printf '  Compute/CMake:    %s / %s\n' "${GPU_CC}" "${CMAKE_CUDA_ARCH}"
printf '  VRAM profile:     %s GB\n' "${PROFILE}"; printf '  VRAM used/free:   %s MiB / %s MiB\n' "${GPU_USED_MIB}" "${GPU_FREE_MIB}"
printf '  RAM available:    %s MiB\n' "$((RAM_AVAILABLE_KIB / 1024))"; printf '  Context/cache:    %s / %s\n' "${CTX_SIZE}" "${MOE_CACHE}"
printf '  KV cache:         %s / %s\n' "${KV_K^^}" "${KV_V^^}"; printf '  Listening/model:  0.0.0.0:11434 / qwen3.8-flash\n'
printf '  Server PID:       %s\n' "${SERVER_PID}"; printf '  Supervisor:       %s\n' "${SUPERVISOR_MODE}"; printf '  Resolved config:  %s\n' "${PROFILE_FILE}"
if [[ "${PROFILE}" == 32 && "${GPU_FREE_MIB}" -lt 2048 ]]; then printf 'WARNING: Locked 32 GB profile has unexpectedly low GPU headroom (%s MiB).\n' "${GPU_FREE_MIB}" >&2; fi
SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
printf '\nOpenAI-compatible endpoint: http://%s:11434/v1\n' "${SERVER_IP:-SERVER_IP}"
printf 'API key remains in %s (mode 0600).\n' "${API_KEY_FILE}"
printf 'Manage with: %s {start|stop|restart|status|logs}\n' "${CONTROL_SCRIPT}"
