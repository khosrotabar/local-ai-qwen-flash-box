#!/usr/bin/env bash
# One-command bootstrap for Qwen3.8-Flash-Next UD-Q3_K_XL on an RTX 5090.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly QWEN_ROOT="/root/qwen38"
readonly LLAMA_DIR="${QWEN_ROOT}/llama.cpp"
readonly BUILD_DIR="${LLAMA_DIR}/build-qwen38"
readonly SERVER_BIN="${BUILD_DIR}/bin/llama-server"
readonly MODEL_ROOT="${QWEN_ROOT}/models/Qwen3.8-Flash-Next-GGUF"
readonly MODEL_DIR="${MODEL_ROOT}/UD-Q3_K_XL"
readonly MODEL_NAME="Qwen3.8-Flash-Next-UD-Q3_K_XL"
readonly MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}-00001-of-00003.gguf"
readonly API_KEY_FILE="${QWEN_ROOT}/api-key.txt"
readonly LOG_DIR="${QWEN_ROOT}/logs"
readonly LOG_FILE="${LOG_DIR}/qwen38-production.log"
readonly RUNNER="${QWEN_ROOT}/run-qwen38.sh"
readonly TMUX_RUNNER="${QWEN_ROOT}/run-qwen38-tmux.sh"
readonly CONTROL_SCRIPT="${QWEN_ROOT}/start-qwen38.sh"
readonly MODE_FILE="${QWEN_ROOT}/supervisor-mode"
readonly BUILD_STAMP="${BUILD_DIR}/.qwen38-build-stamp"
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
readonly MIN_RAM_KIB=$((64000000000 / 1024)) # 64 GB nominal, expressed in /proc/meminfo KiB.
readonly MIN_CAPACITY_KIB=$((130 * 1024 * 1024))
readonly MIN_FREE_VRAM_MIB=30000
readonly HEALTH_TIMEOUT_SECONDS=1800

STAGE="initialization"
TEMP_FILES=()

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    printf 'ERROR [%s]: %s\n' "${STAGE}" "$*" >&2
    exit 1
}

cleanup() {
    local path
    for path in "${TEMP_FILES[@]:-}"; do
        if [[ -n "${path}" && "${path}" == /tmp/qwen38.* ]]; then
            rm -f -- "${path}"
        fi
    done
}

on_error() {
    local exit_code="$1"
    local line="$2"
    local command="$3"
    printf '\nBOOTSTRAP FAILED\n  stage: %s\n  line: %s\n  command: %s\n  exit: %s\n' \
        "${STAGE}" "${line}" "${command}" "${exit_code}" >&2
    printf 'Re-run this script after correcting the error; completed downloads and builds are reused.\n' >&2
}

trap cleanup EXIT
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

stage() {
    STAGE="$2"
    printf '\n[%s/9] %s\n' "$1" "$2"
}

require_root() {
    [[ "${EUID}" -eq 0 ]] || die "Run this script as root (sudo ./bootstrap.sh)."
}

version_ge() {
    dpkg --compare-versions "$1" ge "$2"
}

systemd_usable() {
    [[ -d /run/systemd/system ]] && systemctl show-environment >/dev/null 2>&1
}

atomic_write() {
    local destination="$1"
    local mode="$2"
    local temporary
    temporary="$(mktemp /tmp/qwen38.XXXXXX)"
    TEMP_FILES+=("${temporary}")
    cat >"${temporary}"
    install -o root -g root -m "${mode}" "${temporary}" "${destination}"
}

require_root
command -v flock >/dev/null 2>&1 || \
    die "The base-system flock utility is missing. Install util-linux, then rerun; no system changes have been made."
exec 9>/var/lock/qwen38-bootstrap.lock
flock -n 9 || die "Another qwen38 bootstrap is already running."

stage 1 "Hardware validation"

[[ "$(uname -m)" == "x86_64" ]] || die "This bootstrap supports Ubuntu 24.04 x86_64 only."
[[ -r /etc/os-release ]] || die "Cannot identify the operating system."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || \
    die "Ubuntu 24.04 is required; detected ${PRETTY_NAME:-unknown OS}."

command -v nvidia-smi >/dev/null 2>&1 || \
    die "nvidia-smi is unavailable. Install/attach a working NVIDIA driver first; this script will not replace drivers."
nvidia-smi >/dev/null 2>&1 || \
    die "The NVIDIA driver cannot communicate with a GPU. Fix the provider/host driver first."

GPU_NAME="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_CC="$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | xargs || true)"
if [[ "${GPU_NAME}" != *"RTX 5090"* && "${GPU_CC}" != 12.0* ]]; then
    die "GPU 0 must be an RTX 5090 / SM 12.0 Blackwell GPU; detected '${GPU_NAME}' (compute capability '${GPU_CC:-unknown}')."
fi

DRIVER_VERSION="$(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader | head -n1 | xargs)"
version_ge "${DRIVER_VERSION}" "${MIN_DRIVER_VERSION}" || \
    die "NVIDIA driver ${DRIVER_VERSION} is too old for CUDA 13.x minor compatibility (need >= ${MIN_DRIVER_VERSION}). Update it through the GPU provider; the bootstrap deliberately does not replace drivers."

TOTAL_RAM_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
(( TOTAL_RAM_KIB >= MIN_RAM_KIB )) || \
    die "At least a nominal 64 GB RAM host is required; detected $((TOTAL_RAM_KIB / 1024 / 1024)) GiB usable."

mkdir -p "${QWEN_ROOT}"
ROOT_FREE_KIB="$(df -Pk "${QWEN_ROOT}" | awk 'NR==2 {print $4}')"
ROOT_DEVICE="$(stat -c '%d' "${QWEN_ROOT}")"
REUSABLE_KIB=0
# Count only artifacts whose provenance can be tied to this deployment. This
# makes interrupted runs resumable without letting unrelated /root/qwen38 data
# hide a genuinely undersized filesystem.
for shard in "${SHARDS[@]}"; do
    shard_path="${MODEL_ROOT}/${shard}"
    shard_stamp="${shard_path}.qwen38-revision"
    if [[ -s "${shard_path}" && -r "${shard_stamp}" ]]; then
        shard_bytes="$(stat -c '%s' "${shard_path}")"
        if [[ "$(<"${shard_stamp}")" == "repo=${HF_REPO};revision=${HF_REVISION};size=${shard_bytes}" ]]; then
            REUSABLE_KIB=$((REUSABLE_KIB + $(du -k "${shard_path}" | awk '{print $1}')))
        fi
    fi
done
if [[ -r "${ACTIVE_DOWNLOAD_STAMP}" ]]; then
    ACTIVE_DOWNLOAD="$(<"${ACTIVE_DOWNLOAD_STAMP}")"
    for shard in "${SHARDS[@]}"; do
        if [[ "${ACTIVE_DOWNLOAD}" == "repo=${HF_REPO};revision=${HF_REVISION};shard=${shard}" ]]; then
            # Downloads are serialized, so files under the dedicated local-dir
            # cache belong to this explicitly recorded pinned shard.
            INCOMPLETE_KIB="$(find "${MODEL_ROOT}" -type f -name '*.incomplete' \
                -printf '%k\n' 2>/dev/null | awk '{sum += $1} END {print sum + 0}')"
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
   [[ "$(sed -n '1p' "${BUILD_STAMP}")" == "commit=${LLAMA_COMMIT};cuda=${EARLY_NVCC_VERSION};arch=120a-real;gcc=13;profile=qwen38-v1" ]] && \
   [[ "$(sed -n '2p' "${BUILD_STAMP}")" == "$(sha256sum "${SERVER_BIN}" | awk '{print $1}')" ]]; then
    REUSABLE_KIB=$((REUSABLE_KIB + $(du -sk "${LLAMA_DIR}" | awk '{print $1}')))
fi
if [[ -x /usr/local/cuda-13.2/bin/nvcc ]] && \
   [[ "$(stat -c '%d' /usr/local/cuda-13.2)" == "${ROOT_DEVICE}" ]] && \
   /usr/local/cuda-13.2/bin/nvcc --version 2>/dev/null | grep -Eq 'release 13\.2([,[:space:]]|$)'; then
    REUSABLE_KIB=$((REUSABLE_KIB + $(du -sk /usr/local/cuda-13.2 | awk '{print $1}')))
fi
AVAILABLE_CAPACITY_KIB=$((ROOT_FREE_KIB + REUSABLE_KIB))
(( AVAILABLE_CAPACITY_KIB >= MIN_CAPACITY_KIB )) || \
    die "The filesystem needs 130 GiB available for this deployment (free space plus verified reusable artifacts); found $((AVAILABLE_CAPACITY_KIB / 1024 / 1024)) GiB."

GPU_TOTAL_MIB="$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
(( GPU_TOTAL_MIB >= 30000 )) || die "GPU 0 has only ${GPU_TOTAL_MIB} MiB VRAM; this configuration requires a 32 GB-class RTX 5090."
log "GPU: ${GPU_NAME}; compute capability: ${GPU_CC:-not reported}; driver: ${DRIVER_VERSION}"
log "RAM: $((TOTAL_RAM_KIB / 1024 / 1024)) GiB usable; deployment capacity: $((AVAILABLE_CAPACITY_KIB / 1024 / 1024)) GiB"

stage 2 "System dependencies"

export DEBIAN_FRONTEND=noninteractive
apt-get update
packages=(
    build-essential ca-certificates ccache cmake curl g++-13 gcc-13 git jq
    logrotate ninja-build openssl procps python3 python3-pip python3-venv
    tmux util-linux wget
)
missing_packages=()
for package in "${packages[@]}"; do
    dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q 'ok installed' || \
        missing_packages+=("${package}")
done
if ((${#missing_packages[@]})); then
    apt-get install -y --no-install-recommends "${missing_packages[@]}"
else
    log "All required Ubuntu packages are already installed."
fi

stage 3 "CUDA"

CUDA_HOME=""
if [[ -x /usr/local/cuda-13.2/bin/nvcc ]] && \
   /usr/local/cuda-13.2/bin/nvcc --version | grep -Eq 'release 13\.2([,[:space:]]|$)'; then
    CUDA_HOME="/usr/local/cuda-13.2"
elif command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -Eq 'release 13\.2([,[:space:]]|$)'; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
    log "CUDA Toolkit 13.2 is missing; installing the toolkit-only NVIDIA package (no driver meta-package)."
    if ! dpkg-query -W -f='${Status}' cuda-keyring 2>/dev/null | grep -q 'ok installed'; then
        CUDA_KEYRING_DEB="$(mktemp /tmp/qwen38.cuda-keyring.XXXXXX.deb)"
        TEMP_FILES+=("${CUDA_KEYRING_DEB}")
        curl --fail --location --retry 5 --retry-all-errors \
            --output "${CUDA_KEYRING_DEB}" "${CUDA_KEYRING_URL}"
        dpkg -i "${CUDA_KEYRING_DEB}"
    fi
    apt-get update
    apt-cache show cuda-toolkit-13-2 >/dev/null 2>&1 || \
        die "NVIDIA's Ubuntu 24.04 repository does not currently offer cuda-toolkit-13-2. No driver changes were made."
    apt-get install -y --no-install-recommends cuda-toolkit-13-2
    CUDA_HOME="/usr/local/cuda-13.2"
fi

NVCC="${CUDA_HOME}/bin/nvcc"
[[ -x "${NVCC}" ]] || die "CUDA 13.2 installation completed without ${NVCC}."
"${NVCC}" --version | grep -Eq 'release 13\.2([,[:space:]]|$)' || \
    die "The selected nvcc is not CUDA 13.2: $("${NVCC}" --version | tail -n1)"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
log "Using $("${NVCC}" --version | tail -n1) from ${CUDA_HOME}."

stage 4 "llama.cpp"

mkdir -p "${QWEN_ROOT}" "${QWEN_ROOT}/models" "${LOG_DIR}"
if [[ ! -e "${LLAMA_DIR}" ]]; then
    git clone --filter=blob:none --no-checkout "${LLAMA_REPO}" "${LLAMA_DIR}"
elif [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    die "${LLAMA_DIR} exists but is not a git checkout. Move it aside and rerun."
else
    EXISTING_ORIGIN="$(git -C "${LLAMA_DIR}" remote get-url origin 2>/dev/null || true)"
    case "${EXISTING_ORIGIN}" in
        "${LLAMA_REPO}"|"${LLAMA_REPO%.git}"|git@github.com:GenerelSchwerz/llama.cpp.git) ;;
        *) die "${LLAMA_DIR} has unexpected origin '${EXISTING_ORIGIN}'. Refusing to repurpose it." ;;
    esac
fi

if ! git -C "${LLAMA_DIR}" cat-file -e "${LLAMA_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${LLAMA_DIR}" fetch --no-tags --depth=1 origin "${LLAMA_COMMIT}"
fi
if [[ -n "$(git -C "${LLAMA_DIR}" status --porcelain --untracked-files=no)" ]]; then
    die "Tracked changes exist in ${LLAMA_DIR}. Preserve or remove them before the bootstrap can check out the pinned commit."
fi
git -C "${LLAMA_DIR}" checkout --detach "${LLAMA_COMMIT}"
ACTUAL_COMMIT="$(git -C "${LLAMA_DIR}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${LLAMA_COMMIT}" ]] || \
    die "llama.cpp checkout verification failed: expected ${LLAMA_COMMIT}, got ${ACTUAL_COMMIT}."
log "Pinned llama.cpp fork at ${ACTUAL_COMMIT}."

stage 5 "Build"

NVCC_VERSION="$("${NVCC}" --version | tail -n1)"
BUILD_SIGNATURE="commit=${LLAMA_COMMIT};cuda=${NVCC_VERSION};arch=120a-real;gcc=13;profile=qwen38-v1"
EXISTING_SIGNATURE=""
EXISTING_BINARY_HASH=""
if [[ -r "${BUILD_STAMP}" ]]; then
    EXISTING_SIGNATURE="$(sed -n '1p' "${BUILD_STAMP}")"
    EXISTING_BINARY_HASH="$(sed -n '2p' "${BUILD_STAMP}")"
fi
CURRENT_BINARY_HASH=""
[[ -x "${SERVER_BIN}" ]] && CURRENT_BINARY_HASH="$(sha256sum "${SERVER_BIN}" | awk '{print $1}')"
if [[ -x "${SERVER_BIN}" && "${EXISTING_SIGNATURE}" == "${BUILD_SIGNATURE}" && \
      -n "${EXISTING_BINARY_HASH}" && "${CURRENT_BINARY_HASH}" == "${EXISTING_BINARY_HASH}" ]]; then
    log "Reusing the verified pinned llama-server build."
else
    cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
        -DCMAKE_CUDA_COMPILER="${NVCC}" \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
        -DCMAKE_CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
        -DCMAKE_CUDA_ARCHITECTURES=120a-real \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=OFF \
        -DGGML_CUDA_FORCE_MMQ=OFF \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        -DGGML_BACKEND_DL=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_CCACHE=ON
    cmake --build "${BUILD_DIR}" --target llama-server --parallel "$(nproc)"
    [[ -x "${SERVER_BIN}" ]] || die "Build finished without ${SERVER_BIN}."
    printf '%s\n%s\n' "${BUILD_SIGNATURE}" "$(sha256sum "${SERVER_BIN}" | awk '{print $1}')" >"${BUILD_STAMP}"
    chmod 600 "${BUILD_STAMP}"
fi

SERVER_HELP="$("${SERVER_BIN}" --help 2>&1 || true)"
required_flags=(
    --offline --spec-type --ctx-size --batch-size --ubatch-size --parallel
    --threads --gpu-layers --flash-attn --fit --load-mode --lazy-mode
    --moe-expert-cache-size --cache-type-k --cache-type-v --kv-offload
    --cache-ram --jinja --no-warmup --experimental-logs --alias
    --api-key-file --host --port
)
for required_flag in "${required_flags[@]}"; do
    grep -Fq -- "${required_flag}" <<<"${SERVER_HELP}" || \
        die "Pinned llama-server does not advertise required option ${required_flag}; refusing an unsafe partial configuration."
done

stage 6 "Model download"

VENV_DIR="${QWEN_ROOT}/.venv-huggingface"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
fi
if [[ ! -x "${VENV_DIR}/bin/hf" ]] || \
   ! "${VENV_DIR}/bin/python" -c 'import huggingface_hub, hf_xet' >/dev/null 2>&1; then
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install --upgrade huggingface_hub hf_xet
fi

mkdir -p "${MODEL_DIR}"
all_shards_valid=true
for shard in "${SHARDS[@]}"; do
    if [[ ! -s "${MODEL_ROOT}/${shard}" ]] || \
       (( $(stat -c '%s' "${MODEL_ROOT}/${shard}" 2>/dev/null || printf '0') <= 1048576 )); then
        all_shards_valid=false
    fi
done
MODEL_SIGNATURE=""
if [[ "${all_shards_valid}" == true ]]; then
    MODEL_SIGNATURE="repo=${HF_REPO};revision=${HF_REVISION}"
    for shard in "${SHARDS[@]}"; do
        MODEL_SIGNATURE+=";${shard}=$(stat -c '%s' "${MODEL_ROOT}/${shard}")"
    done
fi
EXISTING_MODEL_SIGNATURE=""
[[ -r "${MODEL_STAMP}" ]] && EXISTING_MODEL_SIGNATURE="$(<"${MODEL_STAMP}")"
global_model_valid=false
if [[ "${all_shards_valid}" == true && "${EXISTING_MODEL_SIGNATURE}" == "${MODEL_SIGNATURE}" ]]; then
    global_model_valid=true
fi

export HF_HUB_DISABLE_XET=0
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_XET_HIGH_PERFORMANCE=1
for shard in "${SHARDS[@]}"; do
    shard_path="${MODEL_ROOT}/${shard}"
    shard_stamp="${shard_path}.qwen38-revision"
    shard_bytes="$(stat -c '%s' "${shard_path}" 2>/dev/null || printf '0')"
    expected_shard_stamp="repo=${HF_REPO};revision=${HF_REVISION};size=${shard_bytes}"
    if [[ "${global_model_valid}" == true ]] || \
       [[ "${shard_bytes}" -gt 1048576 && -r "${shard_stamp}" && "$(<"${shard_stamp}")" == "${expected_shard_stamp}" ]]; then
        log "Verified model shard already present: ${shard##*/}"
        atomic_write "${shard_stamp}" 600 <<<"${expected_shard_stamp}"
        continue
    fi

    download_ok=false
    atomic_write "${ACTIVE_DOWNLOAD_STAMP}" 600 \
        <<<"repo=${HF_REPO};revision=${HF_REVISION};shard=${shard}"
    for attempt in 1 2 3; do
        log "Pinned download ${shard##*/}, attempt ${attempt}/3 (safe to interrupt and rerun)."
        if "${VENV_DIR}/bin/hf" download "${HF_REPO}" "${shard}" \
            --revision "${HF_REVISION}" --local-dir "${MODEL_ROOT}"; then
            download_ok=true
            break
        fi
        (( attempt == 3 )) || sleep $((attempt * 10))
    done
    [[ "${download_ok}" == true ]] || die "Model shard download failed after three resumable attempts: ${shard}"
    [[ -s "${shard_path}" ]] || die "Downloaded model shard is absent or empty: ${shard_path}"
    shard_bytes="$(stat -c '%s' "${shard_path}")"
    (( shard_bytes > 1048576 )) || die "Downloaded model shard is implausibly small: ${shard_path}"
    expected_shard_stamp="repo=${HF_REPO};revision=${HF_REVISION};size=${shard_bytes}"
    atomic_write "${shard_stamp}" 600 <<<"${expected_shard_stamp}"
    rm -f -- "${ACTIVE_DOWNLOAD_STAMP}"
done
for shard in "${SHARDS[@]}"; do
    shard_path="${MODEL_ROOT}/${shard}"
    [[ -s "${shard_path}" ]] || die "Expected model shard is absent or empty: ${shard_path}"
    (( $(stat -c '%s' "${shard_path}") > 1048576 )) || \
        die "Model shard is implausibly small (possibly an LFS pointer): ${shard_path}"
done
[[ -s "${MODEL_PATH}" ]] || die "The required main model path does not resolve: ${MODEL_PATH}"
MODEL_SIGNATURE="repo=${HF_REPO};revision=${HF_REVISION}"
for shard in "${SHARDS[@]}"; do
    MODEL_SIGNATURE+=";${shard}=$(stat -c '%s' "${MODEL_ROOT}/${shard}")"
done
atomic_write "${MODEL_STAMP}" 600 <<<"${MODEL_SIGNATURE}"
rm -f -- "${ACTIVE_DOWNLOAD_STAMP}"

stage 7 "Production configuration"

if [[ -e "${API_KEY_FILE}" || -L "${API_KEY_FILE}" ]]; then
    [[ -f "${API_KEY_FILE}" && -s "${API_KEY_FILE}" && ! -L "${API_KEY_FILE}" ]] || \
        die "Existing API key path is not a non-empty regular file: ${API_KEY_FILE}"
    chmod 600 "${API_KEY_FILE}"
    log "Preserving the existing API key."
else
    API_KEY="$(openssl rand -hex 32)"
    atomic_write "${API_KEY_FILE}" 600 <<<"${API_KEY}"
    unset API_KEY
    log "Generated a new 256-bit API key."
fi
API_KEY_VALUE="$(awk 'NF && $1 !~ /^#/ {print; exit}' "${API_KEY_FILE}")"
[[ -n "${API_KEY_VALUE}" ]] || \
    die "The existing API key file contains no usable non-comment key; it was preserved unchanged."

touch "${LOG_FILE}"
chmod 600 "${LOG_FILE}"

atomic_write "${RUNNER}" 755 <<'RUNNER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES=0
export LLAMA_ATTN_ROT_DISABLE=1
export COMPILE_OFF=1
exec /root/qwen38/llama.cpp/build-qwen38/bin/llama-server \
    --offline \
    --model /root/qwen38/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf \
    --spec-type none \
    -c 131072 \
    -b 4096 \
    -ub 512 \
    -np 1 \
    -t 8 \
    -ngl all \
    -fa on \
    -fit off \
    --load-mode none \
    --lazy-mode on \
    --moe-expert-cache-size 192 \
    -ctk q8_0 \
    -ctv q8_0 \
    -kvo \
    --cache-ram 0 \
    --jinja \
    --no-warmup \
    --experimental-logs \
    --alias qwen3.8-flash \
    --api-key-file /root/qwen38/api-key.txt \
    --host 0.0.0.0 \
    --port 11434
RUNNER_EOF

atomic_write "${TMUX_RUNNER}" 755 <<'TMUX_RUNNER_EOF'
#!/usr/bin/env bash
set -uo pipefail
readonly LOG_FILE=/root/qwen38/logs/qwen38-production.log
while :; do
    printf '[%s] tmux supervisor starting llama-server\n' "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
    /root/qwen38/run-qwen38.sh >>"${LOG_FILE}" 2>&1
    result=$?
    printf '[%s] llama-server exited with status %s; restarting in 10 seconds\n' \
        "$(date --iso-8601=seconds)" "${result}" >>"${LOG_FILE}"
    sleep 10
done
TMUX_RUNNER_EOF

if systemd_usable; then
    SUPERVISOR_MODE="systemd"
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
    systemctl enable "${SERVICE_NAME}"
else
    SUPERVISOR_MODE="tmux"
fi
atomic_write "${MODE_FILE}" 600 <<<"${SUPERVISOR_MODE}"

atomic_write "${CONTROL_SCRIPT}" 755 <<'CONTROL_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

readonly SERVER_BIN=/root/qwen38/llama.cpp/build-qwen38/bin/llama-server
readonly MODEL_PATH=/root/qwen38/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
readonly TMUX_RUNNER=/root/qwen38/run-qwen38-tmux.sh
readonly MODE_FILE=/root/qwen38/supervisor-mode
readonly LOG_FILE=/root/qwen38/logs/qwen38-production.log
readonly SERVICE=qwen38.service
readonly TMUX_SESSION=qwen38-prod
readonly MIN_FREE_VRAM_MIB=30000

systemd_usable() {
    [[ -d /run/systemd/system ]] && systemctl show-environment >/dev/null 2>&1
}

mode() {
    [[ -r "${MODE_FILE}" ]] && head -n1 "${MODE_FILE}" || printf 'tmux\n'
}

relevant_pids() {
    local proc pid executable commandline resolved_server process_name
    resolved_server="$(readlink -f "${SERVER_BIN}")"
    for proc in /proc/[0-9]*; do
        pid="${proc#/proc/}"
        [[ -r "${proc}/cmdline" ]] || continue
        executable="$(readlink -f "${proc}/exe" 2>/dev/null || true)"
        commandline="$(tr '\0' ' ' <"${proc}/cmdline" 2>/dev/null || true)"
        process_name="$(cat "${proc}/comm" 2>/dev/null || true)"
        # Accept the managed path (including a replaced/deleted old inode), or
        # another llama-server only when its model/alias/port identifies this service.
        if [[ "${executable}" != "${resolved_server}" && \
              "${executable}" != "${resolved_server} (deleted)" && \
              "${process_name}" != "llama-server" ]]; then
            continue
        fi
        if [[ "${commandline}" == *"${MODEL_PATH}"* || \
              "${commandline}" == *"qwen3.8-flash"* || \
              "${commandline}" == *"--port 11434"* || \
              "${commandline}" == *"--port=11434"* ]]; then
            printf '%s\n' "${pid}"
        fi
    done
}

stop_managers() {
    if systemd_usable && systemctl cat "${SERVICE}" >/dev/null 2>&1; then
        systemctl stop "${SERVICE}" || true
    fi
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        tmux kill-session -t "${TMUX_SESSION}"
    fi
}

wait_for_gpu_pid_release() {
    local deadline pid gpu_pid stale_gpu_owner
    local -a target_pids=("$@") gpu_pids=()
    ((${#target_pids[@]})) || return 0
    deadline=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
        stale_gpu_owner=false
        for pid in "${target_pids[@]}"; do
            for gpu_pid in "${gpu_pids[@]}"; do
                if [[ "${pid}" == "${gpu_pid//[[:space:]]/}" ]]; then
                    stale_gpu_owner=true
                    break 2
                fi
            done
        done
        [[ "${stale_gpu_owner}" == false ]] && return 0
        sleep 1
    done
    printf 'A stopped llama-server PID is still present in the NVIDIA compute process table after 30 seconds.\n' >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true
    return 1
}

stop_relevant_processes() {
    local deadline
    local -a pids=() target_pids=()
    mapfile -t pids < <(relevant_pids)
    ((${#pids[@]})) || return 0
    target_pids=("${pids[@]}")

    printf 'Stopping managed llama-server PID(s): %s\n' "${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
        mapfile -t pids < <(relevant_pids)
        ((${#pids[@]} == 0)) && break
        sleep 1
    done
    mapfile -t pids < <(relevant_pids)
    if ((${#pids[@]})); then
        printf 'Graceful stop timed out; force-stopping managed PID(s): %s\n' "${pids[*]}" >&2
        kill -KILL "${pids[@]}" 2>/dev/null || true
        sleep 2
    fi
    mapfile -t pids < <(relevant_pids)
    ((${#pids[@]} == 0)) || {
        printf 'Managed llama-server PID(s) survived stop: %s\n' "${pids[*]}" >&2
        return 1
    }
    wait_for_gpu_pid_release "${target_pids[@]}"
}

verify_no_stale_gpu_owner() {
    local -a stale=()
    mapfile -t stale < <(relevant_pids)
    ((${#stale[@]} == 0)) || {
        printf 'A managed server still exists: %s\n' "${stale[*]}" >&2
        return 1
    }
}

wait_for_vram() {
    local deadline free_mib processes
    deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
        free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | xargs)"
        (( free_mib >= MIN_FREE_VRAM_MIB )) && return 0
        sleep 2
    done
    free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | xargs)"
    processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
    printf 'Only %s MiB GPU memory is free; at least %s MiB is required before startup.\n' \
        "${free_mib}" "${MIN_FREE_VRAM_MIB}" >&2
    printf 'GPU compute owners:\n%s\n' "${processes:-none reported}" >&2
    return 1
}

stop_all() {
    local -a original_pids=()
    mapfile -t original_pids < <(relevant_pids)
    stop_managers
    stop_relevant_processes
    wait_for_gpu_pid_release "${original_pids[@]}"
    verify_no_stale_gpu_owner
}

start_server() {
    [[ -x "${SERVER_BIN}" ]] || { printf 'Missing server binary: %s\n' "${SERVER_BIN}" >&2; return 1; }
    [[ -s "${MODEL_PATH}" ]] || { printf 'Missing model: %s\n' "${MODEL_PATH}" >&2; return 1; }
    stop_all
    wait_for_vram
    if [[ "$(mode)" == systemd ]]; then
        systemd_usable || { printf 'Configured for systemd, but systemd is unavailable. Re-run bootstrap.sh to select tmux.\n' >&2; return 1; }
        systemctl start "${SERVICE}"
        systemctl is-active --quiet "${SERVICE}"
    else
        command -v tmux >/dev/null 2>&1 || { printf 'tmux is unavailable.\n' >&2; return 1; }
        tmux new-session -d -s "${TMUX_SESSION}" "${TMUX_RUNNER}"
        sleep 2
        tmux has-session -t "${TMUX_SESSION}" 2>/dev/null
    fi
}

show_status() {
    local -a pids=()
    if [[ "$(mode)" == systemd ]] && systemd_usable; then
        systemctl --no-pager --full status "${SERVICE}" || true
    else
        if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
            printf 'tmux session %s is running.\n' "${TMUX_SESSION}"
        else
            printf 'tmux session %s is stopped.\n' "${TMUX_SESSION}"
        fi
    fi
    mapfile -t pids < <(relevant_pids)
    if ((${#pids[@]})); then
        printf 'Managed llama-server PID(s): %s\n' "${pids[*]}"
        return 0
    fi
    printf 'No managed llama-server process is running.\n'
    return 3
}

action="${1:-status}"
case "${action}" in
    start) start_server ;;
    stop) stop_all ;;
    restart) stop_all; start_server ;;
    status) show_status ;;
    logs) exec tail -n 200 -F "${LOG_FILE}" ;;
    pid) relevant_pids ;;
    *) printf 'Usage: %s {start|stop|restart|status|logs}\n' "$0" >&2; exit 64 ;;
esac
CONTROL_EOF

atomic_write /etc/logrotate.d/qwen38 644 <<'LOGROTATE_EOF'
/root/qwen38/logs/qwen38-production.log {
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

stage 8 "Server startup"

"${CONTROL_SCRIPT}" restart
log "Server launched using ${SUPERVISOR_MODE}; waiting for model load."

stage 9 "Verification"

AUTH_HEADER="$(mktemp /tmp/qwen38.auth.XXXXXX)"
MODELS_RESPONSE="$(mktemp /tmp/qwen38.models.XXXXXX.json)"
CHAT_RESPONSE="$(mktemp /tmp/qwen38.chat.XXXXXX.json)"
TEMP_FILES+=("${AUTH_HEADER}" "${MODELS_RESPONSE}" "${CHAT_RESPONSE}")
chmod 600 "${AUTH_HEADER}" "${MODELS_RESPONSE}" "${CHAT_RESPONSE}"
printf 'Authorization: Bearer %s\n' "${API_KEY_VALUE}" >"${AUTH_HEADER}"

deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
healthy=false
while (( SECONDS < deadline )); do
    if curl --silent --show-error --fail --max-time 10 \
        --header @"${AUTH_HEADER}" \
        --output "${MODELS_RESPONSE}" \
        http://127.0.0.1:11434/v1/models 2>/dev/null && \
       jq -e '.data | type == "array"' "${MODELS_RESPONSE}" >/dev/null 2>&1; then
        healthy=true
        break
    fi
    sleep 5
done
if [[ "${healthy}" != true ]]; then
    "${CONTROL_SCRIPT}" status || true
    tail -n 120 "${LOG_FILE}" >&2 || true
    die "The authenticated /v1/models endpoint did not become healthy within ${HEALTH_TIMEOUT_SECONDS} seconds."
fi
UNAUTHENTICATED_STATUS="$(curl --silent --output /dev/null --write-out '%{http_code}' \
    --max-time 10 http://127.0.0.1:11434/v1/models || true)"
[[ "${UNAUTHENTICATED_STATUS}" != "200" ]] || \
    die "Security verification failed: /v1/models accepted an unauthenticated request."
jq -e '.data[]? | select(.id == "qwen3.8-flash")' "${MODELS_RESPONSE}" >/dev/null || \
    die "The API is healthy but does not advertise the required qwen3.8-flash alias."

CHAT_PAYLOAD='{"model":"qwen3.8-flash","messages":[{"role":"user","content":"Reply with exactly the word READY."}],"temperature":0,"max_tokens":128}'
curl --silent --show-error --fail --max-time 600 \
    --header @"${AUTH_HEADER}" \
    --header 'Content-Type: application/json' \
    --data "${CHAT_PAYLOAD}" \
    --output "${CHAT_RESPONSE}" \
    http://127.0.0.1:11434/v1/chat/completions || {
        tail -n 120 "${LOG_FILE}" >&2 || true
        die "The OpenAI-compatible chat completion request failed."
    }
if ! jq -e '
    (.choices | type == "array" and length > 0) and
    (((.choices[0].message.content // "") | length > 0) or
     ((.choices[0].message.reasoning_content // "") | length > 0))
' "${CHAT_RESPONSE}" >/dev/null; then
    jq . "${CHAT_RESPONSE}" >&2 || true
    die "Chat completion returned neither content nor reasoning_content."
fi

GPU_USED_MIB="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | xargs)"
GPU_FREE_MIB="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | xargs)"
RAM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
SERVER_PID="$("${CONTROL_SCRIPT}" pid | head -n1)"
[[ -n "${SERVER_PID}" ]] || die "Verification succeeded but the managed server PID could not be resolved."
if [[ "${SUPERVISOR_MODE}" == "systemd" ]]; then
    if ! SUPERVISOR_STATUS="$(systemctl is-active "${SERVICE_NAME}" 2>/dev/null)"; then
        SUPERVISOR_STATUS="unknown"
    fi
elif tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    SUPERVISOR_STATUS="running"
else
    SUPERVISOR_STATUS="unknown"
fi

printf '\nDEPLOYMENT COMPLETE\n'
printf '  GPU model:        %s\n' "${GPU_NAME}"
printf '  VRAM used/free:   %s MiB / %s MiB\n' "${GPU_USED_MIB}" "${GPU_FREE_MIB}"
printf '  RAM available:    %s MiB\n' "$((RAM_AVAILABLE_KIB / 1024))"
printf '  Listening:        0.0.0.0:11434\n'
printf '  Model alias:      qwen3.8-flash\n'
printf '  Context size:     131072\n'
printf '  MoE cache size:   192\n'
printf '  Server PID:       %s\n' "${SERVER_PID}"
printf '  Supervisor:       %s (%s)\n' "${SUPERVISOR_MODE}" "${SUPERVISOR_STATUS}"
if (( GPU_FREE_MIB < 1024 )); then
    printf 'WARNING: GPU headroom is dangerously low (%s MiB free). Inspect other GPU consumers.\n' "${GPU_FREE_MIB}" >&2
elif (( GPU_FREE_MIB < 2048 )); then
    printf 'WARNING: GPU headroom is lower than the expected ~3 GiB (%s MiB free).\n' "${GPU_FREE_MIB}" >&2
fi

SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
printf '\nContainer API:\nhttp://%s:11434/v1\n' "${SERVER_IP:-SERVER_IP}"
printf '\nModel:\nqwen3.8-flash\n'
printf '\nAPI key file:\n%s\n' "${API_KEY_FILE}"
printf '\nAPI KEY (SECRET — store it securely and never share it publicly):\n%s\n' "${API_KEY_VALUE}"
printf '\nProvider-side port forwarding and cloud/firewall configuration are outside this script.\n'
printf 'Manage the server with: %s {start|stop|restart|status|logs}\n' "${CONTROL_SCRIPT}"
