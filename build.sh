#!/usr/bin/env bash
# ==============================================================================
# WEDA-LLM-BASE Pipeline — Build & Launch Script
#
# Usage:
#   ./build.sh              # Build/Pull, start, and enter the container
#   ./build.sh --run        # Start only (skip build/pull, use existing image)
#   ./build.sh --shell      # Attach to already-running container
#   ./build.sh --stop       # Stop and remove the container
#   ./build.sh --test       # Run wise-bench-container.py verifications
#   ./build.sh --help
# ==============================================================================
set -euo pipefail
IFS=$'\n\t'

# ── Path Anchoring ────────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}"

# ── Logging ───────────────────────────────────────────────────────────────────
log_info()    { echo -e "\e[1;36m[INFO]\e[0m  $1"; }
log_success() { echo -e "\e[1;32m[  OK]\e[0m  $1"; }
log_warn()    { echo -e "\e[1;33m[WARN]\e[0m  $1"; }
log_error()   { echo -e "\e[1;31m[FAIL]\e[0m  $1" >&2; exit 1; }

# ── Parse Flags ───────────────────────────────────────────────────────────────
MODE="build"
for arg in "$@"; do
    case "$arg" in
        --run)    MODE="run" ;;
        --serve)  MODE="serve" ;;
        --shell)  MODE="shell" ;;
        --stop)   MODE="stop" ;;
        --test)   MODE="test" ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  (no flag)   Build/Pull the container, start it, wait for hydration, open shell"
            echo "  --run       Start the container (skip pull, use existing image)"
            echo "  --serve     Start the container (skip pull), flush memory caches, and exit without opening a shell"
            echo "  --shell     Attach a shell to the already-running container"
            echo "  --stop      Stop and remove the container"
            echo "  --test      Run wise-bench-container.py verification suite"
            echo "  --help      Show this help message"
            exit 0 ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\033[1;36m"
cat << "EOF"
   ╦ ╦╔═╗╔╦╗╔═╗   ╦  ╦  ╔╦╗  ╔╗ ╔═╗╔═╗╔═╗
   ║║║║╣  ║║╠═╣───║  ║  ║║║──╠╩╗╠═╣╚═╗║╣
   ╚╩╝╚═╝═╩╝╩ ╩   ╩═╝╩═╝╩ ╩  ╚═╝╩ ╩╚═╝╚═╝
          Jetson Thor Optimized  |  Powered by Advantech Co. Ltd
EOF
echo -e "\033[0m"

# ── Load Config from .env ─────────────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

[[ -f "${ENV_FILE}" ]] || log_error ".env not found at ${ENV_FILE}"

set -a
source "${ENV_FILE}"
set +a

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    COMPOSE=()
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    log_warn "HF_TOKEN is empty in .env — gated models will fail to pull."
fi

# ── Static Variable Validation ────────────────────────────────────────────────
REQUIRED_STATIC=(
    WEDA_LLM_IMAGE
    WEDA_LLM_CONTAINER_NAME
    WEDA_LLM_PORT
)

MISSING=()
for var in "${REQUIRED_STATIC[@]}"; do
    [[ -z "${!var:-}" ]] && MISSING+=("${var}")
done
[[ ${#MISSING[@]} -gt 0 ]] && log_error "Missing required vars in .env: ${MISSING[*]}"

# ── Setup Volumes ─────────────────────────────────────────────────────────────
HF_CACHE="${HF_CACHE:-${SCRIPT_DIR}/model_cache/huggingface}"
MOE_CONFIGS_DIR="${MOE_CONFIGS_DIR:-${SCRIPT_DIR}/artifacts/moe-configs}"
TEST_LOGS_DIR="${TEST_LOGS_DIR:-${SCRIPT_DIR}/test_logs}"

mkdir -p "${HF_CACHE}" "${MOE_CONFIGS_DIR}" "${TEST_LOGS_DIR}"

# ── Handle --stop ─────────────────────────────────────────────────────────────
if [[ "${MODE}" == "stop" ]]; then
    echo -e "\033[1;34m[1/1] Stopping container...\033[0m"
    if docker ps -a -q -f name="^${WEDA_LLM_CONTAINER_NAME}$" | grep -q .; then
        "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" down --remove-orphans
        log_success "Container stopped and removed."
    else
        log_warn "Container '${WEDA_LLM_CONTAINER_NAME}' is not running."
    fi
    exit 0
fi

# ── Handle --shell ────────────────────────────────────────────────────────────
if [[ "${MODE}" == "shell" ]]; then
    echo -e "\033[1;34m[1/1] Attaching to running container...\033[0m"
    if docker ps -q -f name="^${WEDA_LLM_CONTAINER_NAME}$" | grep -q .; then
        log_success "Entering container '${WEDA_LLM_CONTAINER_NAME}'..."
        docker exec -it "${WEDA_LLM_CONTAINER_NAME}" /bin/bash || true
    else
        log_error "Container '${WEDA_LLM_CONTAINER_NAME}' is not running. Start it first with: ./build.sh"
    fi
    exit 0
fi

# ── Handle --test ─────────────────────────────────────────────────────────────
if [[ "${MODE}" == "test" ]]; then
    echo -e "\033[1;34m[1/1] Running verification tests...\033[0m"
    "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" run --rm --name "${WEDA_LLM_CONTAINER_NAME}-test" weda-llm \
      python3 /wise-edge/weda-llm-base/tests/wise-bench-container.py \
        --report-dir /wise-edge/weda-llm-base/test_logs
    log_success "Test logs saved to ${TEST_LOGS_DIR}/"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline: preflight → build/pull → launch → shell
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_STEPS=3
[[ "${MODE}" == "run" || "${MODE}" == "serve" ]] && TOTAL_STEPS=2

# ── [1/N] Pre-flight Checks ──────────────────────────────────────────────────
echo -e "\033[1;34m[1/${TOTAL_STEPS}] Pre-flight checks...\033[0m"

command -v docker &> /dev/null || log_error "docker not found."
if [[ ${#COMPOSE[@]} -eq 0 ]]; then
    log_error "Docker Compose not found."
fi

if ! docker info &> /dev/null; then
    log_error "Docker daemon not running or not accessible."
fi
log_success "Docker daemon running."

# ── [2/N] Build or Pull ──────────────────────────────────────────────────────
STEP=2
if [[ "${MODE}" == "build" ]]; then
    echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Synchronizing container image...\033[0m"
    "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
        pull 2>/dev/null || log_warn "Pull failed or image is local-only. Assuming local image exists."
    log_success "Image synchronization complete."
    STEP=$((STEP + 1))
fi

# ── [N/N] Launch ──────────────────────────────────────────────────────────────
echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Launching container...\033[0m"

if [[ "${MODE}" == "serve" ]]; then
    log_info "Flushing host page caches and dentries to reclaim memory..."
    sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' || log_warn "Failed to flush host caches (check sudo permissions)."
fi

docker stop "${WEDA_LLM_CONTAINER_NAME}" &> /dev/null || true
docker rm   "${WEDA_LLM_CONTAINER_NAME}" &> /dev/null || true

"${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
    up -d --build --remove-orphans weda-llm open-webui

log_success "Container '${WEDA_LLM_CONTAINER_NAME}' is live."

# ── Wait for Hydration ────────────────────────────────────────────────────────
echo "  Waiting for environment hydration..."
docker exec "${WEDA_LLM_CONTAINER_NAME}" rm -f /wise-edge/weda-llm-base/.ready 2>/dev/null || true

TIMEOUT=60
elapsed=0
while [[ $elapsed -lt $TIMEOUT ]]; do
    if docker exec "${WEDA_LLM_CONTAINER_NAME}" \
        ls /wise-edge/weda-llm-base/.ready &> /dev/null; then
        printf "\r  [\033[1;32m%s\033[0m] 100%%\n" "$(printf "%*s" 50 | tr ' ' '#')"
        break
    fi
    pct=$(( elapsed * 100 / TIMEOUT ))
    hashes=$(( (pct * 50) / 100 ))
    spaces=$(( 50 - hashes ))
    bar="$(printf "%*s" $hashes "" | tr ' ' '#')"
    empty="$(printf "%*s" $spaces "")"
    printf "\r  [\033[1;36m%s\033[0m%s] %3d%% " "$bar" "$empty" "$pct"
    sleep 2
    elapsed=$(( elapsed + 2 ))
done

if [[ $elapsed -ge $TIMEOUT ]]; then
    echo -e "\n  [\033[1;31mTIMEOUT\033[0m]"
    log_warn "Hydration timeout or .ready file not created. Check: docker logs ${WEDA_LLM_CONTAINER_NAME}"
fi

# ── Shell ─────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "serve" ]]; then
    log_success "Serving container launched in background."
    exit 0
fi

echo -e "\n\033[1;33m+══════════════════════════════════════════════════════════+\033[0m"
echo -e "\033[1;33m|       WEDA-LLM-BASE Pipeline — Shell                     |\033[0m"
echo -e "\033[1;33m+══════════════════════════════════════════════════════════+\033[0m"
echo -e "  Diagnostics:    \033[1;32m./scripts/wise-bench\033[0m"

docker exec -it "${WEDA_LLM_CONTAINER_NAME}" /bin/bash || true