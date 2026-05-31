#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_deepspeed.sh
#
# Usage:
#   conda activate leva-tts
#   bash scripts/install_deepspeed.sh
#
# What this does:
#   1. Locates nvcc — first in the active conda env, then in system paths,
#      then borrows it from any other conda env found on this machine.
#   2. Installs deepspeed==0.14.4 with DS_BUILD_OPS=0 (no custom CUDA kernel
#      compilation).  DS_BUILD_OPS=0 is enough for inference (init_inference,
#      fp16) which is all leva-tts needs.
#   3. If nvcc is truly absent it prints the one-liner to fix it permanently.
#
# NOTE: DS_BUILD_OPS=0 does NOT skip the CUDA version check that deepspeed
#       runs at metadata-preparation time.  A working nvcc must be visible.
#       The search below covers system paths AND other conda environments.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DeepSpeed 0.14.4 installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Find nvcc (extended search) ────────────────────────────────────────────
# Returns the root directory whose bin/nvcc is a valid executable.
_find_nvcc() {
    # Candidates in priority order
    local dirs=()

    # a) Active conda env first
    [[ -n "${CONDA_PREFIX:-}" ]] && dirs+=("${CONDA_PREFIX}")

    # b) Known system paths (versioned)
    dirs+=(
        /usr/local/cuda-12.4
        /usr/local/cuda-12.1
        /usr/local/cuda-12.3
        /usr/local/cuda-12.2
        /usr/local/cuda-12.0
        /usr/local/cuda-11.8
        /usr/local/cuda
        /usr/cuda
        /opt/cuda
    )

    for d in "${dirs[@]}"; do
        if [[ -f "${d}/bin/nvcc" && -x "${d}/bin/nvcc" ]]; then
            echo "${d}"; return 0
        fi
    done

    # c) Borrow from any other conda environment on this machine
    #    (they often have cuda-nvcc installed as a dependency)
    for nvcc_bin in /usr/local/envs/*/bin/nvcc; do
        if [[ -f "${nvcc_bin}" && -x "${nvcc_bin}" ]]; then
            echo "$(dirname "$(dirname "${nvcc_bin}")")"
            return 0
        fi
    done

    # d) Anything on PATH
    local n
    n=$(command -v nvcc 2>/dev/null || true)
    if [[ -n "${n}" ]]; then
        echo "$(dirname "$(dirname "${n}")")"
        return 0
    fi

    return 1
}

# Always unset a potentially stale CUDA_HOME before searching —
# if it points to a path without nvcc it will cause deepspeed to fail.
unset CUDA_HOME CUDA_PATH 2>/dev/null || true

if found=$(_find_nvcc 2>/dev/null); then
    export CUDA_HOME="${found}"
    export CUDA_PATH="${CUDA_HOME}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    NVCC_VER=$(nvcc --version 2>&1 | grep -oP 'release \K[\d.]+' || echo "?")
    echo "  ✅  nvcc found  : ${CUDA_HOME}/bin/nvcc  (CUDA ${NVCC_VER})"
else
    echo "  ❌  nvcc not found anywhere on this machine."
    echo ""
    echo "  Fix (add nvcc permanently to the leva-tts env):"
    echo "    conda install -n leva-tts -c nvidia cuda-nvcc=12.1"
    echo "  Then re-run this script."
    exit 1
fi

# ── 2. Install deepspeed with DS_BUILD_OPS=0 (inference-only) ─────────────────
# DS_BUILD_OPS=0 skips compilation of custom CUDA training kernels
# (FusedAdam, CPUAdam, etc.) but deepspeed.init_inference() still works.
echo ""
echo "  Installing deepspeed==0.14.4 (DS_BUILD_OPS=0 — no kernel compilation)…"

DS_BUILD_OPS=0 \
CUDA_HOME="${CUDA_HOME}" \
CUDA_PATH="${CUDA_PATH}" \
    pip install deepspeed==0.14.4 --no-build-isolation

# ── 3. Verify ─────────────────────────────────────────────────────────────────
echo ""
python - <<'PYEOF'
import deepspeed, torch
print(f"  deepspeed {deepspeed.__version__}  ✅")
print(f"  torch     {torch.__version__}  CUDA: {torch.cuda.is_available()}")
PYEOF

echo ""
echo "  ✅  Done."
echo ""
echo "  ℹ️   To add nvcc permanently to leva-tts (speeds up future DS installs):"
echo "        conda install -n leva-tts -c nvidia cuda-nvcc=12.1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
