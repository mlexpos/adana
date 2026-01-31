#!/bin/bash
# =============================================================================
# config.sh - Cluster and user settings for DSTAR training
# =============================================================================
# Source this file from launch.sh and restart scripts.
#
# This file handles two things:
#   1. Cluster detection + module loads (same for all users on a cluster)
#   2. User config via scripts/config.local (per-user paths, credentials, etc.)
#
# Users should create scripts/config.local with their settings.
# See scripts/config.local.example for a template.
# =============================================================================

# =============================================================================
# 1. User config file (scripts/config.local)
# =============================================================================
# Source user config first so it can set DATASETS_DIR, WANDB_API_KEY, etc.
# before the defaults below kick in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.local" ]; then
    source "$SCRIPT_DIR/config.local"
fi

# --- Defaults (overridden by scripts/config.local or environment) ---
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-danastar}"
export DATASETS_DIR="${DATASETS_DIR:-./src/data/datasets/}"
export RESULTS_BASE_FOLDER="${RESULTS_BASE_FOLDER:-./exps}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/tiktoken_cache}"
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# =============================================================================
# 2. Cluster detection and module loads
# =============================================================================
DSTAR_CLUSTER="${DSTAR_CLUSTER:-auto}"
HOSTNAME_SHORT=$(hostname -s 2>/dev/null || echo "unknown")

if [ "$DSTAR_CLUSTER" = "auto" ]; then
    case "$HOSTNAME_SHORT" in
        nar*|narval*|blg*|beluga*|cedar*)
            DSTAR_CLUSTER="narval"
            ;;
        tamia*|tam*)
            DSTAR_CLUSTER="tamia"
            ;;
        fir*|cedar*|login3*)
            DSTAR_CLUSTER="fir"
            ;;
        rorqual*|rg*)
            DSTAR_CLUSTER="rorqual"
            ;;
        math*|slurm*)
            DSTAR_CLUSTER="math-slurm"
            ;;
        *)
            DSTAR_CLUSTER="local"
            ;;
    esac
fi

case "$DSTAR_CLUSTER" in
    narval)
        module load arrow/21.0.0 2>/dev/null || true
        module load python/3.13 2>/dev/null || true
        source "$HOME/danastarenv/bin/activate" 2>/dev/null || true
        ;;
    tamia)
        module load python/3.12 2>/dev/null || true
        source "$HOME/danastarenv/bin/activate" 2>/dev/null || true
        ;;
    fir)
        module load python/3.13 2>/dev/null || true
        source "$HOME/danastarenv/bin/activate" 2>/dev/null || true
        ;;
    rorqual)
        module load gcc 2>/dev/null || true
        module load arrow/21.0.0 2>/dev/null || true
        module load python/3.13 2>/dev/null || true
        source "$HOME/danastarenv/bin/activate" 2>/dev/null || true
        ;;
    math-slurm)
        module load miniconda/miniconda-winter2025 2>/dev/null || true
        ;;
    local)
        # No module loads needed for local development
        ;;
esac

echo "[config.sh] Host: $HOSTNAME_SHORT, Cluster: $DSTAR_CLUSTER"
