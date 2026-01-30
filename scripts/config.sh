#!/bin/bash
# =============================================================================
# config.sh - User and cluster settings for DSTAR training
# =============================================================================
# Source this file from launch.sh and restart scripts.
# Override any variable before sourcing, or set via environment.
#
# NOTE: WANDB_RUN_GROUP is per-experiment and belongs in launch.sh, not here.
# =============================================================================

# --- WANDB Configuration ---
# Set these in your environment or override before sourcing.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-danastar}"

# --- Data Paths ---
export DATASETS_DIR="${DATASETS_DIR:-./src/data/datasets/}"
export RESULTS_BASE_FOLDER="${RESULTS_BASE_FOLDER:-./exps}"

# --- Caches ---
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/tiktoken_cache}"
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/hf}"

# --- PyTorch Memory ---
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Cluster-Specific Module Loads and Venv ---
# Detect cluster from hostname and load appropriate modules.
# Override DSTAR_CLUSTER to force a specific configuration.
DSTAR_CLUSTER="${DSTAR_CLUSTER:-auto}"

if [ "$DSTAR_CLUSTER" = "auto" ]; then
    HOSTNAME_SHORT=$(hostname -s 2>/dev/null || echo "unknown")
    case "$HOSTNAME_SHORT" in
        nar*|narval*|blg*|beluga*|cedar*)
            DSTAR_CLUSTER="narval"
            ;;
        tamia*|tam*)
            DSTAR_CLUSTER="tamia"
            ;;
        fir*|cedar*)
            DSTAR_CLUSTER="fir"
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
        export DATASETS_DIR="${DATASETS_DIR:-$HOME/scratch/datasets}"
        ;;
    tamia)
        module load python/3.12 2>/dev/null || true
        source "$HOME/danastarenv/bin/activate" 2>/dev/null || true
        ;;
    fir)
        module load python/3.12 2>/dev/null || true
        source "$HOME/danastarenv/bin/activate" 2>/dev/null || true
        ;;
    math-slurm)
        module load miniconda/miniconda-winter2025 2>/dev/null || true
        ;;
    local)
        # No module loads needed for local development
        ;;
esac

echo "[config.sh] Cluster: $DSTAR_CLUSTER"
