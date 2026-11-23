#!/bin/bash
#SBATCH --account=aip-cypaquet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=2:30:00
#SBATCH --output=/scratch/e/epaq/adana-runs/slurm/%x_%j.out
# =============================================================================
# DDP against FSDP on one tamia node (4 x H100): step time and peak memory, at matched and at doubled micro-batch, global batch
# 512, 24 and 28 heads.  AdamW (supported by both backends) isolates the backend; one d-muon DDP run at 24 heads checks that tamia
# matches rorqual (2.62 s/step for b512_dmuon_h24, micro-batch 16).  Each run: 40 steps, no eval, no wandb, no checkpoints kept.
# usage: sbatch -J bench_backend scripts/tamia_bench_backend.sh
# =============================================================================
set -uo pipefail
ADANA_DIR="${ADANA_DIR:-/scratch/e/epaq/adana}"
cd "$ADANA_DIR"
export DSTAR_CLUSTER=tamia
source scripts/config.sh
export WANDB_MODE=disabled
export XDG_CACHE_HOME="/scratch/e/epaq/.cache" MPLCONFIGDIR="/scratch/e/epaq/.cache/matplotlib"
export OMP_NUM_THREADS=4
export RESULTS_BASE_FOLDER="/scratch/e/epaq/adana-bench/${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS_BASE_FOLDER"
LOGDIR="$RESULTS_BASE_FOLDER/logs"; mkdir -p "$LOGDIR"
SUMMARY="$RESULTS_BASE_FOLDER/summary.tsv"
printf "heads\topt\tbackend\ttag\tmicro\tacc\tstatus\tmedian_iter_dt_s\tpeak_alloc_GiB\tpeak_reserved_GiB\n" > "$SUMMARY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() {   # heads opt backend micro acc [tag extra-args...]
    local h=$1 opt=$2 be=$3 mb=$4 acc=$5 tag=${6:-}; shift 5; [ $# -gt 0 ] && shift
    local name="bench_h${h}_${opt}_${be}${tag:+_$tag}_mb${mb}"
    local log="$LOGDIR/$name.log"
    echo "[$(date +%T)] $name"
    bash scripts/launch.sh --arch enoki --heads "$h" --opt "$opt" --nproc 4 --distributed_backend "$be" \
        --batch_size "$mb" --acc_steps "$acc" --iterations 40 --warmup_steps 5 --eval_interval 100000 --log_interval 1 \
        --latest_ckpt_interval 0 --permanent_ckpt_interval 0 --no_wandb --no_auto_resume "$@" > "$log" 2>&1
    local status=ok dt mem res
    grep -qi "out of memory\|OutOfMemoryError" "$log" && status=OOM
    [ "$status" = ok ] && ! grep -q "Train: Iter=" "$log" && status=FAILED
    dt=$(grep -o "Train: Iter=[0-9]* .*iter_dt=[0-9.e+-]*s" "$log" | awk -F'Iter=' '{split($2,a," "); if (a[1]+0 >= 20) print}' \
         | grep -o "iter_dt=[0-9.e+-]*" | cut -d= -f2 | sort -g | awk '{v[NR]=$1} END{if (NR) print v[int((NR+1)/2)]}')
    mem=$(grep -o "peak_allocated=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
    res=$(grep -o "peak_reserved=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$h" "$opt" "$be" "${tag:--}" "$mb" "$acc" "$status" "${dt:--}" "${mem:--}" "${res:--}" >> "$SUMMARY"
    if [ "$status" != ok ]; then echo "  -> $status; end of $log:"; grep -v "^\s*$" "$log" | tail -15 | cut -c1-300; fi
    # drop checkpoint folders only (the logs live in $LOGDIR)
    find "$RESULTS_BASE_FOLDER" -path "$LOGDIR" -prune -o -type d -name "ckpts" -prune -exec rm -rf {} + 2>/dev/null
}

# global batch 512 = micro x acc (launch.sh: --batch_size x --acc_steps is the global batch).  RUNSET picks the list.
case "${RUNSET:-backend}" in
backend)   # DDP against FSDP (job 776257)
    run 24 d-muon nccl 16 32
    run 24 adamw  nccl 16 32
    run 24 adamw  fsdp 16 32
    run 24 adamw  fsdp 32 16
    run 24 adamw  nccl 32 16
    run 28 adamw  nccl  8 64
    run 28 adamw  fsdp  8 64
    run 28 adamw  fsdp 16 32
    run 28 adamw  nccl 16 32 ;;
fsdp2)     # FSDP2 with one gather per optimizer step, bf16 gathers, fp32 accumulation and reduce-scatter (the new defaults),
           # and two runs that each restore one old setting
    run 24 adamw fsdp 16 32 step
    run 28 adamw fsdp  8 64 step
    run 28 adamw fsdp 16 32 step
    run 28 adamw fsdp 16 32 step_fp32par --fsdp_param_dtype float32
    run 28 adamw fsdp 16 32 micro_bf16 --fsdp_reshard microstep
    run 24 adamw fsdp 32 16 step ;;
parity)    # same model, data and schedule on every backend: 60 steps, warm-up over 59, negligible weight decay (omega 1e-3);
           # DDP and new FSDP twice each (run-to-run noise), old FSDP once.  FSDP starts from rank 0's weights, as DDP does.
    P="--iterations 60 --warmup_steps 59 --omega 0.001"
    run 24 adamw nccl 16 32 par1 $P
    run 24 adamw nccl 16 32 par2 $P
    run 24 adamw fsdp 16 32 par1 $P
    run 24 adamw fsdp 16 32 par2 $P
    run 24 adamw fsdp 16 32 parold $P --fsdp_reshard microstep --fsdp_param_dtype float32 --fsdp_reduce_dtype bfloat16 ;;
dmuon)     # d-muon under FSDP2 as it stands (Newton-Schulz on the sharded DTensors): speed against DDP, and parity on the schedule above
    P="--iterations 60 --warmup_steps 59 --omega 0.001"
    run 24 d-muon fsdp 16 32 step
    run 24 d-muon fsdp 32 16 step
    run 28 d-muon nccl 16 32
    run 28 d-muon fsdp 16 32 step
    run 24 d-muon nccl 16 32 par1 $P
    run 24 d-muon fsdp 16 32 par1 $P ;;
esac

echo "=== summary ($SUMMARY)"
column -t -s $'\t' "$SUMMARY"
