# Batch Size Benchmark Results

**Cluster:** Rorqual (4x H100-80GB per node)
**Architecture:** Enoki (head_dim=64)
**Optimizer:** dana-star-mk4 (kappa=0.85)
**Backend:** FSDP (4 GPUs), with optional tensor parallelism (TP4)
**Sequence length:** 2048

All batch sizes below are **global batch** (the value passed to `--batch_size`).
With FSDP (tp=1, 4 DP ranks), each GPU processes `global_batch / 4` samples.
With FSDP+TP4 (tp=4, 1 DP rank), each GPU processes the full `global_batch`.

## Summary: Max Global Batch, Throughput, and Chinchilla Time Estimates

| Heads | Params | tp=1 max global | tp=1 local batch | tp=1 iter_dt | tp=1 tok/s | Chinchilla days (tp=1) | tp=4 max global | tp=4 local batch | tp=4 iter_dt | tp=4 tok/s |
|-------|--------|----------------|-----------------|-------------|-----------|----------------------|----------------|-----------------|-------------|-----------|
| 28 | 0.99B | 32 | 8 | 607ms | 108K | 2.1 | 32 | 32 | 995ms | 66K |
| 32 | 1.41B | 32 | 8 | 779ms | 84K | 3.9 | 16 | 16 | 770ms | 43K |
| 36 | 1.95B | 32 | 8 | 985ms | 67K | 6.7 | 16 | 16 | 957ms | 34K |
| 40 | 2.62B | 32 | 8 | 1430ms | 46K | 13.2 | 16 | 16 | 1330ms | 25K |
| 44 | 3.42B | 16 | 4 | 920ms | 36K | 22.0 | 8 | 8 | 912ms | 18K |
| 48 | 4.39B | 16 | 4 | 1100ms | 30K | 33.9 | 8 | 8 | 1060ms | 15K |

Tokens/s = `global_batch * 2048 / iter_dt`.

**Chinchilla days** assume 20N total tokens, global_batch=512 (with gradient accumulation), on a single 4x H100 node using tp=1.
Estimated as: `(20 * params) / (tok/s * 86400)`. The iter_dt scales linearly with `acc_steps` (= 512 / max_global_batch).

## Conclusion

**TP4 is strictly worse** for these model sizes on 4x H100. At every head count:

1. **Lower max global batch.** tp=1 can fit equal or larger global batches because FSDP distributes the local batch across 4 GPUs (local = global/4), while tp=4 puts the full global batch on every GPU.

2. **Lower throughput.** Even at matched global batch sizes, tp=1 is 1.5-2x faster due to TP communication overhead.

3. **Higher memory per GPU.** At the same global batch, tp=4 uses ~2x more memory because each GPU holds activations for 4x as many samples.

TP becomes useful only when a model's *parameters* don't fit on a single GPU even with FSDP sharding, which doesn't apply at these scales (up to ~4.4B params on 80GB H100s).

## Recommended Configuration

For Chinchilla-optimal runs (global_batch=512, acc_steps to fill), single 4x H100 node:

| Heads | Params | batch_size | acc_steps | local batch | Backend | Est. days |
|-------|--------|-----------|-----------|------------|---------|-----------|
| 28 | 0.99B | 32 | 16 | 8 | FSDP (tp=1) | 2.1 |
| 32 | 1.41B | 32 | 16 | 8 | FSDP (tp=1) | 3.9 |
| 36 | 1.95B | 32 | 16 | 8 | FSDP (tp=1) | 6.7 |
| 40 | 2.62B | 32 | 16 | 8 | FSDP (tp=1) | 13.2 |
| 44 | 3.42B | 16 | 32 | 4 | FSDP (tp=1) | 22.0 |
| 48 | 4.39B | 16 | 32 | 4 | FSDP (tp=1) | 33.9 |

## Full Results

### tp=1 (FSDP, 4 DP ranks, local_batch = global_batch / 4)

| Heads | Global Batch | Peak Mem (GPU 0) | iter_dt | Status |
|-------|-------------|-----------------|---------|--------|
| 28 | 4 | 8.81 GiB | 255ms | OK |
| 28 | 8 | 12.47 GiB | 296ms | OK |
| 28 | 16 | 19.79 GiB | 402ms | OK |
| 28 | 32 | 34.45 GiB | 607ms | OK |
| 32 | 4 | 11.42 GiB | 300ms | OK |
| 32 | 8 | 16.15 GiB | 366ms | OK |
| 32 | 16 | 25.61 GiB | 505ms | OK |
| 32 | 32 | 44.52 GiB | 779ms | OK |
| 36 | 4 | 14.50 GiB | 355ms | OK |
| 36 | 8 | 20.44 GiB | 443ms | OK |
| 36 | 16 | 32.31 GiB | 627ms | OK |
| 36 | 32 | 56.06 GiB | 985ms | OK |
| 40 | 4 | 18.07 GiB | 418ms | OK |
| 40 | 8 | 25.36 GiB | 535ms | OK |
| 40 | 16 | 39.94 GiB | 763ms | OK |
| 40 | 32 | 69.10 GiB | 1430ms | OK |
| 44 | 4 | 22.20 GiB | 492ms | OK |
| 44 | 8 | 30.98 GiB | 638ms | OK |
| 44 | 16 | 48.54 GiB | 920ms | OK |
| 44 | 32 | -- | -- | OOM |
| 48 | 4 | 26.90 GiB | 574ms | OK |
| 48 | 8 | 37.31 GiB | 749ms | OK |
| 48 | 16 | 58.15 GiB | 1100ms | OK |
| 48 | 32 | -- | -- | OOM |

### tp=4 (FSDP+TP4, 1 DP rank, local_batch = global_batch)

| Heads | Global Batch | Peak Mem (GPU 0) | iter_dt | Status |
|-------|-------------|-----------------|---------|--------|
| 28 | 4 | 11.24 GiB | 404ms | OK |
| 28 | 8 | 18.08 GiB | 449ms | OK |
| 28 | 16 | 32.16 GiB | 625ms | OK |
| 28 | 32 | 60.43 GiB | 995ms | OK |
| 32 | 4 | 14.14 GiB | 478ms | OK |
| 32 | 8 | 22.84 GiB | 531ms | OK |
| 32 | 16 | 40.66 GiB | 770ms | OK |
| 32 | 32 | -- | -- | OOM |
| 36 | 4 | 17.45 GiB | 545ms | OK |
| 36 | 8 | 28.28 GiB | 633ms | OK |
| 36 | 16 | 50.31 GiB | 957ms | OK |
| 36 | 32 | -- | -- | OOM |
| 40 | 4 | 21.19 GiB | 607ms | OK |
| 40 | 8 | 34.39 GiB | 767ms | OK |
| 40 | 16 | 61.13 GiB | 1330ms | OK |
| 40 | 32 | -- | -- | OOM |
| 44 | 4 | 25.37 GiB | 691ms | OK |
| 44 | 8 | 41.19 GiB | 912ms | OK |
| 44 | 16 | -- | -- | OOM |
| 48 | 4 | 30.02 GiB | 799ms | OK |
| 48 | 8 | 48.71 GiB | 1060ms | OK |
| 48 | 16 | -- | -- | OOM |
