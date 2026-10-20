# The hybrid: d-muon's step with ADana's long momentum

`--opt hybrid-v1` trains with the hybrid optimizer (version 1, global allocation):

- **Plain step.** d-muon's update: `msign` (Newton–Schulz, Nesterov buffer β = 0.95, Moonlight scale) on the hidden matrices,
  and d-muon's AdamW (β = 0.8, 0.999) on the embedding, the readout and the norm gains. Both run at d-muon's learning rate `L`.
- **Long momentum.** ADana's log-time momentum on every tensor, with gain κ_t = s(t) γ0 r_t / Δ_t, where Δ_t = δ/(δ+t),
  s(t) is the learning-rate schedule and γ0 = L/64.
- **The ratio r_t.** One global ratio r_t = min{1, s_b B_eff μ_t / N(Δ_t)}, with:
  - N, the noise kernel, read by stochastic Lanczos quadrature of the Adam-preconditioned Gauss–Newton matrix;
  - μ_t, the gradient-noise fraction from the two halves of the batch;
  - s_b = 1/4.

  The measurements refresh at steps 4, 8, 16, … and then at a fixed interval, and take a few percent of the run.
- **Weight decay.** ADana's log-time weight decay (ω = 4 over T/10).

The code is [`src/optim/hybrid_slq.py`](src/optim/hybrid_slq.py) (with [`src/optim/adana_slq.py`](src/optim/adana_slq.py) and
[`src/optim/slq_torch.py`](src/optim/slq_torch.py)). `hybrid-v1` is a preset in [`scripts/launch.sh`](scripts/launch.sh) that maps to
`--opt hybrid-slq` with these defaults.

## Running it

```bash
# 12 heads, batch 512 = 32 sequences x 16 accumulation steps, one GPU
bash scripts/launch.sh --arch enoki --opt hybrid-v1 --heads 12 --batch_size 32 --acc_steps 16

# 20 heads on 4 GPUs (DDP)
bash scripts/launch.sh --arch enoki --opt hybrid-v1 --heads 20 --batch_size 32 --acc_steps 16 --nproc 4
```

| flag | default | meaning |
|---|---|---|
| `--lr` | AdamW's batch-512 formula rate | d-muon's rate `L`, used by the `msign` step and the AdamW step |
| `--hyb_g0_div` | 64 | γ0 = L / this |
| `--slq_s` | 0.25 | budget fraction s_b |
| `--slq_alloc` | `global` | one ratio for all tensors (`waterfill`: one ratio per tensor) |

Any `--hyb_*` or `--slq_*` flag given on the command line overrides the preset.

**Learning rate.** The default `L` is d-muon's tuned rate from 12 heads up. At 6 and 8 heads, d-muon's optimum is twice the
formula rate, so pass `--lr` at 2× there.

**Batch layout.** Keep `--batch_size × --acc_steps = 512`, and keep `--slq_no_cache` (it is in the preset) whenever more
than one GPU is used. Micro-batches that fit on 80 GB H100s:
- 32 sequences up to 20 heads;
- 16 at 24 heads (`--batch_size 16 --acc_steps 32`);
- 8 at 28 heads (`--batch_size 8 --acc_steps 64`).

**Distributed backend.** DDP only (`--distributed_backend nccl`, the default). The Lanczos measurements use forward-mode AD
through the whole model and are not supported under FSDP.

## Results (Enoki, FineWeb, Chinchilla horizon, batch 512, seed 0)

| heads | non-emb. params | hybrid | d-muon (best lr) | SLQ-ADana (best) | AdamW (best lr) |
|---|---|---|---|---|---|
| 12 | 64M | **3.287** | 3.334 | 3.406 | 3.497 |
| 16 | 151M | **3.060** | 3.108 | 3.166 | 3.215 |
| 20 | 295M | **2.895** | 2.943 | 2.977 | 3.013 |
| 24 | 510M | **2.769** | 2.813 | 2.838 | 2.886 |

Final validation loss.

- **Seed spread.** At 12 heads the hybrid spans 3.286–3.292 over three seeds (water-filling: 3.286–3.290 over six), and d-muon
  spans 3.334–3.338 over six seeds.
- **Compute gain.** Under the broken power-law fit of the ADana paper, d-muon needs about 1.4× the hybrid's compute to reach
  the same loss, from 12 to 24 heads.
- **γ0.** At 12 heads the result is flat in γ0 from L/32 to L/128 (3.285–3.291).
