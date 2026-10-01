# Running notes: buffer-energy adaptive momentum experiments

## 2026-09-26 — setup
- Remote: math-slurm = math-h100-r01, 1x RTX PRO 6000 Blackwell Max-Q (96 GB), JAX 0.9.2, flax 0.12.6, optax 0.2.8. Runs launched directly (nohup), code in ~/dana-exp, outputs ~/dana-exp/runs.
- Key design: PLRF | W is *exactly* Gaussian least squares (cov K = W^T D W, label noise P*). Simulate in the eigenbasis of K with an exact chi^2 sampler for minibatch gradients: Z^T Z w = |w|(chi w_hat + sqrt(chi) P_perp xi), chi ~ chi^2_B. Cost O(G d) per step, independent of B. Split-batch noise estimator = two independent halves; Fisher trace = model-sampled residuals (for LS, tr F = tr K).
- Delta_n = delta/(delta+n) (<= 1, avoids 1-Delta<0 for n<delta); EMA rate rho_n = min(1, max(Delta_n, 10/(n+1))).
- Adaptive rule: g3 = g2 * min{ s / (Nhat/B), 1 },  Nhat/B = 2 g2 <|y|^2> <trF>/<trC>.

## 2026-09-26 — validation (tests/test_sim.py)
- chi^2 trick vs explicit Gaussian minibatch: mean exact, covariance within 0.6% (400k draws).
- SGD stability threshold (exact v3 formula): alive at 0.95 g2max, diverged at 1.05 g2max (B=2, 8). Exact.
- buffer estimator at gamma3=0 (d=400, B=8, Delta_n=4/(4+n)): Nhat/B vs N_eff/B = 20.1 vs 21.6 at n=2e5 (-7%, lag of the EMA under decaying Delta).
- reduced eigenbasis sim vs explicit PLRF (v=96,d=48,B=4, 400 seeds): mean risk ratio 0.99-1.01 for SGD and DANA at n=10..3000.
- E1 launched 23:46 (drivers/run_e1.sh): 4 instances x 4 batch regimes, 576 configs each, 4 seeds, 1e6-2e6 steps.
- E2 trainer (adana/trainer.py, optim_tree.py): per-leaf buffer-energy rule, split-batch trC, model-sampled trF. Smoke: 2-layer adaptive 0.014 vs SGD best 0.021 at 5k steps.
- E3 task (tasks.modarith): p=97, T=32, hidden increment a ~ Zipf(1.2) on permuted Z_p, eta ~ disc. Gaussian(0.7). Bayes CE 1.172 (uniform 4.57). Smoke (B=32, 6k steps): Adam 0.051 excess, SGD 0.056, adaptive-DANA(lr .1) 0.059; adaptive-DANA diverged late at lr 0.3, 1.0 (Nesterov cap gamma3=gamma2 requires g2*lam1 < ~4/3; must set g2 as a fraction of the SGD edge, as in PLRF protocol).
- E2 launched concurrently (drivers/run_e2.sh). 2-layer shrunk to v=512, m=256 for speed.

## 2026-09-27 — E3 task design probe (drivers/e3_probe.py, B=32, 40k steps)
| variant | Bayes CE | Adam best (lr 1e-3) excess @2.5k/7.5k/23k | SGD best (lr 1.0) |
|---|---|---|---|
| v1 p97 T32 zipf1.2 eta0.7 | 1.172 | 0.078/0.049/0.033 | 0.088/0.053/0.035 |
| v2 p113 T64 zipf1.05 eta1.0 | 1.474 | 0.063/0.037/0.027 | 0.064/0.040/0.027 |
| v3 p97 T32 zipf0.9 eta0.7 | 1.193 | 0.084/0.049/0.037 | 0.095/0.054/0.042 |
All show a slow ~n^-0.2..-0.3 tail after a fast drop (rare Zipf increments). SGD ~ Adam here. Chose v1 (cheapest).
- E2 nrf a=1 (floors pending): B=4 SGD 1.73e-4, adapt 1.61e-4, oracle 1.60e-4 ; B=64 SGD 1.569e-4, adapt 1.579e-4, oracle 1.532e-4 — all near an irreducible floor; need floor to judge.
- Floors (drivers/floor_nrf.py, LS optimum from 2M samples on the fixed eval set): nrf a=1: 1.4638e-4; nrf a=0.4: 0.10937.
- E2 nrf a=1 EXCESS risk at 1e5 steps (best-tuned per method): B=4 SGD 2.64e-5, adapt 1.48e-5, oracle 1.35e-5 | B=64 SGD 1.05e-5, adapt 1.15e-5, oracle 0.68e-5 | B=1024 SGD 0.62e-5, adapt 1.02e-5, oracle 0.58e-5.
  -> adaptive wins at small B, loses at medium/large B although oracle DANA still beats SGD there. Hypotheses: (i) adaptive g2 <= edge/2 while tuned SGD sits at the edge; (ii) cap g3 <= g2 too tight when noise is small. To check in analysis (g3 trajectories vs oracle's best schedule).

## 2026-09-27 00:20 — diagnosis of adaptive-DANA on nrf (a=1)
- Adaptive best at the SMALLEST s in grid (0.0625) for every B; final excess increases monotonically with s.
- At n=1e5, B=64: adaptive g3/g2 = 0.028 (s=.25, fac .5) vs best oracle schedule 3e-4 (c=0.1, kappa=0.5, fac 1/8). B=1024: adaptive hits the cap g3=g2 around n~2e4 while oracle is at 7e-4.
- Interpretation: the buffer rule sizes g3 at the STABILITY limit (momentum may add a fraction m of the current risk P).
  Here P is dominated by the irreducible floor (1.46e-4) while the excess is ~1e-5, so a stability-limited momentum
  adds variance m*P >> excess. Stability-limited == loss-optimal only in the bias/signal-dominated regime
  (excess >> floor), as in noiseless PLRF far from convergence. Near a noise floor, the loss-optimal g3 is smaller by ~excess/P.
- Action: extended s grid to {0.004, 0.016, 0.0625, 0.25, 1} for remaining E2 tasks; later rerun nrf a=1 with the low grid.
  Candidate fix to test later (clearly an extension): scale the budget by a measured signal fraction, e.g. gradient SNR
  <g1,g2> / (trC/B) from the split halves (unbiased estimate of |grad|^2), which -> O(g2 lam) at the noise floor.

## 2026-09-27 00:30 — E3 v1 B=16 (30k steps), excess CE over Bayes
- SGD best (lr 1/16; lr_edge 1.0): 0.0456 | adaptive global best (fac .25, s .125): 0.0468 | per-leaf best (fac .25, s .125): 0.0390 | oracle DANA best (c .63, kappa .5, fac .25): 0.0271.
- Adaptive diverges for s >= 0.25 (excess 1-3 nats), per-leaf worse. NhB estimate erratic (0.8 -> 206 between n=13.6k and 30k); g3 jumps to the cap after transients.
- Oracle (smooth c/sqrt(n)) is 40% better than SGD: momentum helps, but the raw buffer rule is too noisy/unsafe on the transformer.
- Candidate robustifications: (a) monotone g3 (running min of the rule), (b) longer EMA, (c) variance-aware budget (see E2 diagnosis).
- OOM incident 00:23: E1 had default JAX preallocation (72 GB); E3 B=128 died. Restarted E1 (PREALLOCATE=false; lost 37 min) and E3 at 00:26.

## 2026-09-27 01:50 — E3 v1 SGD phase complete (excess CE over Bayes 1.172)
| B (steps) | SGD tuned | adapt global best | adapt per-leaf best | oracle DANA best |
|---|---|---|---|---|
| 16 (30k) | 0.0456 | 0.0468 | 0.0390 | 0.0271 |
| 128 (15k) | 0.0287 | 0.0310 | 0.0263 | 0.0164 |
| 512 (4k) | 0.0378 | 0.0229 | 0.0220 | 0.0183 |
Momentum (decaying) beats SGD at every B; adaptive captures part of it, most at large B.
## E2 nrf a=0.4 (floor 0.10937), excess: B=4 SGD 9e-4, adapt 6.1e-3, oracle 5.9e-3 | B=64 SGD 7e-5, adapt 5.2e-4, oracle 3.3e-4 | B=1024 SGD 1e-5, adapt 3.1e-4, oracle 3e-5.
Protocol flaw: DANA arms restricted to g2 <= edge/2 while tuned SGD uses its best lr (near the edge). Below the high-d line SGD wants max lr. Follow-up must give DANA arms g2 = SGD-best lr (then the rule's budget m=1/2-k_SGD should shrink g3 -> 0 and recover SGD).

## 2026-09-27 02:45 — speed fix + first E1 result
- XLA_FLAGS command buffers (CUDA graphs incl. WHILE) -> PLRF sim 2.41 -> 0.17 ms/step (14x). All run scripts now set it. E1/E2 restarted 02:29.
- E1 a=1 b=0.7 d=4096 B=2, 2e6 steps (seed-mean excess): SGD best (0.5 g2max) 5.03e-7 | adaptive s=.25 fac=.25 6.51e-8 (7.7x lower; 13.6x fewer steps to SGD's final loss) | theory rule (true N_eff, s=.25) 5.64e-8 | oracle best (c=10, kappa=.7, fac .25) 3.3e-8 (26x fewer steps).
  s has an interior optimum at 0.25 (theory: s = 2m, m~1/8..1/4). Oracle flat over kappa in [0.3, 0.7] once c tuned.
  Estimator: Nhat/B over-estimates true N_eff/B by 2-4x mid-run (bias signal in the buffer => conservative), converges at the end (216 vs 248).
- E3 adam/laprop phase launched (run_e3b.sh).

## 2026-09-27 03:05 — E1 alpha=1 all B except 32768; key failure mode found and fixed
- E1 a=1 (excess): B=64: SGD 2.5e-7, adapt 1.7e-8, theory 3.3e-8, oracle 1.8e-9 | B=1024: SGD 2.4e-7, adapt 1.9e-8, theory 2.8e-8, oracle 1.9e-10.
  At medium/large B adaptive (buffer) beats SGD by 10x but trails oracle by 10-100x.
- Cause: at large B the buffer energy is dominated by the MEAN gradient (signal accumulated during the transient), not noise.
  Nhat/B overestimates N_eff/B massively (B=1024, n=3.7k: 5.47 vs true 0.033, 180x) -> g3 far too small -> never reaches the
  Nesterov regime that large B permits. Splitting the batch cannot remove it (the identity needs the common response term).
- Fix = estimator (B) of dana_adaptive.tex: SHADOW buffer driven by synthetic model-sampled noise (cov K/B) with the curvature
  applied to the shadow state (exact K u, or minibatch (Z^T Z/B) u = one Hessian/GN-vector product per step). Signal-free by construction.
  Test (d=1024, B=1024, n=3.7k): shadow_batch Nhat/B 0.028 vs true 0.033; excess 7.2e-7 = theory rule, vs 7.9e-6 buffer rule (11x).
  B=8: shadow_batch 3.50 vs true 3.73; loss = theory.
- E1s (shadow_batch, shadow_exact, adapt_mono arms; same keys) chained after E1.
- E2 2layer a=1 B=1024: SGD 1.06e-2 | adaptive 3.5e-4 (30x) | oracle 3.7e-4.
- E3 B=16 adam phase: Adam 0.0361, LaProp 0.0362, adaptive LaProp-DANA 0.0468, oracle LaProp-DANA 0.248 (unstable grid).
- NN shadow estimator implemented (tasks.*.gnvp = J^T Lambda J u via jvp/vjp; optim 'dana_shadow'). Smoke (3k steps, lr fixed, s=.25):
  2layer B=256: SGD 0.0242 | buffer 0.00335 | shadow 0.00249 ; modarith B=256: SGD 0.0640 | buffer 0.0441 | shadow 0.0365 ; small B: shadow ~ buffer.
- E1 a=1 B=32768: SGD 2.4e-7 | adapt 1.63e-9 == theory 1.63e-9 (both at cap g3=g2 throughout) | oracle 1.4e-11. Oracle's extra 100x must come from g2 fac or late momentum decay -> check.
- E3 B=128 adam: Adam 0.0218 (SGD 0.0287, oracle DANA 0.0164).  E2 2layer a=0.4 B=4 risk: SGD 0.177, adapt 0.160, oracle 0.144.
- Chained: run_shadow_nn.sh (E2+E3 shadow arms) and run_e2b.sh (E2 adam phase) after run_e2.sh; run_e1s.sh after run_e1.sh.

## 2026-09-27 03:20 — large-B law: optimal g3 is batch-independent beyond small B
Oracle-best schedule (c=31.6, kappa=0.7, fac .25) is IDENTICAL for B = 64, 1024, 32768; product (g3/g2)*N_eff(n): 5 -> 1 over n=1e2..1e6.
At B=2 (noise-limited) product 1.35 -> 0.23 (rule sB = 0.5).  => empirical law: g3/g2 ~ min{sB, C}/N_eff(n), C ~ 1-2.
The stability-limited rule (cap g3=g2, i.e. Nesterov) is NOT optimal deterministically: at B=32768 constant g3=0.01 g2 beats the cap by 45x;
the optimum decays DANA-like even without noise (theory bounds max stable g3, not optimal g3).
Modified rule: s_eff = min(s, C/B). E1c (shadow/buffer/mono x C in {0.5,1,2,4}) chained after E1 (runs in parallel with E1s).

## 2026-09-27 04:00 — LaProp-DANA protocol bug and fix; shadow results
- E2 nrf a=1 shadow (excess): B=4 1.36e-5 (= oracle 1.35e-5; buffer 1.48e-5; SGD 2.64e-5) | B=64 0.80e-5 (buffer 1.15e-5, SGD 1.05e-5, oracle 0.68e-5).
- E2 adam phase nrf a=1: Adam/LaProp excess 3.6e-5 (B=4, worse than SGD 2.6e-5), 0.99e-5 (B=64). LaProp-DANA (old grid) 6e-2..0.4: broken.
- Diagnosis: u = g/sqrt(v) has O(1) noise even at the optimum; LaProp's step lr*EMA_b1(u) averages it (variance x (1-b1)/(1+b1) ~ 0.05),
  so its lr_edge is large. LaProp-DANA used g2 = fac*lr_edge on the UN-averaged u (RMSprop-like) and the DANA buffer (memory n/delta)
  accumulates O(1) normalized noise -> blow-up. With momentum off it is just RMSprop and behaves.
  Fixed protocol: g2 in {1, .3, .1} x (best LaProp lr). Debug (nrf/2layer, B=8, 1e4 steps): LaProp 0.00116/0.00584 vs LaProp-DANA
  0.00112/0.00737 at best g2; the rule keeps g3 negligible (results independent of s) because preconditioned noise >> signal.
- Relaunched as drivers/run_adam_all.sh: Adam/LaProp baselines (phase adam) then corrected LaProp-DANA arms (phase lpd) for E2+E3.
- E1 a=0.8 B=2 (excess): SGD 1.93e-6 | buffer best 9.1e-7 | theory (s>=.25) 1.83e-6 | oracle 7.1e-7.
- E2 2layer a=0.4 B=64 (risk): SGD 0.098 | adaptive 0.067 | oracle 0.053.   E3 B=128: adaptive LaProp-DANA(old grid) 0.0242 vs Adam 0.0218.
- 04:45 WARNING (fairness): nrf a=1 B=4 LaProp best lr = 1e-4 = bottom of grid logspace(-4,-0.5). Baselines at grid edges must be re-tuned
  on an extended grid before concluding. TODO after current chains: edge-check pass over all E2/E3 baseline sweeps (SGD, Adam, LaProp).
- PLRF a=0.8 B=32768: SGD 1.6e-7 | buffer 4.4e-8 | theory 1.6e-7 | oracle 1.0e-10 (1600x).  a=0.4 B=2: SGD 1.4e-3 | buffer 8.7e-3 | oracle 7.4e-3 | theory 1.3e-2
  (DANA arms at g2 <= g2max/2; SGD best at higher g2 -> E1d round with g2 in {.7,.85} g2max chained).

## 2026-09-27 06:40 — E1 main sweep complete; below-the-line regime is VARIANCE-dominated
- SGD final loss decreases monotonically as g2 decreases for a=0.4 and a=0.3 at EVERY B (incl. 32768); best at grid edge fac=1/16.
  Cause: large irreducible P* for 2a<1; stationary SGD excess ~ g2 trK P*/(2B) dominates at n=1e6 (bias resolved). Any momentum adds
  variance; the right tool here is step-size decay, not momentum (consistent with no outscaling below the high-d line).
- My earlier "restricted g2 is unfair" reading was backwards; E1d 'hig2' round (g2 .7/.85 g2max) confirmed it (all ~5e-2 at a=0.4 B=2) -> killed.
  Replaced by E1d 'lowg2': SGD extended to fac 1/32..1/256; shadow/buffer (s .004-.0625) and oracle at fac 1/16..1/64.
- E1 main table (seed-mean excess, s=.25, fac .25 unless noted):
  a=1: B=2 SGD 5.0e-7 | buffer best 6.5e-8 | oracle 3.3e-8 ; B=64 2.7e-7 | 1.9e-8 | 3.2e-9 ; B=1024 2.4e-7 | 2.0e-8 | 2.7e-10 ; B=32768 2.4e-7 | 1.7e-9 | 2.0e-11
  a=0.8: B=2 2.0e-6 | 1.0e-6 | 8.3e-7 ; B=64 4.4e-7 | 2.5e-7 | 6.2e-8 ; B=1024 2.3e-7 | 1.1e-7 | 4.8e-9 ; B=32768 1.6e-7 | 4.7e-8 | 1.5e-10
  a=0.4/0.3: SGD (fac 1/16) best everywhere; DANA arms 3-300x worse at fac>=1/4.

## 2026-09-27 07:30 — follow-up rounds (log metric = best config, min over seeds)
PLRF a=1 (excess):              B=2        B=64       B=1024
  SGD (tuned)                 4.4e-7     2.5e-7     2.4e-7
  oracle DANA (grid)          3.0e-8     1.8e-9     1.9e-10
  buffer rule                 5.8e-8     1.7e-8     1.9e-8
  buffer rule, monotone       3.8e-8     3.5e-9     4.4e-10
  shadow rule                 3.3e-8     1.5e-8     1.5e-8
  shadow + saturating (C=1)   4.9e-8*    2.3e-9     2.5e-10   (*C/B >= s at B=2: identical to shadow at s=.25 fac .25)
  shadow + sat (C=.5/2/4)       -        1.9/4.3/7.5e-9   5.2/2.9/5.0e-10
  buffer + saturating         6.3e-8     1.9-3.3e-7 3.4e-7   (buffer overestimates N_eff; saturation starves momentum -> ~SGD)
=> shadow estimator + saturating batch (C~1) matches the oracle at every B tested so far; monotone buffer is a strong cheap alternative.
- a=0.4 lowg2: B=2 SGD(ext) 3.1e-4 | oracle 3.6e-4 | buffer 4.0e-4 | shadow 4.0e-4 ; B=32 SGD 6.9e-5 | DANA arms 2.7-3.2e-4 (limited to fac>=1/64).
- E3 lpd (corrected): B=16 LaProp 0.0362 | adaptive LaProp-DANA 0.0477 | oracle LaProp-DANA 0.0258 ; B=128 LaProp 0.0217 | adapt 0.0339? (B=512) | oracle 0.0155.

### 2026-09-27 ~08:00 — E1d lowg2, α=0.4 B=32768
- SGD (extended small-step grid) 8.1e-8 | oracle DANA 3.4e-7 | buffer adapt 4.7e-5 | shadow 4.6e-5.
- Non-saturating adaptive rules sit at the Nesterov cap at huge B (noise budget never binds) — same failure mode the saturating law fixes above the line.
- Below the line even the best momentum schedule is ~4x worse than small-step SGD: momentum does not help in the variance-dominated regime.

## 2026-09-27 08:20 — report drafted; E1s/E1c trimmed
- report/body.tex + summary.tex drafted from current data (tables auto-generated: e1_table.tex, e1_oracle.tex, nn_table.tex); PDF compiles (10 pp).
- Oracle-best kappa above the line = 1/(2a)+0.2 with c at the top of the grid (31.6): finite-horizon preference for "cap early, decay faster". Below the line oracle-best c = grid bottom (no momentum).
- Remote: stop_after_a08.sh stops the E1s/E1c chains once a=0.8 is done (below-line shadow arms already covered by E1d) to free the GPU for E3 shadow/satsh + edgefix.
- 08:45 E2 2layer a=0.4 shadow: B=4 0.144 | B=64 0.0529 | B=1024 0.0102 (oracle 0.144/0.053/0.0122; Adam 0.150/0.052/0.0094; SGD 0.177/0.098/0.060). Shadow closes the buffer-rule gap.
- nrf a=0.4 B=8192 shadow excess ~8e-4 vs SGD 5e-6 (below-line, as expected).
- 08:52 E3 B=512 lpd: oracle LaProp-DANA 0.0154 (LaProp 0.0245, -37%); adaptive LPD 0.0339 (worse than LaProp). E3 shadow phases started 08:45.

## 2026-09-27 09:10 — LaProp-DANA fix (user: "normalized methods need LR scheduling; stay ~2x below stability; fix buffer rule + dana + laprop")
Diagnosis of old adaptive LPD (g3 ~ 0 everywhere, NhB 1e3..1e6):
- wrong coordinates: traces/buffer taken in u = g/P space (weights P^-2). The buffer identity holds where the step is plain SGD:
  z = P^{1/2} theta, gradient r*g with r = P^{-1/2}, curvature H_z = r H r, noise r C r (weights P^-1).
- oracle LPD always picked g2 = 0.3 x LaProp best lr, c 0.06-0.4, kappa .25-.5.
- normalized curvature H/sqrt(v) grows as noise shrinks -> fixed step drifts to instability; needs a step schedule.
New optimizer 'lpd2' (optim_tree.py): g2_n = min(lr_LaProp, f * 2/lam_hat), lam_hat = top eigenvalue of the minibatch GN in z
(one power-iteration step per step; the minibatch lam includes the tr/B noise term), f = 1/2; shadow estimator in z;
g3 = g2_n min{min(s, C/B) B / Nhat, cap}; y_z = (1-D) y_z + r*g; theta -= g2_n g/P + g3 r*y_z.  Cost: 2 GNVPs/step.
Quick test 2layer a=1 B=64, 5e4 steps, 1 seed:
  f=.5, lr ceiling = LaProp best, s=.125, no saturation (C=inf): 7.56e-4  | LaProp tuned 1.65e-3 | oracle LPD 7.95e-4 | old adaptive LPD 2.1e-3
  same, s=.03: 1.30e-3 ; C=2: 1.27e-3 (saturation too tight for normalized coords: product (g3/g2) N_eff ~ 8 = sB here, not ~1)
  s=0 (step schedule only): 2.3e-3 ; step decays to 0.18-0.35x LaProp lr by the end (automatic annealing).
  NO lr ceiling: step settles at 3.5x LaProp lr, loss 0.033 (self-consistent trap: big steps -> high loss -> big v -> flat z-curvature).
  => the stability-tracked step needs the tuned base lr as a ceiling (one tuned scalar shared with LaProp).
Full panel launched: drivers/run_lpd2.sh (2layer a=1, modarith B16/128/512, nrf a=1, 2layer a=0.4).

## 2026-09-27 09:25 — second compute site: rorqual (Alliance, slurm, H100 80GB, account rrg-epaq_gpu)
- env: ~/adanaenv (module StdEnv/2023 python/3.12 cuda/12.6; pip --no-index jax[cuda12]==0.10.2). Code+runs in ~/scratch/dana-exp (symlink ~/dana-exp).
- sbatch wrapper ~/dana-exp/slurm/job.sh "<python args>" [matmul precision]; logs slurm/<name>_<id>.out. Fetch: scripts/fetch_rq.sh.
- Moved off math-slurm (waiters killed there): satsh phases (E2 nrf/2layer a1/2layer a.4, E3 B16/128/512), edgefix, lpd2 (modarith x3, nrf, 2layer a.4),
  + new lpd2 'his' grid (s .25-1, lr ceiling .3x) on 2layer a1 and modarith. math-slurm keeps: E1 tail (a.8 B32768, a.3 B32768), E3 shadow, adam_all lpd (nrf a.4), lpd2 2layer a1.
- lpd2 2layer a1 full (3 seeds): B=4 6.27e-3 (LaProp 7.42e-3, oracle LPD 5.17e-3; stability step never binds at B=4, s=.125 at grid edge) ; B=64 7.34e-4 (LaProp 1.65e-3, oracle LPD 7.95e-4).
  Uncapped (no lr ceiling) diverges at B=4 (early power-iteration underestimates lam_max).
- 09:26 lpd2 'his' grid, 2layer a1 (3 seeds). Frozen candidate (lr ceiling = LaProp best, f=.5, s=.25, C=inf):
  B=4 6.13e-3 | B=64 6.34e-4 | B=1024 1.55e-4   vs LaProp 7.42e-3/1.65e-3/3.94e-4, Adam 7.29e-3/1.66e-3/4.32e-4, oracle LPD 5.17e-3/7.95e-4/6.81e-4.
  Best in grid at B=4: ceiling .3x, s=.5: 4.51e-3 (< oracle LPD). s=1 bad everywhere; s in [.125,.5] good.
  B=1024: stability-tracked step anneals to 0.004x LaProp lr by the end; momentum g3/g2 ~ .045 carries progress (automatic LR schedule).

## 2026-09-27 09:35 — E1 complete; lpd2 on the transformer
- E1 a=0.8 final (seed-mean excess): sat. shadow C=1: B=64 1.07e-7 | B=1024 7.2e-9 | B=32768 1.9e-10 vs oracle 6.2e-8/4.8e-9/1.45e-10, SGD 4.4e-7/2.3e-7/1.6e-7.
  B=2: shadow 1.88e-6 ~ SGD 2.03e-6 (oracle 8.3e-7, buffer best 1.03e-6): at a=0.8 small B the frozen s=1/4 captures little.
  E1s/E1c chains stopped after a=0.8 (as planned). E1d a=0.3 complete: SGD beats oracle by 1.7-4x at every B.
- lpd2 (unguarded) transformer, best in grids: B=16 0.0275 | B=128 0.0152 | B=512 0.0177  vs LaProp .0362/.0217/.0245, oracle LPD .0258/.0155/.0154.
  Failure modes: (1) the z-shadow blows up after ~5k steps (nhb 30 -> 1e13 -> inf -> NaN into g3 and params) — rare-token curvature spikes
  make the linearized noise dynamics unstable; (2) at B=512 the stability-tracked step collapses (0.002x LaProp lr at s=.25; s=0 stalls at 0.94):
  z-curvature H/sqrt(v) grows as gradients shrink; real LaProp is self-stabilized by normalization, so linear z-stability is too conservative late at large B.
  2-layer: the same schedule helps a lot (B=1024 step 0.004x, loss 2.5x below LaProp).
- v2.1 guard (optim lpd2): shadow reset + hold on blow-up (non-finite or >100x EMA), g3 NaN-guard, optional step backoff bk on each reset. Grid 'g' submitted on rorqual.
- 09:35 edgefix (rorqual) DONE: 11 edge cases; every extension past the edge is worse (Adam/LaProp 3e-6..3e-5 all worse than 1e-4;
  SGD low-side worse; 2layer a.4 B1024 SGD high side diverges). => no baseline was grid-limited; baseline numbers stand.
- satsh 2layer a=0.4: B=4 0.144 (= shadow) | B=64 0.078 (shadow 0.053) | B=1024 0.052 (shadow 0.0102): saturation C~1 HURTS in NNs;
  the PLRF-derived C does not transfer (same as in LaProp z-coords, where (g3/g2) N_eff ~ sB ~ 8 at optimum).
- 09:42 lpd2 guarded ('g' grid). Candidate frozen default: lr ceiling = 0.3 x LaProp best lr, f=.5, s=.25 (bk irrelevant; resets 0-2/run):
  2layer a1: B=4 4.32e-3 | B=64 6.22e-4 | B=1024 1.54e-4 (LaProp 7.42e-3/1.65e-3/3.94e-4; oracle LPD 5.17e-3/7.95e-4/6.81e-4) -> best method at every B.
  modarith B=512: 0.0151 (LaProp 0.0245, oracle LPD 0.0154). With ceiling 1x the step collapses to .0018x (0.038); with .3x it anneals only to .15-.19x.
- 09:42 E3 shadow (SGD adaptive-DANA, GN shadow): B=16 0.0232 (SGD .0456, Adam .0361, oracle DANA .0271, oracle LPD .0258 -> best of all) | B=128 0.0186 (SGD .0287, Adam .0217, oracle .0164); mono 0.0270/0.0188. B=512 running (math-slurm).
- 09:59 E3 shadow B=512: 0.0206 (mono 0.0186) vs oracle DANA .0183, Adam .0235, SGD .0378. ALL RUNS COMPLETE (math-slurm + rorqual).

## 2026-09-27 10:10 — report finalized (report/report.pdf, 13 pp)
- Tables regenerated: e1_table.tex, e1_oracle.tex, nn_table_sgd.tex, nn_table_lap.tex. Body: LaProp-DANA v2/v2.1 section, conclusions + recipes + open issues.
- Corrections made while checking against tables: v2.1 default equals oracle LPD only at B=512 (14-15% behind at B=16/128; best v2 beats it);
  PLRF a=1 B=32768 needs C=2 (C=1 is 2.8x off oracle); nRF a=0.4: oracle LPD edges SGD by 1.06-1.5x (SGD beats all adaptive methods).

## 2026-09-27 evening — SLQ report (report/slq_dana.tex, 20 pp; figures plots/report_figs.py -> report/figs_slq)
- SLQ design (P1-P5), certified Gauss/Radau stopping, det-eq de-biasing (scalar outputs), deflation, threshold from quadrature,
  automatic step tau*g2max_hat, water-filling. PLRF full setup + accuracy/adaptive/threshold/in-loop N_eff/loss curves/gamma3 vs
  DANA-decaying (above and below the line). Deep MLP full setup + exact check + scheduled comparison + trajectories + allocation.
  Transformer: scheduled baselines, certified N_eff (bracket), correlated tokens (T_eff 25-30 of 31; last-layer proxy OK,
  embedding proxy fails early), B_eff correction; B_eff runs (be_ma*) still running on rorqual.
- Corrections recorded in the report: below the line auto-SGD at tau=1/16 is 5-17x worse than small-step SGD (best fac 1/128-1/256);
  deflation at m=64 reduces the B=1 threshold spread 13% -> 8%; earlier transformer claims vs UNscheduled baselines do not stand.

## 2026-09-27 — gamma_3-only framing (gamma_2 swept), stochastic vs Lipschitz EoS
- MLP (sched, gamma_3 from SLQ rule, alloc wf-buffer best): SGD-DANA 5.58e-3/1.88e-3/9.48e-4 (B 8/128/1024) vs sched Adam
  6.37e-3/2.20e-3/1.14e-3 (1.14/1.17/1.20x); LaProp-DANA 4.61e-3/1.37e-3/7.63e-4 (1.38/1.61/1.49x). LaProp-DANA optimum
  gamma_2 ~1e-4..3e-4 (~10x below LaProp's), interior (checked with --lo grid).
- EoS+trace, MLP: lambda_max response -0.8..-0.9 (SGD), tr F -0.4..-0.5; tuned SGD at Lipschitz EoS (B128 step 7.1 vs 2/lam 6.4);
  stochastic threshold 2B/trF 1.6x (B8) / 16x (B128) above.
- User hypothesis: Lipschitz EoS is self-maintaining; stochastic ratio chi_S = eta trF/(2B_eff) ~ lr^(1-b) grows; crossing it breaks the
  run; max lr ~ crossover / (2-4). Extrapolated crossover fits B8 (div. between lr 4 and 8; crossover ~18) not B128 (~700 vs div. 8-16).
  Direct test: eos_diag.py --mults (ramp to 16x tuned), logs chi_L, chi_S time courses; jobs eot_ramp_mlp{8,128,1024}, eot_ramp_ma{16,128}.
- STASHED (user, 2026-09-27): adaptive gamma_2 / stochastic-vs-Lipschitz crossover. Conclusion: adapting gamma_2 is fraught because
  the curvature adapts to the step (EoS). Partial ramp data (MLP, lr x0.5): chi_S late 0.18/0.043/0.010 at B 8/128/1024, far below 1
  at large B although SGD diverges at 2x tuned -> crossover picture not supported at large B. eot_ramp_ma* jobs to be cancelled.
- FOCUS: adaptive gamma_3 only, gamma_2 swept. Transformer B512: SGD-DANA 1.88e-2, LaProp-DANA 1.78e-2 (grid edge) vs sched Adam
  1.34e-2, SGD 2.37e-2.
- Next: is the rule's gamma_3 near-optimal? sched.py arms dana_oracle/lpd_oracle (fixed DANA-decaying g3 = g2 min{c(1+n)^-kappa,1},
  (c,kappa) grid incl. kappa=0 constant ratio, at swept-optimal g2 x {1/2,1,2}) and dana_s/lpd_s (rule with s = 1/16..2).
  optim_tree: 'dana'/'laprop_dana' now scale the whole update by sigma(n) when hp T > 0 (old drivers have T=0: unchanged).
- PLRF scheduled (tasks.plrf, original basis, d=4096 v=2d, beta=0.7, 30k steps, sched_mode 1; math-slurm). Final excess risk:
  a=1:   B8   SGD 4.7e-6 Adam 2.0e-6 | rule SGD-DANA 4.0e-7 LPD 3.7e-7 | oracle SGD-DANA 2.5e-7 LPD 2.7e-7
         B128 SGD 2.9e-6 Adam 4.6e-7 | rule 5.3e-8 / 3.8e-8 | oracle 4.1e-8 / 3.3e-8
         B1024 SGD 2.9e-6 Adam 3.4e-7 | rule 1.5e-8 / 1.07e-8 | oracle 1.5e-8 (=Nesterov) / 9.8e-9
  a=0.4: B8   SGD 2.8e-4 Adam 2.8e-4 | rule 2.9e-4 / 3.0e-4 | oracle 2.5e-4 / 2.5e-4
         B128 SGD 2.8e-5 Adam 3.3e-5 | rule 9.1e-5 / 1.0e-4 FAIL | oracle 2.7e-5 / 2.6e-5
         B1024 SGD 4.4e-6 Adam 4.5e-6 | rule 9.2e-5 / 1.3e-4 FAIL | oracle 4.3e-6 / 3.0e-6 ; Nesterov 8.4e-5 / 7.9e-4
  gamma_3 trajectories: SGD-DANA a=1 g3/g2 ~ n^-0.50 (=DANA-decaying); B1024 at cap (Nesterov). LPD a=1 decays n^-1.24/-0.77.
  a=0.4: rule at cap early then n^-0.66..-0.92; oracle optimum kappa=1 (bounded amplification g3/Delta ~ const).
- Transformer oracle (fixed g3 = g2 min{c(1+n)^-kappa,1}): LPD 0.0165/0.0104/0.0118 (B16/128/512) beats sched Adam 1.14/1.20/1.14x;
  winners kappa 0.75-1; Nesterov 1.64/0.19/0.018. SGD-DANA oracle 0.0204/0.0146/0.0156 (beats SGD, not Adam).
  Rule mode 1: SGD-DANA 0.039/0.026/0.019, LPD 0.031/0.116/0.018 (mode 0: 0.069/1.45/0.019, 0.061/0.25/0.018).
- MLP oracle vs rule: rule within ~0-11% (rule wins SGD-DANA B1024, LPD B8). s-sensitivity: s=1/4 optimal in 5/6.
- Theory (sympy, frozen coeffs): per-mode noise transfer T(a,b,D) exact; denominator (a+D-aD)((2-a)(2-D)-b) -> Lipschitz
  condition b < (2-a)(2-D); small coeffs: T ~ a/2 + b/(2(a+D)) -> g3/g2 < (2B - g2 trK)/N_eff (rule s=1/4 is 1/8 of it).
  Frozen-coefficient theory gives no B-independent cap; the data (a=0.4, transformer: kappa~1 winners) needs one -> C runs.
- Saturation C (PLRF): C=1 fixes a=0.4 (SGD-DANA 2.4e-5/3.7e-6 at B128/1024 beats oracle; LPD 2.4e-5 / 4.4e-6 w/ C=4) but costs
  4-17x at a=1 (kills Nesterov-type acceleration). No constant C works.
- Mechanism (exact moment recursion drivers/exact_moments.py, matches measured finals within ~5%): stationary excess floor
  ~ F (E+P*)/(1-F); the rule pins F ~ s/2 -> floor ~ s P*/2 (a=0.4: P*=4.5e-3 -> the observed 1e-4 plateau; a=1: P*=2.3e-7, harmless).
  B=inf: uncapped rule reaches 1e-12 at a=0.4 -> pure noise-floor effect.  Greedy argument: dE/dt = -r(F)(E - F P*), r ~ F
  -> F* = min(F_stab, E/(2P*)) : momentum floor = half the current excess.  In sim: budget x min(1, 4 E/(E+P*)) best everywhere.
- Measurable proxy (hp gns=k): budget x min(1, k B/B_noise), B/B_noise = 4<g1,g2>/|g1-g2|^2 (RAW coords, EMA window n/3).
  Bug fixed: z-coords + n/10 window drove the multiplier to 0 for LPD.  Results (final excess):
  a=0.4 SGD-DANA  B8 2.51e-4 (k16) | B128 1.63e-5 (k16; SGD 2.8e-5) | B1024 1.25e-6 (k1; SGD 4.4e-6)
  a=0.4 LPD       B8 2.58e-4 (k16) | B128 1.11e-5 (k4; Adam 3.3e-5, 3.0x) | B1024 2.81e-6 (k1; Adam 4.5e-6)
  a=1: identical to uncapped rule for k>=4 at all B (k=1 costs 1.5x at B8).  Best k ~ 1/B -> proxy mismatch
  B/B_noise = (B lam_bar/trK) E/(E+P*).  Next: hp gnsb=k' (multiplier min(1, k'/B_noise)), k' in {256,1024,4096}.

## 2026-09-28 — LM: adana-slq on Enoki (branch mlexpos/adana:adana-slq, wandb ep-rmt-ml-opt/momentum-reloaded)
- Implementation: ADana with alpha(t) -> alpha_T = min(a_T S/N(Delta_t), cap)/Delta_t, S = s B_eff mu; SLQ of r GN r (exact jvp/vjp,
  math SDPA); FD GN products were wrong on Enoki (per-tensor err >100%). Gauss estimate converges slowly (spectrum ~6 decades):
  m<=64, Gauss-convergence stop, Aitken correction. N dominated by lm_head (~90%) and wte; waterfill gives blocks A up to 1/Delta.
  mu = 1 throughout (B_noise <= 64 seqs at B=512) -> k' inactive on Enoki. 16h+ need --slq_chunk 1 --slq_no_cache (OOM otherwise).
- Best final val loss, global B=512 x 2048, 20 tok/param, cos_inf (LR multipliers relative to Enoki_512 formula of each opt;
  adana-slq relative to ADana's):
    3h : AdamW 5.133 (x.5) | ADana 5.237 | adana-slq(s=.25) 5.159 (lr 6.7e-4)
    6h : AdamW 4.225 | ADana 4.435 | MK4 4.381 | adana-slq s=.5 4.147 (x1/8), s=1.0 4.128; s>=.5 plateau (s=2 4.139)
    8h : AdamW 3.895 | ADana 3.976 | MK4 3.978 | adana-slq 3.822 (x1/8)
    12h: AdamW 3.497 (x2; x4 3.503) | ADana 3.475 | MK4 3.487 | adana-slq 3.407 (x1/8, edge)
    16h: AdamW 3.215 | ADana 3.212 | MK4 3.203 | adana-slq 3.192 (x1/8; x1/4, x1/2 running)
    20h: AdamW 3.013 (x4, edge) | ADana 2.999 | MK4 <=3.017 (running) | adana-slq 2.987 (x1/8, edge)
  waterfill > global by ~0.1 (6h); cap 2 no effect; MK4 per-element clip hurts. Bracketing runs queued: slq x1/16 at 12/16/20h,
  AdamW x8 at 20h.
