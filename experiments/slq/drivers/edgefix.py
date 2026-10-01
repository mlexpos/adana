"""Fairness pass: for every E2/E3 baseline sweep (SGD, Adam, LaProp) whose best lr is at a grid edge, extend the grid
by 3 points (factor sqrt(10) steps for Adam/LaProp, 2x for SGD) past that edge and re-run. Output <tag>_edgefix.npz with
the extended-grid curves (same eval points, seeds, keys)."""
import os, sys, glob, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer

KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
root = os.path.expanduser("~/dana-exp/runs")
logf = open(os.path.join(root, "edgefix.log"), "a")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()


def make_task(name):
    if name.startswith("nrf_relu_a1.0"):
        return tasks.nonlinear_rf(v=2048, d=1024, alpha=1.0, beta=0.7, act="relu", n_eval=16384)
    if name.startswith("nrf_relu_a0.4"):
        return tasks.nonlinear_rf(v=4096, d=1024, alpha=0.4, beta=0.5, act="relu", n_eval=16384)
    if name.startswith("2layer_a1.0"):
        return tasks.two_layer(v=512, m=256, alpha=1.0, teacher_width=512, n_eval=16384)
    if name.startswith("2layer_a0.4"):
        return tasks.two_layer(v=512, m=256, alpha=0.4, teacher_width=512, n_eval=16384)
    if name.startswith("modarith"):
        return tasks.modarith(p=97, T=32, zipf=1.2, eta_sig=0.7)
    raise ValueError(name)


jobs = []
for path in sorted(glob.glob(os.path.join(root, "e2", "*.npz")) + glob.glob(os.path.join(root, "e3", "*.npz"))):
    if path.endswith("_edgefix.npz") or "_shadow" in path or "_lpd" in path:
        continue
    z = np.load(path, allow_pickle=True)
    for key, gkey, opt in (("sgd_risk", "sgd_lr" if "sgd_lr" in z.files else "sgd_grid", "sgd"),
                           ("adam_risk", "adam_grid", "adam"), ("lap_risk", "lap_grid", "laprop")):
        if key not in z.files:
            continue
        grid = np.asarray(z[gkey], dtype=float)
        fin = z[key][-1].mean(0); fin = np.where(np.isfinite(fin), fin, np.inf)
        i = int(np.argmin(fin))
        if i in (0, len(grid) - 1):
            step = grid[1] / grid[0]
            ext = grid[0] / step ** np.arange(1, 4) if i == 0 else grid[-1] * step ** np.arange(1, 4)
            jobs.append((path, key, opt, sorted(ext.tolist()), i))

log(f"{len(jobs)} edge cases: " + "; ".join(f"{os.path.basename(p)}:{k}:{'low' if i == 0 else 'high'}" for p, k, _, _, i in jobs))
for path, key, opt, ext, i in jobs:
    out = path[:-4] + f"_{key.replace('_risk', '')}_edgefix.npz"
    if os.path.exists(out):
        continue
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    task = make_task(meta["task"])
    ev = z["n"]
    rows = [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in ext]
    t0 = time.time()
    r = trainer.run(task, opt, "adapt", rows, KEYS, S=meta["seeds"], B=meta["B"], n_max=int(ev[-1]), eval_pts=ev, log=None)
    fin_old = np.nanmin(np.where(np.isfinite(z[key][-1].mean(0)), z[key][-1].mean(0), np.inf))
    fin_new = r["risk"][-1].mean(0)
    np.savez_compressed(out, n=ev, risk=r["risk"], grid=np.asarray(ext), meta=json.dumps(meta))
    log(f"{os.path.basename(path)} {opt}: old best {fin_old:.5g}; extended grid {np.round(ext, 7)} -> {np.round(fin_new, 6)} ({time.time()-t0:.0f}s)")
log("EDGEFIX_DONE")
