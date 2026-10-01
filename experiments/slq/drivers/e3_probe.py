"""E3 task-design probe: long Adam/SGD runs on task variants; print excess-CE curves."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
from adana.sim_ls import eval_points
KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
variants = {
    "v1_p97_z1.2": dict(p=97, T=32, zipf=1.2, eta_sig=0.7),
    "v2_p113_z1.05_T64": dict(p=113, T=64, zipf=1.05, eta_sig=1.0),
    "v3_p97_z0.9": dict(p=97, T=32, zipf=0.9, eta_sig=0.7),
}
which = sys.argv[1:] or list(variants)
nmax = 40000; ev = eval_points(nmax, 20)
for name in which:
    task = tasks.modarith(**variants[name])
    print(f"== {name}: Bayes {task['bayes']:.4f}", flush=True)
    for opt, rows in [("adam", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in (3e-4, 1e-3, 3e-3)]),
                      ("sgd", [dict(lr=l) for l in (0.3, 1.0, 3.0)])]:
        t0 = time.time()
        out = trainer.run(task, opt, "adapt", rows, KEYS, S=1, B=32, n_max=nmax, eval_pts=ev, log=None)
        r = out["risk"][:, 0, :]
        best = np.nanargmin(np.where(np.isfinite(r[-1]), r[-1], np.inf))
        print(f"  {opt} best lr={rows[best]['lr']}: n,excess = " + " ".join(f"{n}:{x:.3f}" for n, x in zip(ev[1::2], r[1::2, best])) + f"  ({time.time()-t0:.0f}s)", flush=True)
