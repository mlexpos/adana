import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax
from adana import tasks, trainer
KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
task = tasks.modarith()
print("Bayes loss", task["bayes"], "uniform", np.log(97))
ev = [0, 200, 1000, 3000, 6000]
for opt, rows in [("adam", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in (3e-4, 1e-3, 3e-3)]),
                  ("sgd", [dict(lr=l) for l in (0.1, 0.3, 1.0)]),
                  ("dana", [dict(lr=l, s=0.5, delta=4.0, cap=1.0) for l in (0.1, 0.3, 1.0)])]:
    t0 = time.time()
    out = trainer.run(task, opt, "adapt", rows, KEYS, S=1, B=32, n_max=6000, eval_pts=ev, log=None)
    print(opt, "excess CE over time (rows=configs):")
    print(np.round(out["risk"][:, 0, :].T, 3), f"{time.time()-t0:.0f}s")
