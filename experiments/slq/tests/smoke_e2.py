import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax
from adana import tasks, trainer
KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
for name, task in [("nrf", tasks.nonlinear_rf(v=512, d=256, n_eval=8192)), ("2layer", tasks.two_layer(v=128, m=64, teacher_width=128, n_eval=8192))]:
    ev = [0, 100, 1000, 5000]
    t0 = time.time()
    sgd = trainer.run(task, "sgd", "adapt", [dict(lr=l) for l in (0.25, 0.5, 1.0, 2.0)], KEYS, S=2, B=8, n_max=5000, eval_pts=ev, log=None)
    t1 = time.time()
    ad = trainer.run(task, "dana", "adapt", [dict(lr=l, s=s, delta=4.0, cap=1.0) for l in (0.25, 0.5) for s in (0.25, 1.0)], KEYS, S=2, B=8, n_max=5000, eval_pts=ev, log=None)
    t2 = time.time()
    print(name, "SGD final risk (lr .25,.5,1,2):", np.round(sgd["risk"][-1].mean(0), 5), f"{t1-t0:.0f}s")
    print(name, "adaptive-DANA final risk:", np.round(ad["risk"][-1].mean(0), 5), " g3 last (per leaf):", np.round(ad["g3"][-1][0], 5), " NhB:", np.round(ad["nhb"][-1][0], 3), f"{t2-t1:.0f}s")
