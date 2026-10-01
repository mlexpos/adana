import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
task = tasks.nonlinear_rf(v=512, d=256, n_eval=4096)
ev = [0, 10, 100, 1000, 3000]
for lr in (0.003, 0.01, 0.03, 0.1):
    lap = trainer.run(task, "laprop", "adapt", [dict(lr=lr, b1=0.9, b2=0.99, eps=1e-8)], KEYS, S=1, B=8, n_max=3000, eval_pts=ev, log=None)
    orc0 = trainer.run(task, "laprop_dana", "oracle", [dict(lr=lr, c=0.0, kappa=0.0, b2=0.99, eps=1e-8, delta=4.0, cap=1.0)], KEYS, S=1, B=8, n_max=3000, eval_pts=ev, log=None)
    print(f"lr={lr}: laprop risk {np.round(lap['risk'][:,0,0],5)} | laprop_dana(g3=0) {np.round(orc0['risk'][:,0,0],5)}", flush=True)
