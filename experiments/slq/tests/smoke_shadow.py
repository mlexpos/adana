import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
for name, task, lr in [("nrf", tasks.nonlinear_rf(v=512, d=256, n_eval=8192), 0.5), ("2layer", tasks.two_layer(v=128, m=64, teacher_width=128, n_eval=8192), 2.0),
                       ("modarith", tasks.modarith(n_eval=512), 0.25)]:
    ev = [0, 100, 1000, 3000]
    for B in (8, 256):
        out = {}
        for opt, mode in (("sgd", "adapt"), ("dana", "adapt_global"), ("dana_shadow", "shadow")):
            t0 = time.time()
            r = trainer.run(task, opt, mode, [dict(lr=lr, s=0.25, delta=4.0, cap=1.0)], KEYS, S=1, B=B, n_max=3000, eval_pts=ev, log=None)
            out[opt] = (float(r["risk"][-1].mean()), float(r["nhb"][-2].mean()) if "nhb" in r else np.nan, time.time() - t0)
        print(name, "B", B, {k: (f"{v[0]:.4g}", f"NhB {v[1]:.3g}", f"{v[2]:.0f}s") for k, v in out.items()}, flush=True)
