import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana import tasks, trainer
KEYS = ("lr", "b1", "b2", "eps", "delta", "cap", "c", "kappa", "s")
for name, task, lrs in [("nrf", tasks.nonlinear_rf(v=512, d=256, n_eval=4096), (0.003, 0.01, 0.03)),
                        ("2layer", tasks.two_layer(v=128, m=64, teacher_width=128, n_eval=4096), (0.003, 0.01, 0.03))]:
    ev = [0, 1000, 10000]
    lap = trainer.run(task, "laprop", "adapt", [dict(lr=l, b1=0.9, b2=0.99, eps=1e-8) for l in lrs], KEYS, S=2, B=8, n_max=10000, eval_pts=ev, log=None)
    fin = lap["risk"][-1].mean(0); lb = lrs[int(np.argmin(fin))]
    print(name, "LaProp final by lr", np.round(fin, 5), "best lr", lb, flush=True)
    for f in (0.1, 0.03, 0.01):
        rows = [dict(lr=f * lb * 10, s=s, b2=0.99, eps=1e-8, delta=4.0, cap=1.0) for s in (0.016, 0.0625, 0.25)]   # g2 = f*lb/(1-b1)... scan
        ad = trainer.run(task, "laprop_dana", "adapt_global", rows, KEYS, S=2, B=8, n_max=10000, eval_pts=ev, log=None)
        print(f"   LaProp-DANA g2={f*lb*10:.2e} (={f*10:.1f} x lb): final by s(.016,.0625,.25) {np.round(ad['risk'][-1].mean(0), 5)}", flush=True)
