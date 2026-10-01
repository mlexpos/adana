import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adana.sim_ls import make_configs, simulate
from adana.plrf import gaussian_ls
lam = np.arange(1, 4097.) ** -2.
inst = gaussian_ls(lam, np.ones(4096) * 0.01, 1e-4)
hp = make_configs([dict(g2=0.2, mode="adapt", s=0.25, seed=s % 4) for s in range(576)])
simulate(inst, hp, 2, 200, n_pts=3)            # compile
t0 = time.time(); simulate(inst, hp, 2, 20000, n_pts=3); print(os.environ.get("XLA_FLAGS", "default"), f"{(time.time()-t0)/20000*1e3:.3f} ms/step")
