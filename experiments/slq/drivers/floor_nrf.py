"""Irreducible risk of the nonlinear-random-features tasks: least-squares optimum from 2M samples (float64 normal
equations), evaluated on the task's fixed eval set (the set used in all E2 risk numbers)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from adana import tasks
out = {}
for (a, b, v) in ((1.0, 0.7, 2048), (0.4, 0.5, 4096)):
    task = tasks.nonlinear_rf(v=v, d=1024, alpha=a, beta=b, act="relu", n_eval=16384)
    d = 1024
    A = np.zeros((d, d)); c = np.zeros(d)
    key = jax.random.PRNGKey(123)
    feat = jax.jit(lambda prm, bt: jax.jacobian(lambda th: task["apply"](dict(theta=th), bt))(prm))
    for i in range(100):
        key, k = jax.random.split(key)
        bt = task["sample"](k, 20000)
        Phi = np.asarray(feat(jnp.zeros(d, jnp.float32), bt), dtype=np.float64)   # (n, d): f = Phi theta
        A += Phi.T @ Phi; c += Phi.T @ np.asarray(bt["y"], dtype=np.float64)
    th = np.linalg.solve(A, c)
    floor = float(task["evaluate"](dict(theta=jnp.asarray(th, jnp.float32)))["risk"])
    out[task["name"]] = floor
    print(task["name"], "floor risk on eval set:", floor, flush=True)
json.dump(out, open(os.path.expanduser("~/dana-exp/runs/e2/floors.json"), "w"), indent=1)
