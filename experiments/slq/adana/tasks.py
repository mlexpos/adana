"""Tasks for E2 (nonlinear PLRF-like panel) and E3 (smooth modular arithmetic).

Each task provides:
  init(key) -> params
  sample(key, B) -> batch
  apply(params, batch) -> predictions (regression: (B,), classification: (B, T, V) logits)
  loss(params, batch) -> mean loss
  sampled_loss(params, batch, key) -> mean loss with model-sampled labels (gradient = model-sampled gradient)
  eval(params) -> dict of population-ish metrics on a fixed held-out set
"""
import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------- E1 in pytree form: linear PLRF (original, non-diagonal basis)
def plrf(v=8192, d=4096, alpha=1.0, beta=0.7, seed=0):
    """Linear PLRF, x_j = j^-alpha z_j (j <= v), W ~ N(0, 1/d) (v x d), prediction <W^T x, theta>, target <x, b>, b_j = j^-beta.
    Sampled exactly in reduced form: features x~ = W^T x ~ N(0, K), K = W^T D W = V diag(lam) V^T, and y = <x~, theta*> + eps,
    Var eps = P* (independent of x~).  theta lives in the ORIGINAL (random, non-eigen) basis so diagonal preconditioners get
    no free rotation.  Exact excess risk  sum_j lam_j (V^T(theta - theta*))_j^2  (reported as 'risk')."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((v, d)) / np.sqrt(d)
    j = np.arange(1, v + 1, dtype=np.float64)
    Dg = j ** (-2 * alpha); b = j ** (-beta)
    K = (W * Dg[:, None]).T @ W
    lam, V = np.linalg.eigh(K); lam = np.maximum(lam, 0.0)
    th_star = np.linalg.lstsq(K, W.T @ (Dg * b), rcond=None)[0]
    Pstar = float(np.sum(Dg * (W @ th_star - b) ** 2))
    A = jnp.asarray(V * np.sqrt(lam)[None, :], jnp.float32)            # x~ = A z
    Vt = jnp.asarray(V.T, jnp.float32); sl = jnp.asarray(np.sqrt(lam), jnp.float32)
    ths = jnp.asarray(th_star, jnp.float32); sP = float(np.sqrt(max(Pstar, 0.0)))

    def sample(key, B):
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (B, d)) @ A.T
        return dict(x=x, y=x @ ths + sP * jax.random.normal(k2, (B,)))

    def init(key):
        return dict(theta=jnp.zeros((d,), jnp.float32))

    def apply(params, batch):
        return batch["x"] @ params["theta"]

    def loss(params, batch):
        return 0.5 * jnp.mean((apply(params, batch) - batch["y"]) ** 2)

    def sampled_loss(params, batch, key):
        f = apply(params, batch)
        eps = jax.random.normal(key, f.shape)
        return jnp.mean(-jax.lax.stop_gradient(eps) * f)

    def gnvp(params, batch, u):
        f = lambda prm: apply(prm, batch)
        _, Ju = jax.jvp(f, (params,), (u,))
        _, vjp = jax.vjp(f, params)
        return vjp(Ju / Ju.shape[0])[0]

    def evaluate(params):
        return dict(risk=jnp.sum((sl * (Vt @ (params["theta"] - ths))) ** 2))
    return dict(name=f"plrf_a{alpha}_b{beta}_d{d}_v{v}", init=init, sample=sample, apply=apply, loss=loss,
                sampled_loss=sampled_loss, evaluate=evaluate, gnvp=gnvp, lam=lam, Pstar=Pstar)


# ---------------------------------------------------------------- E2a: nonlinear random features (trainable readout)
def nonlinear_rf(v=2048, d=1024, alpha=1.0, beta=0.7, act="relu", seed=0, n_eval=65536):
    r = np.random.default_rng(seed)
    W = jnp.asarray(r.standard_normal((v, d)) / np.sqrt(v), jnp.float32)
    sq = jnp.asarray(np.arange(1, v + 1.0) ** (-alpha), jnp.float32)
    b = jnp.asarray(np.arange(1, v + 1.0) ** (-beta), jnp.float32)
    phi = {"relu": lambda z: jnp.sqrt(2.0) * jax.nn.relu(z), "tanh": jnp.tanh}[act]
    zscale = float(np.sqrt(np.sum(np.arange(1, v + 1.0) ** (-2 * alpha)) / v))   # std of (W^T x)_k

    def feats(x):
        return phi(x @ W / zscale)

    def sample(key, B):
        x = jax.random.normal(key, (B, v)) * sq
        return dict(x=x, y=x @ b)

    def init(key):
        return dict(theta=jnp.zeros((d,), jnp.float32))

    def apply(params, batch):
        return feats(batch["x"]) @ params["theta"] / np.sqrt(d)

    def loss(params, batch):
        return 0.5 * jnp.mean((apply(params, batch) - batch["y"]) ** 2)

    def sampled_loss(params, batch, key):
        f = apply(params, batch)
        eps = jax.random.normal(key, f.shape)
        return jnp.mean(-jax.lax.stop_gradient(eps) * f)


    def gnvp(params, batch, u):
        f = lambda prm: apply(prm, batch)
        _, Ju = jax.jvp(f, (params,), (u,))
        _, vjp = jax.vjp(f, params)
        return vjp(Ju / Ju.shape[0])[0]

    ev = sample(jax.random.PRNGKey(10_000 + seed), n_eval)

    def evaluate(params):
        return dict(risk=2 * loss(params, ev))      # P = E(f - y)^2
    return dict(name=f"nrf_{act}_a{alpha}_b{beta}_d{d}", init=init, sample=sample, apply=apply, loss=loss,
                sampled_loss=sampled_loss, evaluate=evaluate, gnvp=gnvp)


# ---------------------------------------------------------------- E2b: two-layer tanh teacher-student, power-law inputs
def two_layer(v=512, m=256, alpha=1.0, teacher_width=512, seed=0, n_eval=65536, noise=0.0):
    r = np.random.default_rng(seed)
    sq = jnp.asarray(np.arange(1, v + 1.0) ** (-alpha), jnp.float32)
    zs = float(np.sqrt(np.sum(np.arange(1, v + 1.0) ** (-2 * alpha))))
    Wt = jnp.asarray(r.standard_normal((v, teacher_width)), jnp.float32)
    at = jnp.asarray(r.standard_normal((teacher_width,)) * np.arange(1, teacher_width + 1.0) ** (-0.5), jnp.float32)

    def teacher(x):
        return jnp.tanh(x @ Wt / zs) @ at

    def sample(key, B):
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (B, v)) * sq
        return dict(x=x, y=teacher(x) + noise * jax.random.normal(k2, (B,)))

    def init(key):
        k1, _ = jax.random.split(key)
        return dict(W1=jax.random.normal(k1, (v, m)) / 1.0, a=jnp.zeros((m,), jnp.float32))

    def apply(params, batch):
        h = jnp.tanh(batch["x"] @ params["W1"] / zs)
        return h @ params["a"] / np.sqrt(m)

    def loss(params, batch):
        return 0.5 * jnp.mean((apply(params, batch) - batch["y"]) ** 2)

    def sampled_loss(params, batch, key):
        f = apply(params, batch)
        eps = jax.random.normal(key, f.shape)
        return jnp.mean(-jax.lax.stop_gradient(eps) * f)


    def gnvp(params, batch, u):
        f = lambda prm: apply(prm, batch)
        _, Ju = jax.jvp(f, (params,), (u,))
        _, vjp = jax.vjp(f, params)
        return vjp(Ju / Ju.shape[0])[0]

    ev = sample(jax.random.PRNGKey(20_000 + seed), n_eval)

    def evaluate(params):
        return dict(risk=2 * loss(params, ev))
    return dict(name=f"2layer_a{alpha}_v{v}_m{m}", init=init, sample=sample, apply=apply, loss=loss,
                sampled_loss=sampled_loss, evaluate=evaluate, gnvp=gnvp)


# ---------------------------------------------------------------- E4: deep MLP teacher-student, power-law inputs
def deep_mlp(v=256, width=128, depth=3, alpha=1.0, teacher_width=256, teacher_depth=3, seed=0, n_eval=16384, noise=0.0):
    """Student: depth tanh hidden layers (with biases) + linear readout, inputs x ~ N(0, diag(k^-2alpha)).
    Teacher: teacher_depth tanh layers of width teacher_width, readout weights ~ k^-1/2.  MSE; risk = E(f - y)^2."""
    r = np.random.default_rng(seed)
    sq = jnp.asarray(np.arange(1, v + 1.0) ** (-alpha), jnp.float32)
    zs = float(np.sqrt(np.sum(np.arange(1, v + 1.0) ** (-2 * alpha))))
    tW = [jnp.asarray(r.standard_normal((v, teacher_width)) / zs, jnp.float32)] + \
         [jnp.asarray(r.standard_normal((teacher_width, teacher_width)) / np.sqrt(teacher_width), jnp.float32)
          for _ in range(teacher_depth - 1)]
    ta = jnp.asarray(r.standard_normal((teacher_width,)) * np.arange(1, teacher_width + 1.0) ** (-0.5), jnp.float32)

    def teacher(x):
        h = x
        for W in tW:
            h = jnp.tanh(h @ W)
        return h @ ta

    def sample(key, B):
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (B, v)) * sq
        return dict(x=x, y=teacher(x) + noise * jax.random.normal(k2, (B,)))

    def init(key):
        ks = jax.random.split(key, depth)
        p = {}
        fan = [v] + [width] * depth
        for i in range(depth):
            scale = zs if i == 0 else np.sqrt(fan[i])
            p[f"W{i}"] = jax.random.normal(ks[i], (fan[i], width)) / scale
            p[f"b{i}"] = jnp.zeros((width,), jnp.float32)
        p["a"] = jnp.zeros((width,), jnp.float32)
        return p

    def apply(params, batch):
        h = batch["x"]
        for i in range(depth):
            h = jnp.tanh(h @ params[f"W{i}"] + params[f"b{i}"])
        return h @ params["a"] / np.sqrt(width)

    def loss(params, batch):
        return 0.5 * jnp.mean((apply(params, batch) - batch["y"]) ** 2)

    def sampled_loss(params, batch, key):
        f = apply(params, batch)
        eps = jax.random.normal(key, f.shape)
        return jnp.mean(-jax.lax.stop_gradient(eps) * f)

    def gnvp(params, batch, u):
        f = lambda prm: apply(prm, batch)
        _, Ju = jax.jvp(f, (params,), (u,))
        _, vjp = jax.vjp(f, params)
        return vjp(Ju / Ju.shape[0])[0]

    ev = sample(jax.random.PRNGKey(30_000 + seed), n_eval)

    def evaluate(params):
        return dict(risk=2 * loss(params, ev))
    return dict(name=f"mlp_a{alpha}_v{v}_w{width}_L{depth}", init=init, sample=sample, apply=apply, loss=loss,
                sampled_loss=sampled_loss, evaluate=evaluate, gnvp=gnvp)


# ---------------------------------------------------------------- E3: smooth modular arithmetic, small transformer
def _eta_probs(width, sig):
    ks = np.arange(-width, width + 1)
    w = np.exp(-0.5 * (ks / sig) ** 2)
    return ks, w / w.sum()


def modarith(p=97, T=32, zipf=1.2, eta_sig=0.7, eta_w=3, d_model=128, n_layers=2, n_heads=4, seed=0,
             n_eval=2048, mlp_mult=4):
    """Sequences s_{t+1} = s_t + a + eta_t (mod p). Hidden increment a ~ Zipf(zipf) over a random permutation of Z_p;
    eta_t ~ discretized Gaussian (std eta_sig, support +-eta_w). Next-token CE; Bayes loss computed exactly."""
    r = np.random.default_rng(seed)
    perm = r.permutation(p)
    pa = np.arange(1, p + 1.0) ** (-zipf); pa /= pa.sum()
    prob_a = np.zeros(p); prob_a[perm] = pa
    ks, pk = _eta_probs(eta_w, eta_sig)
    logit_a = jnp.asarray(np.log(prob_a), jnp.float32)
    logit_k = jnp.asarray(np.log(pk), jnp.float32)
    ks_j = jnp.asarray(ks)

    def sample(key, B):
        k1, k2, k3 = jax.random.split(key, 3)
        a = jax.random.categorical(k1, logit_a, shape=(B,))
        s0 = jax.random.randint(k2, (B,), 0, p)
        eta = ks_j[jax.random.categorical(k3, logit_k, shape=(B, T))]
        inc = a[:, None] + eta
        s = (s0[:, None] + jnp.concatenate([jnp.zeros((B, 1), inc.dtype), jnp.cumsum(inc[:, :-1], 1)], 1)) % p
        return dict(tok=s.astype(jnp.int32))

    D, H, L = d_model, n_heads, n_layers
    def init(key):
        ks_ = jax.random.split(key, 4 + 6 * L)
        sc = 1 / np.sqrt(D)
        prm = dict(emb=jax.random.normal(ks_[0], (p, D)) * sc, pos=jax.random.normal(ks_[1], (T, D)) * 0.02,
                   out=jax.random.normal(ks_[2], (D, p)) * sc, lnf=jnp.ones((D,)))
        for l in range(L):
            kk = ks_[4 + 6 * l: 4 + 6 * (l + 1)]
            prm[f"l{l}"] = dict(q=jax.random.normal(kk[0], (D, D)) * sc, k=jax.random.normal(kk[1], (D, D)) * sc,
                                v=jax.random.normal(kk[2], (D, D)) * sc, o=jax.random.normal(kk[3], (D, D)) * sc / np.sqrt(2 * L),
                                w1=jax.random.normal(kk[4], (D, mlp_mult * D)) * sc,
                                w2=jax.random.normal(kk[5], (mlp_mult * D, D)) * sc / np.sqrt(mlp_mult * 2 * L),
                                ln1=jnp.ones((D,)), ln2=jnp.ones((D,)))
        return prm

    def ln(x, g):
        mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return g * (x - mu) / jnp.sqrt(var + 1e-5)

    mask = jnp.tril(jnp.ones((T, T), bool))

    def apply(prm, batch):
        x = prm["emb"][batch["tok"]] + prm["pos"][None]
        Bn = x.shape[0]
        for l in range(L):
            P_ = prm[f"l{l}"]
            h = ln(x, P_["ln1"])
            q = (h @ P_["q"]).reshape(Bn, T, H, D // H)
            k = (h @ P_["k"]).reshape(Bn, T, H, D // H)
            v = (h @ P_["v"]).reshape(Bn, T, H, D // H)
            att = jnp.einsum("bthd,bshd->bhts", q, k) / np.sqrt(D // H)
            att = jnp.where(mask[None, None], att, -1e9)
            att = jax.nn.softmax(att, -1)
            o = jnp.einsum("bhts,bshd->bthd", att, v).reshape(Bn, T, D)
            x = x + o @ P_["o"]
            h = ln(x, P_["ln2"])
            x = x + jax.nn.gelu(h @ P_["w1"]) @ P_["w2"]
        return ln(x, prm["lnf"]) @ prm["out"]

    def apply_probe(prm, batch, probes):
        """apply() with additive residual-stream probes: probes[0] at the input (after embedding), probes[l+1] after
        block l.  Gradients w.r.t. the (zero) probes are the per-position backprop signals dL/dh^l_t of every layer."""
        x = prm["emb"][batch["tok"]] + prm["pos"][None] + probes[0]
        Bn = x.shape[0]
        for l in range(L):
            P_ = prm[f"l{l}"]
            h = ln(x, P_["ln1"])
            q = (h @ P_["q"]).reshape(Bn, T, H, D // H)
            k = (h @ P_["k"]).reshape(Bn, T, H, D // H)
            v = (h @ P_["v"]).reshape(Bn, T, H, D // H)
            att = jnp.einsum("bthd,bshd->bhts", q, k) / np.sqrt(D // H)
            att = jnp.where(mask[None, None], att, -1e9)
            att = jax.nn.softmax(att, -1)
            o = jnp.einsum("bhts,bshd->bthd", att, v).reshape(Bn, T, D)
            x = x + o @ P_["o"]
            h = ln(x, P_["ln2"])
            x = x + jax.nn.gelu(h @ P_["w1"]) @ P_["w2"] + probes[l + 1]
        return ln(x, prm["lnf"]) @ prm["out"]

    def ce(logits, tgt):
        lp = jax.nn.log_softmax(logits, -1)
        return -jnp.take_along_axis(lp, tgt[..., None], -1)[..., 0]

    def loss_w(prm, batch, w):
        """token-weighted loss: sum_{b,t} w_bt ce_bt / sum(w) (w: (B, T-1)); token-split halves use w = mask, 1 - mask"""
        lg = apply(prm, batch)[:, :-1]
        return jnp.sum(w * ce(lg, batch["tok"][:, 1:])) / jnp.maximum(jnp.sum(w), 1.0)

    def loss_probe(prm, batch, probes):
        lg = apply_probe(prm, batch, probes)[:, :-1]
        return jnp.mean(ce(lg, batch["tok"][:, 1:]))

    def teff(prm, batch):
        """effective number of independent tokens per sequence from the LAST block's residual-stream gradient
        (validated against the sequence-vs-token split ratio): T' / (E||sum_t d_t||^2 / E sum_t ||d_t||^2)"""
        Bn = batch["tok"].shape[0]
        probes = [jnp.zeros((Bn, T, D)) for _ in range(L + 1)]
        d = jax.grad(loss_probe, argnums=2)(prm, batch, probes)[-1][:, :T - 1]
        d = d - d.mean(0, keepdims=True)
        R = jnp.mean(jnp.sum(d.sum(1) ** 2, -1)) / jnp.maximum(jnp.mean(jnp.sum(d ** 2, (1, 2))), 1e-30)
        return jnp.clip((T - 1) / jnp.maximum(R, 1e-6), 1.0, T - 1.0)

    def loss(prm, batch):
        lg = apply(prm, batch)[:, :-1]
        return jnp.mean(ce(lg, batch["tok"][:, 1:]))

    def sampled_loss(prm, batch, key):
        lg = apply(prm, batch)[:, :-1]
        yt = jax.lax.stop_gradient(jax.random.categorical(key, lg))
        return jnp.mean(ce(lg, yt))


    def gnvp(prm, batch, u):
        f = lambda q: apply(q, batch)[:, :-1]
        lg, Ju = jax.jvp(f, (prm,), (u,))
        p_ = jax.nn.softmax(lg, -1)
        LJu = p_ * Ju - p_ * jnp.sum(p_ * Ju, -1, keepdims=True)
        _, vjp = jax.vjp(f, prm)
        return vjp(LJu / (lg.shape[0] * lg.shape[1]))[0]

    ev = sample(jax.random.PRNGKey(30_000 + seed), n_eval)
    # exact Bayes loss on the eval set: posterior over a given prefix increments
    tok = np.asarray(ev["tok"]); incs = (tok[:, 1:] - tok[:, :-1]) % p            # (n, T-1)
    lk = np.full(p, -np.inf); lk_eta = np.full((p,), -np.inf)
    eta_lp = np.full(p, -np.inf)
    for kk, pp in zip(ks, pk):
        eta_lp[kk % p] = np.log(pp)
    logpost = np.log(prob_a)[None].repeat(len(tok), 0)                             # (n, p)
    bayes = []
    for t in range(T - 1):
        # predictive for inc_t: sum_a post(a) P(eta = inc - a)
        post = np.exp(logpost - logpost.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        lp_inc = eta_lp[(incs[:, t][:, None] - np.arange(p)[None]) % p]            # (n, p)
        bayes.append(-np.log(np.sum(post * np.exp(lp_inc), 1)))
        logpost = logpost + lp_inc
    bayes_loss = float(np.mean(np.stack(bayes, 1)))

    def evaluate(prm):
        return dict(risk=loss(prm, ev) - bayes_loss, loss=loss(prm, ev))
    return dict(name=f"modarith_p{p}_T{T}_z{zipf}_s{eta_sig}_D{D}_L{L}", init=init, sample=sample, apply=apply,
                loss=loss, sampled_loss=sampled_loss, evaluate=evaluate, bayes=bayes_loss, gnvp=gnvp, apply_probe=apply_probe, loss_w=loss_w, loss_probe=loss_probe, teff=teff, n_layers=L, d_model=D, seq_len=T)
