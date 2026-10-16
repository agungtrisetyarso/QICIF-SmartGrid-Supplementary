"""Learned baselines: SFQ-type quantum circuit (cf. Soltaninia & Zhan 2025) and PINNs."""
import time
import numpy as np
from scipy.optimize import minimize
import torch

# ---------------- SFQ-type two-qubit circuit (statevector, exact) ----------------
I2 = np.eye(2); CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], float)
Z0 = np.kron(np.diag([1., -1.]), I2)


def ry(th):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -s], [s, c]])


def sfq_forward(p, x, L):
    """Data re-uploading two-qubit circuit. x: (n,) normalised time in [0,1].
    params: per layer [w (input frequency), b, th1, th2] -> 4L, then tau0..tau2."""
    n = len(x)
    psi = np.zeros((n, 4)); psi[:, 0] = 1.0
    for l in range(L):
        w, b, t1, t2 = p[4 * l:4 * l + 4]
        ang = w * x + b
        c, s = np.cos(ang / 2), np.sin(ang / 2)
        # embedding I (x) Ry(ang) on qubit 1, then trainable Ry on both, then CNOT
        U1 = np.kron(ry(t1), I2)
        psi = psi.reshape(n, 2, 2)
        a0 = c[:, None] * psi[:, :, 0] - s[:, None] * psi[:, :, 1]
        a1 = s[:, None] * psi[:, :, 0] + c[:, None] * psi[:, :, 1]
        psi = np.stack([a0, a1], axis=2).reshape(n, 4)
        psi = psi @ (np.kron(ry(t1), ry(t2)).T) @ CNOT.T
    u = np.einsum('ni,ij,nj->n', psi, Z0, psi)
    t0, t1_, t2_ = p[4 * L:4 * L + 3]
    return t0 + t1_ * u + t2_ * u ** 2


def train_sfq(t, y, L=6, seed=0, maxiter=3000):
    rng = np.random.default_rng(seed)
    x = (t - t[0]) / (t[-1] - t[0])
    scale = np.max(np.abs(y))
    yn = y / scale
    p0 = np.r_[np.column_stack([rng.uniform(5, 60, L), rng.uniform(-np.pi, np.pi, L),
                                rng.uniform(-np.pi, np.pi, (L, 2))]).ravel(), 0.0, 1.0, 0.0]
    loss = lambda p: np.mean((sfq_forward(p, x, L) - yn) ** 2)
    t_start = time.perf_counter()
    best = None
    for k in range(4):     # multi-start: the landscape is highly non-convex
        p0k = p0 if k == 0 else np.r_[np.column_stack([rng.uniform(5, 60, L), rng.uniform(-np.pi, np.pi, L),
                                rng.uniform(-np.pi, np.pi, (L, 2))]).ravel(), 0.0, 1.0, 0.0]
        r = minimize(loss, p0k, method="BFGS", options={"maxiter": maxiter})
        if best is None or r.fun < best.fun:
            best = r
    elapsed = time.perf_counter() - t_start
    predict = lambda tt: scale * sfq_forward(best.x, (tt - t[0]) / (t[-1] - t[0]), L)
    return predict, elapsed, best.fun


# ---------------- PINN on the full linear swing model ----------------
class Net(torch.nn.Module):
    def __init__(self, n_in, n_out, width=64, nfreq=16, fmax=2.5, T=15.0):
        super().__init__()
        self.B = torch.linspace(0.05, fmax, nfreq) * 2 * np.pi     # Fourier features (Hz -> rad/s)
        self.n_in = n_in
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2 * nfreq + n_in, width), torch.nn.Tanh(),
                                       torch.nn.Linear(width, width), torch.nn.Tanh(),
                                       torch.nn.Linear(width, n_out))

    def forward(self, t, a=None):
        ph = t[:, None] * self.B[None, :]
        feats = [torch.sin(ph), torch.cos(ph), t[:, None] / 15.0]
        if a is not None:
            feats.append(a[:, None])
        return self.mlp(torch.cat(feats, 1))


def train_pinn(Afun, x0, T=15.0, a_range=None, iters=6000, seed=0, n_col=1500, lbfgs=0):
    """Afun(a) -> continuous-time state matrix (8x8). x0: initial state (8,) (state-scaled).
    If a_range is None: single-instance PINN at Afun(None). Output state x(t) = x0 + t*N(t)."""
    torch.manual_seed(seed)
    parametric = a_range is not None
    net = Net(2 if parametric else 1, 8)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    x0t = torch.tensor(x0, dtype=torch.float32)
    scale = float(np.max(np.abs(x0))) or 1.0
    if parametric:
        a_grid = np.linspace(*a_range, 9)
        As = {round(a, 6): torch.tensor(Afun(a), dtype=torch.float32) for a in a_grid}
    else:
        A0 = torch.tensor(Afun(None), dtype=torch.float32)

    def state(t, a=None):
        return x0t[None, :] + scale * t[:, None] * net(t, a)

    t0 = time.perf_counter()
    for it in range(iters):
        t = (torch.rand(n_col) * T).requires_grad_(True)
        if parametric:
            idx = np.random.default_rng(it).integers(0, len(a_grid), n_col)
            av = torch.tensor(a_grid[idx], dtype=torch.float32)
            x = state(t, av)
            Aint = torch.stack([As[round(a_grid[i], 6)] for i in idx])
            rhs = torch.einsum('nij,nj->ni', Aint, x)
        else:
            x = state(t)
            rhs = x @ A0.T
        dx = torch.stack([torch.autograd.grad(x[:, j].sum(), t, create_graph=True)[0] for j in range(8)], 1)
        loss = ((dx - rhs) ** 2).mean() / scale ** 2
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if lbfgs and not parametric:      # second-order refinement on a fixed collocation set
        tf = torch.linspace(0, T, 2 * n_col).requires_grad_(True)
        lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs, line_search_fn="strong_wolfe")
        def closure():
            lb.zero_grad()
            x = state(tf); rhs = x @ A0.T
            dx = torch.stack([torch.autograd.grad(x[:, j].sum(), tf, create_graph=True)[0] for j in range(8)], 1)
            l = ((dx - rhs) ** 2).mean() / scale ** 2
            l.backward(); return l
        lb.step(closure)
        loss = closure()
    elapsed = time.perf_counter() - t0

    def predict(tt, a=None):
        with torch.no_grad():
            tt_ = torch.tensor(tt, dtype=torch.float32)
            av = None if a is None else torch.full_like(tt_, a)
            return state(tt_, av).numpy().T
    return predict, elapsed, float(loss.detach())


def train_pinn_marching(Afun, x0, T=15.0, seg=1.5, iters=2000, seed=0, a_range=None, n_col=600, lbfgs=300):
    """Time-marching PINN: the horizon is split into segments trained in sequence, each with
    a hard initial condition taken from the end of the previous segment (causal training).
    Parametric version (a_range given): a is an extra input in every segment."""
    nseg = int(round(T / seg))
    preds = []; total = 0.0
    xs = x0.copy()
    starts_param = None
    for s in range(nseg):
        if a_range is None:
            p, el, l = train_pinn(Afun, xs, T=seg, iters=iters, seed=seed + s, n_col=n_col, lbfgs=lbfgs)
            preds.append(p); total += el
            xs = p(np.array([seg]))[:, 0]
        else:
            raise NotImplementedError
    def predict(tt, a=None):
        out = np.empty((8, len(tt)))
        for i, t in enumerate(tt):
            k = min(int(t // seg), nseg - 1)
            out[:, i] = preds[k](np.array([t - k * seg]))[:, 0]
        return out
    return predict, total, l


def train_pinn_marching_param(Afun, x0_fun, a_range, T=15.0, seg=1.5, iters=2500, seed=0, n_col=1200, n_a=7, lbfgs=300):
    """Parametric time-marching PINN with input (t, a). The initial state of segment s+1 for each
    training value of a is the segment-s prediction at that a; between training values the
    network interpolates in a."""
    torch.manual_seed(seed)
    nseg = int(round(T / seg))
    a_grid = np.linspace(*a_range, n_a)
    As = torch.stack([torch.tensor(Afun(a), dtype=torch.float32) for a in a_grid])
    x_start = torch.tensor(np.array([x0_fun(a) for a in a_grid]), dtype=torch.float32)
    nets, total = [], 0.0
    for s in range(nseg):
        net = Net(2, 8, T=seg)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        scale = float(x_start.abs().max())
        xs_t = x_start.clone()
        t0 = time.perf_counter()
        for it in range(iters):
            t = (torch.rand(n_col) * seg).requires_grad_(True)
            idx = torch.randint(0, n_a, (n_col,))
            av = torch.tensor(a_grid, dtype=torch.float32)[idx]
            x = xs_t[idx] + scale * t[:, None] * net(t, av)
            rhs = torch.einsum('nij,nj->ni', As[idx], x)
            dx = torch.stack([torch.autograd.grad(x[:, j].sum(), t, create_graph=True)[0] for j in range(8)], 1)
            loss = ((dx - rhs) ** 2).mean() / scale ** 2
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if lbfgs:
            tf = torch.linspace(0, seg, 300).repeat(n_a).requires_grad_(True)
            idf = torch.arange(n_a).repeat_interleave(300)
            avf = torch.tensor(a_grid, dtype=torch.float32)[idf]
            lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs, line_search_fn="strong_wolfe")
            def closure():
                lb.zero_grad()
                x = xs_t[idf] + scale * tf[:, None] * net(tf, avf)
                rhs = torch.einsum('nij,nj->ni', As[idf], x)
                dx = torch.stack([torch.autograd.grad(x[:, j].sum(), tf, create_graph=True)[0] for j in range(8)], 1)
                l = ((dx - rhs) ** 2).mean() / scale ** 2
                l.backward(); return l
            lb.step(closure); loss = closure()
        total += time.perf_counter() - t0
        with torch.no_grad():
            tt = torch.full((n_a,), seg); av = torch.tensor(a_grid, dtype=torch.float32)
            x_end = xs_t + scale * tt[:, None] * net(tt, av)
        nets.append((net, xs_t.clone(), scale))
        x_start = x_end

    def predict(tt, a):
        out = np.empty((8, len(tt)))
        with torch.no_grad():
            # initial state for arbitrary a: interpolate the segment start states linearly in a
            for i, t in enumerate(tt):
                k = min(int(t // seg), nseg - 1)
                net, xs_t, scale = nets[k]
                x0a = torch.tensor(np.array([np.interp(a, a_grid, xs_t[:, j].numpy()) for j in range(8)]), dtype=torch.float32)
                tl = torch.tensor([t - k * seg], dtype=torch.float32)
                out[:, i] = (x0a + scale * tl * net(tl, torch.tensor([a], dtype=torch.float32))[0]).numpy()
        return out
    return predict, total, float(loss.detach())
