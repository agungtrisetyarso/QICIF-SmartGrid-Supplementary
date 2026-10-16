"""Operator (two-qubit) model of the two-area reduced dynamics, solved with QuTiP mesolve."""
import numpy as np
import qutip as qt
from scipy.linalg import expm
import model as m

A1 = qt.tensor(qt.destroy(2), qt.qeye(2))
A2 = qt.tensor(qt.qeye(2), qt.destroy(2))
AK = [A1, A2]


def hamiltonian(Om):
    """H = sum_kl Om_kl a_k^dag a_l = sum_k Om_kk n_k + Om_12 (s+ s- + s- s+)."""
    return sum(Om[k, l] * AK[k].dag() * AK[l] for k in range(2) for l in range(2))


def collapse_ops(Gam):
    """Correlated amplitude-damping channels; requires Gamma >= 0 (CP condition)."""
    g, U = np.linalg.eigh(Gam)
    if g.min() < -1e-12:
        raise ValueError(f"Gamma not PSD (min eig {g.min():.3e}): no Lindblad realisation")
    return [np.sqrt(max(gm, 0)) * (U[0, mu] * A1 + U[1, mu] * A2) for mu, gm in enumerate(g)]


def encode(Om, Gam, Mr, x0, xd0):
    """Reduced classical initial condition -> pure two-qubit state and scale.
    psi = Omega q + i (qdot + Gamma q / 2), q = Mr^1/2 x."""
    S = m.msqrt(Mr)
    q0 = S @ x0; qd0 = S @ xd0
    psi = Om @ q0 + 1j * (qd0 + 0.5 * Gam @ q0)
    nrm = np.linalg.norm(psi)
    amp = psi / (np.sqrt(2) * nrm)
    vac = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    e1 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0)); e2 = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
    state = (vac / np.sqrt(2) + complex(amp[0]) * e1 + complex(amp[1]) * e2).unit()
    return state, 2 * nrm   # <a_k>(t) * scale = psi_k(t)


def run_operator(Mr, Dr, Kr, x0, xd0, t, options=None):
    Om, Gam = m.operator_block(Mr, Dr, Kr)
    rho0, scale = encode(Om, Gam, Mr, x0, xd0)
    res = qt.mesolve(hamiltonian(Om), rho0, t, c_ops=collapse_ops(Gam),
                     e_ops=[A1, A2], options=options or {"atol": 1e-10, "rtol": 1e-8})
    psi = scale * np.vstack([np.asarray(res.expect[0]), np.asarray(res.expect[1])])
    Si = np.linalg.inv(m.msqrt(Mr))
    q = np.linalg.solve(Om, psi.real)
    qd = psi.imag - 0.5 * Gam @ q
    return Si @ q, Si @ qd, res          # reduced coordinates and their rates


def run_reduced_classical(Mr, Dr, Kr, x0, xd0, t):
    Mi = np.linalg.inv(Mr)
    A = np.zeros((4, 4)); A[:2, 2:] = np.eye(2)
    A[2:, :2] = -Mi @ Kr; A[2:, 2:] = -Mi @ Dr
    x0 = np.r_[x0, xd0]
    X = np.array([expm(A * tt) @ x0 for tt in t]).T
    return X[:2], X[2:]


def run_full(M, D, K, d0, w0, t):
    A = m.full_state_matrix(M, D, K)
    x0 = np.r_[d0, w0]
    dt = t[1] - t[0]; E = expm(A * dt)
    X = np.empty((8, len(t))); X[:, 0] = x0
    for i in range(1, len(t)):
        X[:, i] = E @ X[:, i - 1]
    return X[:4], X[4:]


def coi(M, x):
    P = np.zeros((2, 4)); P[0, m.AREA == 0] = 1; P[1, m.AREA == 1] = 1
    w = (P @ M)
    return ((P * M) @ x) / (w[:, None] if np.ndim(x) == 2 else w)


def project(M, Wa, x):
    """M-orthogonal projection of a full machine vector onto reduced coordinates."""
    Mr = Wa.T @ np.diag(M) @ Wa
    return np.linalg.solve(Mr, Wa.T @ (M * x))


if __name__ == "__main__":
    t = np.linspace(0, 15, 1501)
    Wa = m.slow_basis()
    M, D = m.machine_params(0.25); K = m.stiffness()
    w0 = np.array([0, 0, -0.1, -0.1]); d0 = np.zeros(4)
    Mr, Dr, Kr = m.reduce(M, D, K, Wa)
    x0, xd0 = project(M, Wa, d0), project(M, Wa, w0)
    xq, xdq, _ = run_operator(Mr, Dr, Kr, x0, xd0, t)
    xc, xdc = run_reduced_classical(Mr, Dr, Kr, x0, xd0, t)
    df, wf = run_full(M, D, K, d0, w0, t)
    to = 1e3 / (2 * np.pi)
    a2 = lambda xd: coi(M, Wa @ xd)[1]
    truth = coi(M, wf)[1]
    print("RWA (op vs reduced classical) RMSE mHz", np.sqrt(np.mean((a2(xdq) - a2(xdc))**2))*to)
    print("op vs full RMSE mHz", np.sqrt(np.mean((a2(xdq) - truth)**2))*to, "peak", np.abs(truth).max()*to)
