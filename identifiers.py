"""Classical modal identifiers fitted to a sampled ring-down y[k] (uniform step dt)."""
import numpy as np


def _modes_to_fit(z, y, n):
    """Least-squares residues for discrete poles z, return reconstruction and poles."""
    k = np.arange(n)
    V = z[None, :] ** k[:, None]
    r, *_ = np.linalg.lstsq(V, y.astype(complex), rcond=None)
    return np.real(V @ r), r


def prony(y, dt, order, lp_order=60, band=(0.1, 1.2)):
    """Linear-prediction Prony with rank-truncated least squares (Kumaresan-Tufts):
    prediction order lp_order (2 s at 30 frames/s), signal-subspace rank = order; the
    `order` roots of largest modulus inside the electromechanical band are kept."""
    n = len(y); p = lp_order
    A = np.column_stack([y[p - 1 - i:n - 1 - i] for i in range(p)])
    b = y[p:]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    c = Vh[:order].T @ ((U[:, :order].T @ b) / s[:order])
    z = np.roots(np.r_[1, -c])
    f = np.abs(np.angle(z)) / (2 * np.pi * dt)
    cand = np.where((f > band[0]) & (f < band[1]))[0]
    z = z[cand[np.argsort(-np.abs(z[cand]))][:order]]
    yhat, r = _modes_to_fit(z, y, n)
    return yhat, z, r


def matrix_pencil(y, dt, order, L=None):
    n = len(y); L = L or n // 3
    Y = np.array([y[i:i + L + 1] for i in range(n - L)])
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh[:order].conj().T
    V1, V2 = V[:-1], V[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    yhat, r = _modes_to_fit(z, y, n)
    return yhat, z, r


def era(y, dt, order, rows=None):
    """Eigensystem Realization Algorithm on the free response (deterministic subspace)."""
    n = len(y); rows = rows or n // 3
    cols = n - rows - 1
    H0 = np.array([y[i:i + cols] for i in range(rows)])
    H1 = np.array([y[i + 1:i + 1 + cols] for i in range(rows)])
    U, s, Vh = np.linalg.svd(H0, full_matrices=False)
    U, s, Vh = U[:, :order], s[:order], Vh[:order]
    Si = np.diag(s ** -0.5)
    Ad = Si @ U.T @ H1 @ Vh.T @ Si
    z = np.linalg.eigvals(Ad)
    yhat, r = _modes_to_fit(z, y, n)
    return yhat, z, r


def hankel_dmd(y, dt, order, delays=None):
    n = len(y); d = delays or 60
    X = np.array([y[i:n - d + i] for i in range(d)])
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    U, s, Vh = U[:, :order], s[:order], Vh[:order]
    At = U.T @ X2 @ Vh.T / s
    z = np.linalg.eigvals(At)
    yhat, r = _modes_to_fit(z, y, n)
    return yhat, z, r


def predict(z, r, n):
    k = np.arange(n)
    return np.real((z[None, :] ** k[:, None]) @ r)


def dominant_mode(z, r, dt, band=(0.4, 0.9)):
    s = np.log(z.astype(complex)) / dt
    f = np.abs(s.imag) / (2 * np.pi)
    sel = (f > band[0]) & (f < band[1])
    if not sel.any():
        return np.nan, np.nan
    i = np.argmax(np.where(sel, np.abs(r), -1))
    return f[i], -s[i].real / np.abs(s[i])


IDENTIFIERS = {"Prony": prony, "Matrix Pencil": matrix_pencil, "ERA": era, "Hankel-DMD": hankel_dmd}
