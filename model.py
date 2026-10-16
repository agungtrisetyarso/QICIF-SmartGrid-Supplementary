"""Two-area four-machine classical model, coherent aggregation, and the operator model.

Conventions (all per unit on a common base, angles in rad, time in s):
  M_i = 2 H_i / w0,  swing:  M d2(delta)/dt2 + D d(delta)/dt + K delta = u(t)
  Automation share a in [0,1] of each area's capacity is supplied by grid-forming IBRs with
  virtual inertia H_v = Tj/2 and net damping D_v (droop damping minus the negative damping
  torque of fast automated loops). Synchronising coefficients K do not depend on inertia.
"""
import numpy as np
from scipy.linalg import sqrtm, expm, eig

F0 = 60.0
W0 = 2 * np.pi * F0

# ---- nominal parameters (Kundur two-area machine data, classical models, tuned network) ----
H_SYNC = np.array([6.5, 6.5, 6.175, 6.175])   # s, Kundur Example 12.6 machine inertias
AREA = np.array([0, 0, 1, 1])
K_LOC = 1.60      # intra-area tie G1-G2 and G3-G4 (pu/rad)
K_TIE = 0.37      # inter-area tie G2-G3 (pu/rad)
K_EXT = 0.245     # tie from G1 and from G4 to the external bulk grid (pu/rad)
D_SYNC = 0.015     # synchronous damping per machine (pu s/rad)
D_V = 0.012       # droop/virtual damping per machine at full IBR share (pu s/rad)
D_LOC = 0.028     # damping on intra-area relative speed (PSS effect on local modes)
DT_0 = 0.0055     # damping on inter-area tie flow at a = 0 (pu s/rad)
DT_V = -0.020     # inter-area tie-flow damping contributed by automated loops at a = 1;
                  # negative = phase-lagging automated loops acting on tie flow
TJ_NOM = 8.0      # nominal virtual-inertia time constant (s); H_v = Tj/2


def laplacian_tie(i, j, n=4):
    L = np.zeros((n, n)); L[i, i] = L[j, j] = 1; L[i, j] = L[j, i] = -1
    return L


def d_tie(a, dtv=DT_V):
    return (1 - a) * DT_0 + a * dtv


def machine_params(a=0.0, Tj=TJ_NOM, Dv=D_V, a2=None, dtv=DT_V, dscale=1.0):
    """Per-machine inertia vector M and damping MATRIX D for automation share a.

    a  : automation (IBR) share in area 1; a2 in area 2 (default a2 = a).
    The tie-flow damping uses the mean share of the two areas.
    """
    a2 = a if a2 is None else a2
    share = np.where(AREA == 0, a, a2)
    H = (1 - share) * H_SYNC + share * (Tj / 2.0)
    Dd = (1 - share) * D_SYNC + share * Dv
    D = np.diag(Dd) + D_LOC * (laplacian_tie(0, 1) + laplacian_tie(2, 3)) \
        + d_tie(0.5 * (a + a2), dtv) * laplacian_tie(1, 2)
    return 2 * H / W0, dscale * D


def stiffness(k_loc=K_LOC, k_tie=K_TIE, k_ext=K_EXT):
    K = np.zeros((4, 4))
    def tie(i, j, k):
        K[i, i] += k; K[j, j] += k; K[i, j] -= k; K[j, i] -= k
    tie(0, 1, k_loc); tie(2, 3, k_loc); tie(1, 2, k_tie)
    K[0, 0] += k_ext; K[3, 3] += k_ext
    return K


def full_state_matrix(M, D, K):
    n = len(M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -K / M[:, None]
    A[n:, n:] = -D / M[:, None]
    return A


def modes(A):
    lam, V = eig(A)
    idx = np.where(lam.imag > 1e-6)[0]
    lam, V = lam[idx], V[:, idx]
    o = np.argsort(lam.imag)
    lam, V = lam[o], V[:, o]
    f = lam.imag / (2 * np.pi)
    z = -lam.real / np.abs(lam)
    return f, z, V


def aggregate(M, D, K):
    """Coherent aggregation by area (area angle = area centre of inertia)."""
    P = np.zeros((2, 4)); P[0, AREA == 0] = 1; P[1, AREA == 1] = 1
    MA = P @ M
    DA = P @ D @ P.T            # intra-area relative damping cancels; tie damping survives
    Kr = np.array([[K_TIE + K_EXT, -K_TIE], [-K_TIE, K_TIE + K_EXT]])
    return MA, DA, Kr


def slow_basis(a=0.25, Tj=TJ_NOM):
    """Two slowest undamped mode shapes at the calibration point, rotated so that the
    columns are as close as possible (least squares) to area indicator vectors.
    Computed once; held fixed under every later parameter change."""
    from scipy.linalg import eigh
    M, _ = machine_params(a, Tj)
    _, V = eigh(stiffness(), np.diag(M))
    W = V[:, :2]
    P = np.zeros((2, 4)); P[0, AREA == 0] = 1; P[1, AREA == 1] = 1
    R = np.linalg.lstsq(W, P.T, rcond=None)[0]
    return W @ R


def reduce(M, D, K, Wa):
    """Galerkin reduction onto span(Wa): Mr, Dr, Kr (2x2, symmetric)."""
    Mm = np.diag(M) if np.ndim(M) == 1 else M
    return Wa.T @ Mm @ Wa, Wa.T @ D @ Wa, Wa.T @ K @ Wa


def msqrt(S):
    w, U = np.linalg.eigh(S)
    return (U * np.sqrt(w)) @ U.T


def operator_block(Mr, Dr, Kr):
    """Single-excitation generator Omega = (Mr^-1/2 Kr Mr^-1/2 - Gamma^2/4)^1/2 and damping
    matrix Gamma = Mr^-1/2 Dr Mr^-1/2 (both 2x2, real symmetric). Closed form; no fitting.
    The evolution psi' = -(i Omega + Gamma/2) psi is exact when [Omega, Gamma] = 0."""
    Mi = np.linalg.inv(msqrt(Mr))
    Gam = Mi @ Dr @ Mi
    Om = msqrt(Mi @ Kr @ Mi - 0.25 * Gam @ Gam)   # damped-frequency generator
    return Om, Gam


def block_modes(Om, gam):
    """Modes of the effective non-Hermitian block Omega - i Gamma/2."""
    lam = np.linalg.eigvals(Om - 0.5j * gam)
    o = np.argsort(lam.real)
    lam = lam[o]
    w = lam.real; f = w / (2 * np.pi); z = -lam.imag / np.sqrt(w ** 2 + lam.imag ** 2)
    # f is the damped frequency; zeta from |s| = natural frequency
    return f, z


def interarea(a=0.0, Tj=TJ_NOM, Dv=D_V, a2=None, dscale=1.0, ktie=1.0):
    M, D = machine_params(a, Tj, Dv, a2, dscale=dscale)
    K = stiffness(k_tie=K_TIE * ktie)
    f, z, _ = modes(full_state_matrix(M, D, K))
    # inter-area mode = lowest-frequency oscillatory mode of the full model
    return f, z


if __name__ == "__main__":
    for a in [0, 0.25, 0.5, 0.75, 1.0]:
        M, D = machine_params(a)
        K = stiffness()
        f, z, _ = modes(full_state_matrix(M, D, K))
        Om, g = operator_block(*aggregate(M, D, K))
        fb, zb = block_modes(Om, g)
        print(f"a={a:.2f} full f={np.round(f,3)} z={np.round(z,4)} | op f={np.round(fb,3)} z={np.round(zb,4)}")
