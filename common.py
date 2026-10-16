import numpy as np
import model as m
import operator_model as om

FS = 30.0                       # PMU reporting rate (frames/s)
T_END = 15.0
T = np.arange(0, T_END + 1e-9, 1 / FS)
W_KICK = np.array([0, 0, -0.1, -0.1])   # rad/s initial speed deviation of G3, G4 (area-2 event)
D_KICK = np.zeros(4)
NOISE_MHZ = 0.5                 # additive PMU frequency noise (std)
TO_MHZ = 1e3 / (2 * np.pi)
A_NOM, TJ_NOM = 0.25, 8.0
WA = m.slow_basis(A_NOM, TJ_NOM)   # the single calibration: nominal slow-mode basis


def full_response(a, Tj=TJ_NOM, **kw):
    M, D = m.machine_params(a, Tj, **kw); K = m.stiffness()
    d, w = om.run_full(M, D, K, D_KICK, W_KICK, T)
    return om.coi(M, w)[1] * TO_MHZ


def _red(a, Tj, **kw):
    M, D = m.machine_params(a, Tj, **kw); K = m.stiffness()
    Mr, Dr, Kr = m.reduce(M, D, K, WA)
    return M, Mr, Dr, Kr


def operator_response(a, Tj=TJ_NOM, **kw):
    M, Mr, Dr, Kr = _red(a, Tj, **kw)
    _, xd, _ = om.run_operator(Mr, Dr, Kr, om.project(M, WA, D_KICK), om.project(M, WA, W_KICK), T)
    return om.coi(M, WA @ xd)[1] * TO_MHZ


def reduced_classical_response(a, Tj=TJ_NOM, **kw):
    M, Mr, Dr, Kr = _red(a, Tj, **kw)
    _, xd = om.run_reduced_classical(Mr, Dr, Kr, om.project(M, WA, D_KICK), om.project(M, WA, W_KICK), T)
    return om.coi(M, WA @ xd)[1] * TO_MHZ


def op_modes(a, Tj=TJ_NOM, **kw):
    _, Mr, Dr, Kr = _red(a, Tj, **kw)
    return m.block_modes(*m.operator_block(Mr, Dr, Kr))


def rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))
