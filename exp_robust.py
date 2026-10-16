"""E4: robustness of the operator model to operating point, inertia, damping, inter-area
detuning, tie strength and the calibration (basis) choice; and identifier spread over noise seeds."""
import json, numpy as np
import model as m
import operator_model as om
import identifiers as ids
from common import T, FS, NOISE_MHZ, TO_MHZ, W_KICK, D_KICK, WA, full_response, rmse

rng = np.random.default_rng(7)
N = 300
rows = []
for i in range(N):
    p = dict(a=rng.uniform(0, 0.7), Tj=rng.uniform(4, 12), dscale=rng.uniform(0.5, 1.5),
             da2=rng.uniform(-0.2, 0.2), ktie=rng.uniform(0.7, 1.3))
    a2 = float(np.clip(p["a"] + p["da2"], 0, 1))
    M, D = m.machine_params(p["a"], p["Tj"], a2=a2, dscale=p["dscale"])
    K = m.stiffness(k_tie=m.K_TIE * p["ktie"])
    f, z, _ = m.modes(m.full_state_matrix(M, D, K))
    d, w = om.run_full(M, D, K, D_KICK, W_KICK, T); truth = om.coi(M, w)[1] * TO_MHZ
    for basis in ["nominal", "local"]:
        Wb = WA if basis == "nominal" else m.slow_basis.__wrapped__(M, K) if hasattr(m.slow_basis, "__wrapped__") else None
        if basis == "local":
            from scipy.linalg import eigh
            _, V = eigh(K, np.diag(M)); W = V[:, :2]
            P = np.zeros((2, 4)); P[0, :2] = 1; P[1, 2:] = 1
            Wb = W @ np.linalg.lstsq(W, P.T, rcond=None)[0]
        Mr, Dr, Kr = m.reduce(M, D, K, Wb)
        Om, G = m.operator_block(Mr, Dr, Kr)
        fb, zb = m.block_modes(Om, G)
        comm = np.linalg.norm(Om @ G - G @ Om, 2) / (np.linalg.norm(Om, 2) * np.linalg.norm(G, 2))
        try:
            _, xd, _ = om.run_operator(Mr, Dr, Kr, om.project(M, Wb, D_KICK), om.project(M, Wb, W_KICK), T)
            yop = om.coi(M, Wb @ xd)[1] * TO_MHZ
            _, xdc = om.run_reduced_classical(Mr, Dr, Kr, om.project(M, Wb, D_KICK), om.project(M, Wb, W_KICK), T)
            ycl = om.coi(M, Wb @ xdc)[1] * TO_MHZ
            ok = True
        except ValueError:
            ok = False; yop = ycl = np.full_like(truth, np.nan)
        peak = np.abs(truth).max()
        rows.append(dict(**p, basis=basis, f_full=f[1], z_full=z[1], f_op=fb[1], z_op=zb[1],
                         f_relerr=abs(fb[1] - f[1]) / f[1], z_abserr=abs(zb[1] - z[1]),
                         nrmse_op_full=rmse(yop, truth) / peak, nrmse_op_reduced=rmse(yop, ycl) / peak,
                         comm=comm, lindblad_ok=ok, gamma_min=float(np.linalg.eigvalsh(G).min())))

def summ(key, basis):
    v = np.array([r[key] for r in rows if r["basis"] == basis], float); v = v[~np.isnan(v)]
    return dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()), n=len(v))

out = {"N": N, "ranges": "a~U[0,0.7], Tj~U[4,12] s, damping scale~U[0.5,1.5], a2-a~U[-0.2,0.2], tie scale~U[0.7,1.3]"}
for b in ["nominal", "local"]:
    out[b] = {k: summ(k, b) for k in ["f_relerr", "z_abserr", "nrmse_op_full", "nrmse_op_reduced", "comm"]}
    rb = [r for r in rows if r["basis"] == b]
    out[b]["lindblad_realisable_frac"] = float(np.mean([r["lindblad_ok"] for r in rb]))
    c = np.array([r["comm"] for r in rb]); e = np.array([r["nrmse_op_reduced"] for r in rb])
    ok = ~np.isnan(e)
    out[b]["corr_comm_vs_transform_error"] = float(np.corrcoef(c[ok], e[ok])[0, 1])
out["rows"] = rows

# identifier spread over 100 noise realisations at the nominal point
y = full_response(0.25, 8.0)
f_true, z_true = m.interarea(0.25, 8.0)
spread = {}
for k, fn in ids.IDENTIFIERS.items():
    zs, fs_ = [], []
    for s in range(100):
        ym = y + np.random.default_rng(100 + s).normal(0, NOISE_MHZ, len(T))
        _, zz, rr = fn(ym, 1 / FS, 4)
        fm, zm = ids.dominant_mode(zz, rr, 1 / FS)
        fs_.append(fm); zs.append(zm)
    zs, fs_ = np.array(zs), np.array(fs_)
    spread[k] = dict(z_mean=float(np.nanmean(zs)), z_ci95=[float(np.nanpercentile(zs, 2.5)), float(np.nanpercentile(zs, 97.5))],
                     f_mean=float(np.nanmean(fs_)), f_ci95=[float(np.nanpercentile(fs_, 2.5)), float(np.nanpercentile(fs_, 97.5))],
                     fail=int(np.isnan(zs).sum()))
out["identifier_spread"] = spread
out["true_mode_nominal"] = dict(f=f_true[1], zeta=z_true[1], f_op=None)
fo, zo = m.block_modes(*m.operator_block(*m.reduce(*m.machine_params(0.25, 8.0), m.stiffness(), WA)))
out["true_mode_nominal"]["f_op"] = fo[1]; out["true_mode_nominal"]["z_op"] = zo[1]
json.dump(out, open("results/robustness.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
