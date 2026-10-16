"""E2/E3: nominal reconstruction and post-change prediction against baselines.
Ground truth = full linearised four-machine model (simulated, noise-free).
'Measured' signals = truth + white PMU noise (std NOISE_MHZ), seed fixed."""
import json, time, sys
import numpy as np
import model as m
import identifiers as ids
import baselines_learned as bl
from common import (T, FS, NOISE_MHZ, TO_MHZ, W_KICK, D_KICK, full_response, operator_response,
                    reduced_classical_response, rmse)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RUN_LEARNED = "--nolearned" not in sys.argv
rng = np.random.default_rng(SEED)
dt = 1 / FS
ORDER = 4
ORDERS = {"Prony": 4, "Matrix Pencil": 4, "ERA": 4, "Hankel-DMD": 4}

SCEN = {
    "nominal": dict(pre=(0.25, 8.0), post=None),
    "S1_automation_step": dict(pre=(0.25, 8.0), post=(0.55, 8.0)),
    "S2_inertia_cut": dict(pre=(0.55, 8.0), post=(0.55, 4.8)),
}


def timed(f, *a, **k):
    t0 = time.perf_counter(); r = f(*a, **k); return r, time.perf_counter() - t0


def Afull(a, Tj):
    M, D = m.machine_params(a, Tj); return m.full_state_matrix(M, D, m.stiffness())


def coi2(M, x):
    return (M[2:] @ x[6:8]) / M[2:].sum()


out = {"seed": SEED, "orders": ORDERS, "noise_mHz": NOISE_MHZ, "fs": FS}
pinn_param = None
for name, sc in SCEN.items():
    a0, Tj0 = sc["pre"]
    y_pre = full_response(a0, Tj0)
    y_pre_m = y_pre + rng.normal(0, NOISE_MHZ, len(T))
    res = {"peak_pre_mHz": float(np.abs(y_pre).max())}
    if sc["post"] is None:     # ---------- reconstruction at nominal ----------
        res["Operator model (forward)"] = dict(rmse=rmse(operator_response(a0, Tj0), y_pre), role="forward simulation")
        res["Two-area classical (forward)"] = dict(rmse=rmse(reduced_classical_response(a0, Tj0), y_pre), role="forward simulation")
        for k, f in ids.IDENTIFIERS.items():
            yhat, z, r = f(y_pre_m, dt, ORDERS[k])
            fm, zm = ids.dominant_mode(z, r, dt)
            res[k] = dict(rmse=rmse(yhat, y_pre), role="fitted to the same record", f=fm, zeta=zm)
        res["noise floor (rmse of noise)"] = dict(rmse=rmse(y_pre_m, y_pre), role="reference")
        if RUN_LEARNED:
            (pred, el, l), tt = timed(bl.train_sfq, T, y_pre_m, 6, SEED)
            res["SFQ circuit (fit)"] = dict(rmse=rmse(pred(T), y_pre), role="fitted to the same record", train_s=el)
        f_full, z_full = m.interarea(a0, Tj0)
        res["true_mode"] = dict(f=f_full[1], zeta=z_full[1])
        out[name] = res
        continue
    a1, Tj1 = sc["post"]
    y_post = full_response(a1, Tj1)
    y_post_m = y_post + rng.normal(0, NOISE_MHZ, len(T))
    res["peak_post_mHz"] = float(np.abs(y_post).max())
    res["pre_post_difference"] = rmse(y_pre, y_post)
    # operator: algebraic update (time only the parameter update + generator assembly)
    def op_update():
        M, D = m.machine_params(a1, Tj1); Mr, Dr, Kr = m.reduce(M, D, m.stiffness(), __import__('common').WA)
        return m.operator_block(Mr, Dr, Kr)
    _, t_up = timed(op_update)
    reps = [timed(op_update)[1] for _ in range(200)]
    res["Operator model (algebraic update)"] = dict(rmse=rmse(operator_response(a1, Tj1), y_post),
        update_s=float(np.median(reps)), new_data=False, retrain=False)
    res["Two-area classical (re-linearised)"] = dict(rmse=rmse(reduced_classical_response(a1, Tj1), y_post),
        update_s=float(np.median(reps)), new_data=False, retrain=False)
    reps_full = [timed(lambda: np.linalg.eigvals(Afull(a1, Tj1)))[1] for _ in range(200)]
    res["Full model (re-linearised; = reference)"] = dict(rmse=0.0, update_s=float(np.median(reps_full)),
        new_data=False, retrain=False)
    for k, f in ids.IDENTIFIERS.items():
        yhat, z, r = f(y_pre_m, dt, ORDERS[k])
        res[k + " (pre-change identification held)"] = dict(rmse=rmse(yhat, y_post), new_data=True, retrain=False)
        (yhat2, _, _), t_id = timed(f, y_post_m, dt, ORDERS[k])
        res[k + " (re-identified on post-change record)"] = dict(rmse=rmse(yhat2, y_post), update_s=t_id,
            new_data=True, retrain=False, note="requires the post-change ring-down itself")
    if RUN_LEARNED:
        pred, el, _ = bl.train_sfq(T, y_pre_m, 6, SEED)
        res["SFQ circuit (trained pre-change, not retrained)"] = dict(rmse=rmse(pred(T), y_post), new_data=False, retrain=False)
        pred2, el2, _ = bl.train_sfq(T, y_post_m, 6, SEED)
        res["SFQ circuit (re-trained on post-change record)"] = dict(rmse=rmse(pred2(T), y_post), update_s=el2,
            new_data=True, retrain=True)
        M0, _ = m.machine_params(a0, Tj0); M1, _ = m.machine_params(a1, Tj1)
        x0 = np.r_[D_KICK, W_KICK]
        p_pre, e_pre, _ = bl.train_pinn_marching(lambda _: Afull(a0, Tj0), x0, seed=SEED)
        res["PINN (trained pre-change, not retrained)"] = dict(rmse=rmse(coi2(M1, p_pre(T)) * TO_MHZ, y_post), new_data=False, retrain=False,
            pre_rmse=rmse(coi2(M0, p_pre(T)) * TO_MHZ, y_pre))
        p_post, e_post, _ = bl.train_pinn_marching(lambda _: Afull(a1, Tj1), x0, seed=SEED)
        res["PINN (re-trained on post-change physics)"] = dict(rmse=rmse(coi2(M1, p_post(T)) * TO_MHZ, y_post), update_s=e_post,
            new_data=False, retrain=True)
        if pinn_param is None:
            pinn_param = bl.train_pinn_marching_param(lambda a: Afull(a, 8.0), lambda a: x0, (0.0, 0.6), seed=SEED)
            out["parametric_pinn_train_s"] = pinn_param[1]
        if Tj1 == 8.0:
            res["Parametric PINN (t, a), a in [0, 0.6]"] = dict(rmse=rmse(coi2(M1, pinn_param[0](T, a1)) * TO_MHZ, y_post),
                new_data=False, retrain=False, note="inside training range")
        else:
            res["Parametric PINN (t, a), a in [0, 0.6]"] = dict(rmse=rmse(coi2(M1, pinn_param[0](T, a1)) * TO_MHZ, y_post),
                new_data=False, retrain=False, note="Tj is not an input: change outside the trained family")
    out[name] = res
    print(name, "done", flush=True)

json.dump(out, open(f"results/transient_seed{SEED}.json", "w"), indent=1, default=float)
for name in SCEN:
    print("==", name)
    for k, v in out[name].items():
        if isinstance(v, dict):
            print(f"  {k:55s} rmse={v.get('rmse', float('nan')):.4f}  " + " ".join(f"{kk}={vv}" for kk, vv in v.items() if kk not in ('rmse',)))
        else:
            print(f"  {k}: {v}")
