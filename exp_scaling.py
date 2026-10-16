"""E1: inertia and automation-share scaling of the inter-area mode: full eigenvalues vs operator."""
import json, numpy as np
import model as m
from common import op_modes

out = {}
out["calibration_nominal"] = {"a": 0.25, "Tj": 8.0, "basis": "two slowest undamped mode shapes"}
op_mode = op_modes

# (a) Tj sweep at full IBR share (a=1) and at a=0.5
for a in [1.0, 0.5]:
    rows = []
    f8 = m.interarea(a, 8.0)[0][1]
    for Tj in [4.0, 4.8, 6.0, 8.0, 10.0, 12.0, 16.0]:
        f, z = m.interarea(a, Tj)
        fo, zo = op_mode(a, Tj)
        rows.append(dict(Tj=Tj, f_full=f[1], z_full=z[1], f_op=fo[1], z_op=zo[1],
                         f_invTj=f8 * 8.0 / Tj, f_invsqrtTj=f8 * np.sqrt(8.0 / Tj)))
    out[f"Tj_sweep_a{a}"] = rows

# (b) automation-share sweep at Tj=8
rows = []
for a in np.round(np.linspace(0, 1, 21), 3):
    f, z = m.interarea(a, 8.0); fo, zo = op_mode(a, 8.0)
    rows.append(dict(a=float(a), f_full=f[1], z_full=z[1], f_op=fo[1], z_op=zo[1],
                     f_ip_full=f[0], z_ip_full=z[0]))
out["a_sweep"] = rows

# (c) automation threshold a* for zeta_min in {0.05, 0.03, 0}: bisection, full vs operator
def thresh(fun, zmin):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if fun(mid) > zmin else (lo, mid)
    return 0.5 * (lo + hi)
th = {}
for zmin in [0.05, 0.03, 0.0]:
    th[str(zmin)] = dict(full=thresh(lambda a: m.interarea(a, 8.0)[1][1], zmin),
                         op=thresh(lambda a: op_mode(a, 8.0)[1][1], zmin))
out["a_star"] = th
# threshold vs Tj
out["a_star_vs_Tj"] = [dict(Tj=Tj, full=thresh(lambda a: m.interarea(a, Tj)[1][1], 0.05),
                            op=thresh(lambda a: op_mode(a, Tj)[1][1], 0.05)) for Tj in [4, 6, 8, 10, 12]]
json.dump(out, open("results/scaling.json", "w"), indent=1)
for k in ["Tj_sweep_a1.0", "Tj_sweep_a0.5"]:
    print(k)
    for r in out[k]:
        print({kk: round(v, 4) for kk, v in r.items()})
print(json.dumps(out["a_star"], indent=1)); print(out["a_star_vs_Tj"]); print(out["calibration_nominal"])
