"""Apply the evaluation to a MEASURED two-area PMU record (not used for any number in the
manuscript until such a record is supplied and cleared for release).

Input CSV columns: t (s), f1, f2 (area frequency, Hz or deviation in Hz), optional ang1, ang2 (deg).
Usage:  python3 run_measured.py record.csv --t0 <event time s> --len 15 [--ambient-from s --ambient-to s]

Outputs results/measured_<name>.json with:
  * inter-area mode (f, zeta) from four identifiers with 200-sample block-bootstrap 95% intervals;
  * operator-block reconstruction of the two-channel ring-down, with the generator fitted to the
    identified modes (two complex modes, mode shapes from the residues) -- reported separately
    from any physics-based prediction, which needs area inertia/tie data;
  * sliding-window M_ent / I_stab / variance / mode-meter tracks on an ambient segment.
"""
import argparse, json, os
import numpy as np
import identifiers as ids
import sys as _s
_argv = _s.argv; _s.argv = [_argv[0]]
import exp_ews as E
_s.argv = _argv


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"]; f1, f2 = d["f1"], d["f2"]
    base = 50.0 if abs(np.median(f1) - 50) < 1 else (60.0 if abs(np.median(f1) - 60) < 1 else 0.0)
    f1 = f1 - base; f2 = f2 - base
    ang = None
    if "ang1" in d.dtype.names:
        ang = np.deg2rad(np.column_stack([d["ang1"], d["ang2"]]))
    return t, np.column_stack([f1, f2]) * 1e3, ang      # mHz


def main():
    p = argparse.ArgumentParser(); p.add_argument("csv"); p.add_argument("--t0", type=float, required=True)
    p.add_argument("--len", type=float, default=15.0); p.add_argument("--ambient-from", type=float)
    p.add_argument("--ambient-to", type=float); a = p.parse_args()
    t, f, ang = load(a.csv); dt = float(np.median(np.diff(t)))
    sel = (t >= a.t0) & (t < a.t0 + a.len)
    y = f[sel, 0] - f[sel, 1]; y = y - y[-int(2 / dt):].mean()
    out = {"source": os.path.basename(a.csv), "fs": 1 / dt, "window": [a.t0, a.t0 + a.len]}
    rng = np.random.default_rng(0)
    for k, fn in ids.IDENTIFIERS.items():
        yh, z, r = fn(y, dt, 4); fm, zm = ids.dominant_mode(z, r, dt)
        res = y - yh; bs = []
        for _ in range(200):     # residual block bootstrap
            blk = int(1 / dt); idx = np.concatenate([np.arange(s, s + blk) for s in rng.integers(0, len(y) - blk, len(y) // blk + 1)])[:len(y)]
            _, zb, rb = fn(yh + res[idx], dt, 4); bs.append(ids.dominant_mode(zb, rb, dt))
        bs = np.array(bs, float)
        out[k] = dict(f=fm, zeta=zm, f_ci95=np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist(),
                      zeta_ci95=np.nanpercentile(bs[:, 1], [2.5, 97.5]).tolist(), rmse_mHz=float(np.sqrt(np.mean(res ** 2))))
    if a.ambient_from is not None and ang is not None:
        s2 = (t >= a.ambient_from) & (t < a.ambient_to)
        E.FS = 1 / dt
        # generator at the nominal calibration; replace by the system's own reduced model when available
        times, vals = E.track(ang[s2], 2 * np.pi * f[s2] / 1e3, np.full(s2.sum(), 0.25))
        out["ambient_tracks"] = dict(t=times.tolist(), **{k: v.tolist() for k, v in vals.items()})
    json.dump(out, open(f"results/measured_{os.path.splitext(os.path.basename(a.csv))[0]}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "ambient_tracks"}, indent=1, default=float))


if __name__ == "__main__":
    main()
