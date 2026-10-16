"""E5: early-warning test on simulated oscillation-onset events.

Ambient (stochastic) load excitation on the full four-machine model; the automation share
a(t) ramps slowly so that the inter-area damping crosses zero. Each indicator is computed on
a sliding window of simulated PMU data (area-COI angle and frequency, 30 frames/s, additive
measurement noise). Alarm thresholds are fixed on independent stationary runs at the
reference condition a_ref (inter-area zeta = 5%) for a 5 % per-run false-alarm target, and
evaluated on (i) further stationary runs at a_ref and at the nominal a = 0.25 (false alarms)
and (ii) ramp runs (detection lead before visible onset, misses)."""
import json, sys, time
import numpy as np
from scipy.linalg import solve_toeplitz
import model as m
from common import WA, TO_MHZ

FS = 30.0; DT = 0.005; DEC = int(round(1 / (FS * DT)))
WIN = 60.0; STEP = 5.0
SIG_P = 0.001          # ambient power-noise intensity (pu per sqrt(s)) on every machine
ANG_NOISE = np.deg2rad(0.01); FRQ_NOISE = 0.5e-3 * 2 * np.pi   # PMU noise: 0.01 deg, 0.5 mHz
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "symmetric"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
A2SHIFT = 0.10 if VARIANT == "asymmetric" else 0.0


def a2_of(a):
    return np.minimum(1.0, a + A2SHIFT)


def simulate(a_of_t, Tend, R, seed):
    """Symplectic Euler-Maruyama on the full model; parameters refreshed every 1 s."""
    rng = np.random.default_rng(seed)
    K = m.stiffness()
    x = np.zeros((R, 4)); v = np.zeros((R, 4))
    nsteps = int(round(Tend / DT)); refresh = int(round(1.0 / DT))
    nout = nsteps // DEC
    ang = np.empty((nout, R, 2)); frq = np.empty((nout, R, 2)); aout = np.empty(nout)
    P = np.zeros((2, 4)); P[0, m.AREA == 0] = 1; P[1, m.AREA == 1] = 1
    for k in range(nsteps):
        if k % refresh == 0:
            a = a_of_t(k * DT)
            M, D = m.machine_params(a, a2=a2_of(a))
            Mi = 1 / M; PM = (P * M) / (P @ M)[:, None]
        acc = (-(x @ K.T) - (v @ D.T)) * Mi
        v = v + DT * acc + (SIG_P * np.sqrt(DT)) * rng.standard_normal((R, 4)) * Mi
        x = x + DT * v
        if (k + 1) % DEC == 0:
            j = (k + 1) // DEC - 1
            ang[j] = x @ PM.T; frq[j] = v @ PM.T; aout[j] = a
    ang += ANG_NOISE * rng.standard_normal(ang.shape)
    frq += FRQ_NOISE * rng.standard_normal(frq.shape)
    return ang, frq, aout


def operator_generator(a):
    M, D = m.machine_params(a, a2=a2_of(a))
    Mr, Dr, Kr = m.reduce(M, D, m.stiffness(), WA)
    Om, G = m.operator_block(Mr, Dr, Kr)
    return Om, G, m.msqrt(Mr)


def window_state(ang, frq, a):
    """Window density operator on span{|10>,|01>} from measured area angles/frequencies.
    Area COI coordinates are used as the reduced coordinates (WA is close to area indicators)."""
    Om, G, S = operator_generator(a)
    q = ang @ S.T; qd = frq @ S.T
    psi = q @ Om.T + 1j * (qd + q @ (0.5 * G).T)
    rho = psi.T @ psi.conj() / len(psi)
    return rho / np.trace(rho).real


def indicators(ang, frq, a):
    rho = window_state(ang, frq, a)
    C = 2 * abs(rho[0, 1])
    p = np.clip(rho[0, 0].real, 1e-12, 1 - 1e-12)
    S = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    d = (frq[:, 0] - frq[:, 1]) * TO_MHZ
    d = d - d.mean()
    var = d.var()
    ds = d[::15]                    # 2 frames/s for the lag-1 autocorrelation
    ac1 = np.corrcoef(ds[:-1], ds[1:])[0, 1]
    zeta_mm = mode_meter(d[::3], 10.0)
    return dict(Ment=1 - C, Istab=1 - S / np.log(2), var=var, ac1=ac1, zeta_mm=zeta_mm)


def mode_meter(y, fs, p=30, band=(0.45, 1.1)):
    """Yule-Walker AR(p) ambient mode meter; damping of the dominant in-band pole."""
    y = y - y.mean(); n = len(y)
    r = np.array([y[:n - k] @ y[k:] for k in range(p + 1)]) / n
    c = solve_toeplitz(r[:p], r[1:p + 1])
    z = np.roots(np.r_[1, -c])
    s = np.log(z.astype(complex)) * fs
    f = s.imag / (2 * np.pi)
    sel = (f > band[0]) & (f < band[1])
    if not sel.any():
        return np.nan
    i = np.where(sel)[0][np.argmax(np.abs(z[sel]))]
    return -s[i].real / abs(s[i])


DIRECTION = dict(Ment=-1, Istab=-1, var=+1, ac1=+1, zeta_mm=-1)   # alarm when indicator moves this way


def track(ang, frq, aout):
    nw = int(WIN * FS); ns = int(STEP * FS)
    times, vals = [], {k: [] for k in DIRECTION}
    for e in range(nw, len(aout) + 1, ns):
        ind = indicators(ang[e - nw:e], frq[e - nw:e], aout[e - 1])
        for k in DIRECTION: vals[k].append(ind[k])
        times.append(e / FS)
    return np.array(times), {k: np.array(v) for k, v in vals.items()}


def a_at_zeta(zt):
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        _, z = m.interarea(mid, a2=a2_of(mid))
        (lo, hi) = (mid, hi) if z[1] > zt else (lo, mid)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    t0 = time.time()
    a_ref = a_at_zeta(0.05); a_zero = a_at_zeta(0.0)
    TNULL = 600.0
    A0, A1, TRAMP = 0.25, min(0.98, a_zero + 0.15), 1200.0
    ramp = lambda t: A0 + (A1 - A0) * min(t, TRAMP) / TRAMP
    out = dict(variant=VARIANT, N=N, a_ref=a_ref, a_zero=a_zero, ramp=[A0, A1, TRAMP], win=WIN, step=STEP,
               sig_p=SIG_P, seeds={})

    def run_set(a_fun, Tend, seed, R):
        res = []
        for b in range(0, R, 50):
            r = min(50, R - b)
            ang, frq, aout = simulate(a_fun, Tend, r, seed + b)
            for i in range(r):
                res.append(track(ang[:, i], frq[:, i], aout))
        return res, ang, frq

    cal, _, _ = run_set(lambda t: a_ref, TNULL, 1000, N)
    thr = {}
    for k, s in DIRECTION.items():
        ext = [np.nanmax(s * v[k]) for _, v in cal]
        thr[k] = float(np.nanpercentile(ext, 95)) * s          # value at which an alarm is raised
    out["thresholds"] = thr
    print("calibrated", time.time() - t0, flush=True)

    def alarms(sets):
        return {k: np.array([np.any(DIRECTION[k] * v[k] > DIRECTION[k] * thr[k]) for _, v in sets]) for k in DIRECTION}
    null_ref, _, _ = run_set(lambda t: a_ref, TNULL, 2000, N)
    null_nom, ang_nom, frq_nom = run_set(lambda t: A0, TNULL, 3000, N)
    out["far_at_a_ref"] = {k: float(v.mean()) for k, v in alarms(null_ref).items()}
    out["far_at_nominal"] = {k: float(v.mean()) for k, v in alarms(null_nom).items()}
    # visible-onset detector: 10 s RMS of the area-difference frequency > 5x nominal stationary RMS
    dnom = (frq_nom[:, :, 0] - frq_nom[:, :, 1]) * TO_MHZ
    rms_nom = float(np.sqrt(np.mean(dnom[int(60 * FS):] ** 2)))
    out["rms_nominal_mHz"] = rms_nom
    print("nulls", time.time() - t0, flush=True)

    leads = {k: [] for k in DIRECTION}; miss = {k: 0 for k in DIRECTION}; tvis_all = []
    t_ref = (a_ref - A0) / (A1 - A0) * TRAMP; t_zero = (a_zero - A0) / (A1 - A0) * TRAMP
    for b in range(0, N, 50):
        r = min(50, N - b)
        ang, frq, aout = simulate(ramp, TRAMP + 300, r, 4000 + b)
        for i in range(r):
            d = (frq[:, i, 0] - frq[:, i, 1]) * TO_MHZ
            w = int(10 * FS)
            rms10 = np.sqrt(np.convolve(d ** 2, np.ones(w) / w, mode="valid"))
            hit = np.where(rms10 > 5 * rms_nom)[0]
            tvis = (hit[0] + w) / FS if len(hit) else np.nan
            tvis_all.append(tvis)
            times, vals = track(ang[:, i], frq[:, i], aout)
            for k, s in DIRECTION.items():
                al = np.where(s * vals[k] > s * thr[k])[0]
                ta = times[al[0]] if len(al) else np.nan
                if np.isnan(ta) or (not np.isnan(tvis) and ta > tvis):
                    miss[k] += 1
                else:
                    leads[k].append(tvis - ta)
    tv = np.array(tvis_all)
    out["t_ref"] = t_ref; out["t_zero"] = t_zero
    out["t_visible"] = dict(mean=float(np.nanmean(tv)), sd=float(np.nanstd(tv)), n_nan=int(np.isnan(tv).sum()))
    out["lead"] = {k: dict(n=len(v), mean=float(np.mean(v)) if v else None,
                           median=float(np.median(v)) if v else None,
                           ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None,
                           miss_rate=miss[k] / N) for k, v in leads.items()}
    # one example trajectory of the indicators for the figure
    ang, frq, aout = simulate(ramp, TRAMP + 300, 1, 9999)
    times, vals = track(ang[:, 0], frq[:, 0], aout)
    out["example"] = dict(t=times.tolist(), a=[float(ramp(t)) for t in times],
                          **{k: v.tolist() for k, v in vals.items()},
                          d_mHz=((frq[::15, 0, 0] - frq[::15, 0, 1]) * TO_MHZ).tolist())
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(f"results/ews_{VARIANT}.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["a_ref", "a_zero", "t_ref", "t_zero", "t_visible", "thresholds", "far_at_a_ref", "far_at_nominal", "lead", "rms_nominal_mHz"]}, indent=1))
