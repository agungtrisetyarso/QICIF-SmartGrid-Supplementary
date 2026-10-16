"""E5b: false alarms when the ambient excitation level differs from the calibration level.
Thresholds are those fixed in exp_ews.py (symmetric case); no recalibration."""
import json, sys
import numpy as np
sys.argv = [sys.argv[0], "symmetric"]
import exp_ews as E

base = json.load(open("results/ews_symmetric.json"))
thr = base["thresholds"]; a_ref = base["a_ref"]
out = {"thresholds_from": "results/ews_symmetric.json"}
for scale in [0.5, 2.0]:
    E.SIG_P = 0.001 * scale
    for label, a in [("a_ref", a_ref), ("nominal", 0.25)]:
        sets = []
        for b in range(0, 200, 50):
            ang, frq, aout = E.simulate(lambda t: a, 600.0, 50, 7000 + int(scale * 100) + b + (0 if label == "nominal" else 500))
            sets += [E.track(ang[:, i], frq[:, i], aout) for i in range(50)]
        far = {k: float(np.mean([np.any(E.DIRECTION[k] * v[k] > E.DIRECTION[k] * thr[k]) for _, v in sets]))
               for k in E.DIRECTION}
        out[f"excitation_x{scale}_{label}"] = far
        print(scale, label, far, flush=True)
json.dump(out, open("results/ews_excitation_shift.json", "w"), indent=1)
