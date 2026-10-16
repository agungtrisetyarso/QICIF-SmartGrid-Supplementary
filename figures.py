import json, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 600})
which = sys.argv[1:] or ["scaling", "ews", "robust", "transient"]


def save(fig, name):
    fig.savefig(f"figs/{name}.pdf", bbox_inches="tight"); fig.savefig(f"figs/{name}.png", bbox_inches="tight", dpi=300)


if "scaling" in which:
    d = json.load(open("results/scaling.json"))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    r = d["Tj_sweep_a1.0"]; Tj = [x["Tj"] for x in r]
    ax[0].plot(Tj, [x["f_full"] for x in r], "o", ms=5, color=C[0], label="Full model, eigenvalues")
    ax[0].plot(Tj, [x["f_op"] for x in r], "-", color=C[0], lw=1.2, label="Operator model (reduced generator)")
    ax[0].plot(Tj, [x["f_invTj"] for x in r], "--", color=C[1], label=r"$f\propto T_j^{-1}$ (both $C,L\propto T_j$)")
    ax[0].set_xlabel(r"Virtual-inertia time constant $T_j$ (s)"); ax[0].set_ylabel("Inter-area mode frequency (Hz)")
    ax[0].set_title("(a) Inertia scaling at full IBR share ($a=1$)", fontsize=8.5, loc="left")
    ax[0].legend(fontsize=7.5)
    s = d["a_sweep"]; a = [x["a"] for x in s]
    ax[1].plot(a, [100 * x["z_full"] for x in s], "o", ms=4, color=C[0], label="Full model, eigenvalues")
    ax[1].plot(a, [100 * x["z_op"] for x in s], "-", color=C[0], lw=1.2, label="Operator model")
    for zmin, ls in [(0.05, "--"), (0.0, ":")]:
        ax[1].axhline(100 * zmin, color=MUTED, ls=ls, lw=0.9)
        ast = d["a_star"][str(zmin)]["full"]
        ax[1].axvline(ast, color=MUTED, ls=ls, lw=0.9)
        ax[1].annotate(f"$a^*_{{{int(100*zmin)}\\%}}={ast:.2f}$", (ast, 100 * zmin), xytext=(4, 6),
                       textcoords="offset points", fontsize=7.5, color=INK)
    ax[1].set_xlabel("Automation share $a$"); ax[1].set_ylabel("Inter-area damping ratio (%)")
    ax[1].set_title(r"(b) Damping versus automation share ($T_j=8$ s)", fontsize=8.5, loc="left")
    ax[1].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(); save(fig, "fig2_scaling")

if "ews" in which:
    d = json.load(open("results/ews_symmetric.json"))
    ex = d["example"]; t = np.array(ex["t"])
    fig, ax = plt.subplots(4, 1, figsize=(7.0, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 1, 1]})
    dd = np.array(ex["d_mHz"]); td = np.arange(len(dd)) / 2.0
    w10 = 20; rms10 = np.sqrt(np.convolve(dd ** 2, np.ones(w10) / w10, mode="valid"))
    hit = np.where(rms10 > 5 * d["rms_nominal_mHz"])[0]; tvis = (hit[0] + w10) / 2.0
    TMAX = 1000.0
    ax[0].plot(td[td <= TMAX], dd[td <= TMAX], color=C[0], lw=0.5)
    lim = 1.1 * np.abs(dd[td <= TMAX]).max(); ax[0].set_ylim(-lim, lim); ax[0].set_xlim(0, TMAX)
    ax[0].set_ylabel("$\\Delta f_1-\\Delta f_2$ (mHz)")
    ax[0].set_title("(a) Simulated PMU signal; automation share ramps from 0.25", fontsize=8.5, loc="left")
    for axx in ax:
        axx.axvline(d["t_ref"], color=MUTED, ls="--", lw=0.8); axx.axvline(d["t_zero"], color=MUTED, ls=":", lw=0.8)
        axx.axvline(tvis, color=INK, ls="-", lw=0.7)
    ax[0].annotate("visible onset", (tvis, -0.85 * lim), xytext=(-62, 0), textcoords="offset points", fontsize=7.5, color=INK)
    ax[0].annotate("$\\zeta=5\\%$", (d["t_ref"], ax[0].get_ylim()[1] * 0.8), xytext=(3, 0), textcoords="offset points", fontsize=7.5)
    ax[0].annotate("$\\zeta=0$", (d["t_zero"], ax[0].get_ylim()[1] * 0.8), xytext=(3, 0), textcoords="offset points", fontsize=7.5)
    th = d["thresholds"]
    for i, (k, lab, sc) in enumerate([("Ment", "$M_{\\rm ent}$", 1), ("zeta_mm", "Mode-meter $\\hat\\zeta$ (%)", 100), ("var", "Variance (mHz$^2$)", 1)]):
        keep = t <= TMAX
        ax[i + 1].plot(t[keep], sc * np.array(ex[k])[keep], color=C[i], lw=1.2)
        ax[i + 1].axhline(sc * th[k], color=INK, lw=0.8, ls="-.")
        ax[i + 1].set_ylabel(lab)
        ax[i + 1].set_title(f"({'bcd'[i]}) " + {"Ment": "Entanglement margin from the window density operator",
                            "zeta_mm": "Ambient mode meter (Yule-Walker AR(30))",
                            "var": "Variance of the area-difference frequency"}[k] + "; dash-dot = alarm threshold",
                            fontsize=8.5, loc="left")
    ax[3].set_yscale("log")
    ax[1].set_ylim(0, 1.05)
    ax[3].set_xlabel("Time (s)")
    fig.tight_layout(); save(fig, "fig4_ews_example")

    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    labs = {"Ment": "$M_{\\rm ent}$", "zeta_mm": "Mode meter", "var": "Variance", "Istab": "$I_{\\rm stab}$", "ac1": "Lag-1 AC"}
    for j, var in enumerate(["symmetric", "asymmetric"]):
        try:
            dv = json.load(open(f"results/ews_{var}.json"))
        except FileNotFoundError:
            continue
        for i, k in enumerate(["Ment", "zeta_mm", "var"]):
            L = dv["lead"][k]
            if L["n"] == 0: continue
            y = i * 3 + j
            ax.errorbar(L["median"], y, xerr=[[L["median"] - L["ci95"][0]], [L["ci95"][1] - L["median"]]],
                        fmt="o" if j == 0 else "s", color=C[i], ms=5, capsize=2, lw=1)
            ax.annotate(f"{labs[k]} ({var}), miss {100*L['miss_rate']:.0f}%", (L["ci95"][1], y), xytext=(4, -3),
                        textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_yticks([]); ax.set_xlabel("Lead time before visible onset (s): median and 95% range over 200 runs")
    ax.set_xlim(0, 1000)
    fig.tight_layout(); save(fig, "fig5_ews_leads")

if "robust" in which:
    d = json.load(open("results/robustness.json"))
    rows = [r for r in d["rows"] if r["basis"] == "nominal" and r["lindblad_ok"]]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].scatter([r["comm"] for r in rows], [100 * r["nrmse_op_reduced"] for r in rows], s=10, color=C[0],
                  edgecolor="white", linewidth=0.4)
    ax[0].set_xlabel(r"$\|[\Omega,\Gamma]\|/(\|\Omega\|\,\|\Gamma\|)$"); ax[0].set_ylabel("Transform error, NRMSE (%)")
    ax[0].set_title("(a) Operator vs two-area classical", fontsize=8.5, loc="left")
    ax[1].scatter([abs(r["a"] - 0.25) + abs(r["Tj"] - 8) / 8 for r in rows], [100 * r["nrmse_op_full"] for r in rows], s=10,
                  color=C[1], edgecolor="white", linewidth=0.4)
    ax[1].set_xlabel(r"Distance from calibration point $|a-0.25|+|T_j-8|/8$")
    ax[1].set_ylabel("Error vs full model, NRMSE (%)")
    ax[1].set_title("(b) Operator vs four-machine model (basis fixed)", fontsize=8.5, loc="left")
    fig.tight_layout(); save(fig, "fig6_robustness")

if "transient" in which:
    import model as m
    from common import T, full_response, operator_response, NOISE_MHZ, FS
    import identifiers as ids
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    y0 = full_response(0.25, 8.0); y1 = full_response(0.55, 8.0)
    y0m = y0 + rng.normal(0, NOISE_MHZ, len(T))
    ax[0].plot(T, y0m, color=GRID, lw=0.8, label="Simulated PMU record (noisy)")
    ax[0].plot(T, y0, color=INK, lw=1.4, label="Four-machine model (truth)")
    ax[0].plot(T, operator_response(0.25, 8.0), "--", color=C[0], lw=1.4, label="Operator model")
    ax[0].set_title("(a) Nominal, $a=0.25$, $T_j=8$ s", fontsize=8.5, loc="left")
    yh, z, r = ids.matrix_pencil(y0m, 1 / FS, 4)
    ax[1].plot(T, y1, color=INK, lw=1.4, label="Four-machine model (truth)")
    ax[1].plot(T, operator_response(0.55, 8.0), "--", color=C[0], lw=1.4, label="Operator model, algebraic update")
    ax[1].plot(T, yh, ":", color=C[1], lw=1.4, label="Matrix Pencil, pre-change fit held")
    ax[1].set_title("(b) After automation step $a: 0.25\\to0.55$", fontsize=8.5, loc="left")
    for a_ in ax:
        a_.set_xlabel("Time (s)"); a_.legend(fontsize=7, loc="lower right")
    ax[0].set_ylabel("Area-2 frequency deviation (mHz)")
    fig.tight_layout(); save(fig, "fig3_transient")
