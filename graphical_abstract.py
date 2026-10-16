import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
C = ["#2a78d6", "#eb6834", "#1baf7a"]; INK = "#1f1f1e"; MUTED = "#6b6a64"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(13.28, 5.31), dpi=300)
ax = fig.add_axes([0, 0, 0.46, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = [("Automation share $a$, virtual inertia $T_j$", "affine laws for inertia and damping"),
         ("Multi-machine swing model", "Galerkin reduction to two areas"),
         ("Complex amplitude $\\psi=\\Omega q+i(\\dot q+\\Gamma q/2)$", "exact if $[\\Omega,\\Gamma]=0$"),
         ("Two-qubit XY Hamiltonian + Lindblad", "one qubit per area; $\\Gamma\\succeq0$"),
         ("Window state $\\rho_w$ from PMU data", "margin $M_{\\rm ent}=1-\\mathcal{C}(\\rho_w)$")]
y = 0.9
for i, (a, b) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((0.05, y - 0.065), 0.9, 0.11, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#f3f6fb", ec=C[0], lw=1.2))
    ax.text(0.5, y + 0.012, a, ha="center", va="center", fontsize=12, color=INK)
    ax.text(0.5, y - 0.042, b, ha="center", va="center", fontsize=10, color=MUTED)
    if i < len(steps) - 1:
        ax.annotate("", (0.5, y - 0.135), (0.5, y - 0.07), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    y -= 0.19
sc = json.load(open("results/scaling.json")); ew = json.load(open("results/ews_symmetric.json"))
a1 = fig.add_axes([0.52, 0.17, 0.2, 0.68])
r = sc["Tj_sweep_a1.0"]; Tj = [x["Tj"] for x in r]
a1.plot(Tj, [x["f_full"] for x in r], "o", color=C[0], ms=6, label="eigenvalues")
a1.plot(Tj, [x["f_op"] for x in r], "-", color=C[0], label="operator model")
a1.plot(Tj, [x["f_invTj"] for x in r], "--", color=C[1], label="$f\\propto T_j^{-1}$")
a1.set_xlabel("$T_j$ (s)"); a1.set_ylabel("Inter-area frequency (Hz)"); a1.legend(frameon=False, fontsize=9.5)
a1.set_title("Inertia law $f\\propto T_j^{-1/2}$", fontsize=12, loc="left")
a2 = fig.add_axes([0.78, 0.17, 0.2, 0.68])
labs = [("Ment", "$M_{\\rm ent}$", C[0]), ("zeta_mm", "Mode meter", C[1]), ("var", "Variance", C[2])]
for i, (k, lab, c) in enumerate(labs):
    L = ew["lead"][k]
    a2.errorbar(L["median"], i, xerr=[[L["median"] - L["ci95"][0]], [L["ci95"][1] - L["median"]]], fmt="o", color=c, capsize=3, ms=7)
a2.set_yticks(range(3)); a2.set_yticklabels([l for _, l, _ in labs])
a2.set_xlabel("Lead before onset (s), 200 runs"); a2.set_xlim(0, 700)
a2.set_title("Early warning (simulated)", fontsize=12, loc="left")
for a_ in (a1, a2):
    a_.spines[["top", "right"]].set_visible(False); a_.grid(color="#e4e3de", lw=0.6)
fig.savefig("paper/graphical_abstract.png", dpi=300); fig.savefig("paper/graphical_abstract.pdf")
print("ok")
