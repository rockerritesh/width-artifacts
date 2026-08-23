#!/usr/bin/env python3
"""Figure 3 — width is the operative variable (not magnitude, not contiguity),
and over-widening is also a failure.

(a) Superadditivity: KL(band k) / sum KL(singles in band), 100 queries, Llama-8B
    Instruct. Additive prediction = 1. Crosses 7.5x at the headline k=12.
(b) Random k-subset (region-matched, 5 seeds) vs contiguous band vs strided
    control, base 8B on NDIF: damage tracks WIDTH, not contiguity/position.
(c) Over-widening destroys identity: id_kept vs K/k*, size attribute, three
    measurements (SigLIP-base L11, so400m standalone, so400m inside PaliGemma).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/path/to/Documents/workspace"
RES = f"{ROOT}/interp4discovery-neurips2026/results"
VM = f"{ROOT}/vision-model-steering/results/metrics"
OUT = f"{ROOT}/interp4discovery-neurips2026/paper/figures"

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1a8f6a", "#8b3fb0"
INK, MUTED, GRID, SURFACE = "#1b1a17", "#55534d", "#d8d6d1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.4,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.labelsize": 6.9, "ytick.labelsize": 6.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(6.9, 2.25),
                                    gridspec_kw=dict(wspace=0.42))

# ---------------- (a) superadditivity ----------------
sa = json.load(open(f"{RES}/superadditivity.json"))
ks = [k for k in sa["ks"] if k >= 1]
ratio = [sa["per_k"][str(k)]["ratio_of_means"] for k in ks]
axA.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
axA.plot(ks, ratio, color=ORANGE, lw=1.7, marker="s", ms=3.8, mew=0, zorder=4)
axA.annotate("additive prediction", xy=(7.6, 1.28), fontsize=6.4, color=MUTED)
axA.annotate(f"{ratio[-2]:.1f}× at k=12", xy=(12, ratio[-2]),
             xytext=(6.4, ratio[-2] + 0.35), fontsize=6.6, color=INK,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                             shrinkA=1, shrinkB=2))
axA.set_xlabel("band width $k$ (layers)", labelpad=2)
axA.set_ylabel("KL(band) / $\\Sigma$ KL(singles)", labelpad=3)
axA.set_xticks([1, 4, 8, 12])
axA.set_ylim(0, 9.0)
axA.grid(axis="y", color=GRID, lw=0.5)
axA.set_axisbelow(True)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
axA.set_title("(a) the band is not the\n     sum of its layers", fontsize=7.6,
              loc="left", pad=4, color=INK)

# ---------------- (b) width, not contiguity ----------------
e5 = json.load(open(f"{RES}/e5_random_vs_contiguous.json"))
KS = [2, 4, 6, 8, 10, 12, 14]
c_kl = [next(r["kl"] for r in e5["contiguous"] if r["k"] == k) for k in KS]
r_mean, r_sd = [], []
for k in KS:
    v = np.array([r["kl"] for r in e5["random_region"] if r["k"] == k])
    r_mean.append(v.mean()); r_sd.append(v.std())
st = e5["strided"][0]

axB.plot(KS, c_kl, color=BLUE, lw=1.7, marker="o", ms=3.8, mew=0, zorder=4,
         label="contiguous band")
axB.errorbar(KS, r_mean, yerr=r_sd, color=GREEN, lw=1.7, marker="^", ms=3.8,
             mew=0, capsize=2, elinewidth=0.8, zorder=4,
             label="random $k$-subset (5 seeds)")
axB.plot([st["k"]], [st["kl"]], marker="D", ms=5.5, mfc=PURPLE, mec=SURFACE,
         mew=0.8, zorder=5, ls="none", label="strided: span 12, width 6")
axB.annotate("width 6\nspan 12", xy=(st["k"], st["kl"]),
             xytext=(st["k"] + 1.1, st["kl"] + 0.75), fontsize=6.2,
             color=PURPLE, va="center", linespacing=1.3,
             arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.6,
                             shrinkA=1, shrinkB=2))
axB.set_xlabel("layers removed $k$", labelpad=2)
axB.set_ylabel("KL from intact model", labelpad=3)
axB.set_xticks([2, 6, 10, 14])
axB.grid(axis="y", color=GRID, lw=0.5)
axB.set_axisbelow(True)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
leg = axB.legend(loc="upper left", fontsize=6.0, frameon=False, handlelength=1.6,
                 labelspacing=0.3, borderpad=0.0)
for t in leg.get_texts():
    t.set_color(INK)
axB.set_title("(b) damage tracks width,\n     not contiguity", fontsize=7.6,
              loc="left", pad=4, color=INK)

# ---------------- (c) over-widening destroys identity ----------------
sl = json.load(open(f"{VM}/subspace_steer_all.json"))
sg = sl["models"]["siglip-base"]["size"]
tw = json.load(open(f"{VM}/so400m_tower_steer.json"))
e3 = json.load(open(f"{VM}/siglip400m_steer.json"))

for d, lab, col, mk in [
    (dict(ranks=sl["ranks"], **{k2: sg[k2] for k2 in ("kstar", "id_kept")}),
     "SigLIP-base L11", ORANGE, "s"),
    (dict(ranks=tw["ranks"] if "ranks" in tw else tw["size"]["ranks"],
          kstar=tw["size"]["kstar"], id_kept=tw["size"]["id_kept"]),
     "so400m in PaliGemma", BLUE, "o"),
    (dict(ranks=e3["size"]["ranks"] if "ranks" in e3["size"] else [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
          kstar=e3["size"]["kstar"], id_kept=e3["size"]["id_kept"]),
     "so400m standalone", GREEN, "^"),
]:
    x = np.array(d["ranks"], float) / d["kstar"]
    axC.plot(x, d["id_kept"], color=col, lw=1.7, marker=mk, ms=3.6, mew=0,
             zorder=4, label=lab)
axC.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
axC.text(1.12, 0.97, "$K = k^*$", color=MUTED, fontsize=6.4, va="top")
axC.set_xscale("log")
axC.set_xticks([0.1, 0.3, 1, 3, 7])
axC.set_xticklabels(["0.1", "0.3", "1", "3", "7"])
axC.tick_params(axis="x", which="minor", bottom=False)
axC.set_xlabel("intervention width $K/k^*$ (log)", labelpad=2)
axC.set_ylabel("object identity kept", labelpad=3)
axC.set_ylim(-0.04, 1.06)
axC.grid(axis="y", color=GRID, lw=0.5)
axC.set_axisbelow(True)
for sp in ("top", "right"):
    axC.spines[sp].set_visible(False)
leg = axC.legend(loc="lower left", fontsize=6.0, frameon=False, handlelength=1.6,
                 labelspacing=0.3, borderpad=0.0)
for t in leg.get_texts():
    t.set_color(INK)
axC.set_title("(c) over-widening destroys\n     identity (size)", fontsize=7.6,
              loc="left", pad=4, color=INK)

fig.savefig(f"{OUT}/fig3.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}/fig3.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote fig3")
