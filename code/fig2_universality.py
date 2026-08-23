#!/usr/bin/env python3
"""Figure 2 — width universality across 7 models (m5w artifacts, E4).

(a) Fraction of the joint-ablation effect realized vs fraction of writers ablated,
    7 models, matched controls in grey. Width-1 points circled.
(b) The predictor: writer-ladder cliff ratio (top1/top2 prev-token score — a
    representation-side quantity, no ablation needed) vs the fraction of the
    joint effect the single-head test realizes. The two "single-head works"
    models are exactly the two with a cliff.

Palette validated (dataviz six checks): #2a78d6 #eb6834 #1a8f6a #8b3fb0 + greys.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROOT = "/path/to/Documents/workspace/gpt2-mech-interp/results"
OUT = "/path/to/Documents/workspace/interp4discovery-neurips2026/paper/figures"

INK, MUTED, GRID, SURFACE = "#1b1a17", "#55534d", "#d8d6d1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.4,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.labelsize": 6.9, "ytick.labelsize": 6.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

MODELS = [
    ("gpt2", "GPT-2 s", "#0f3d73", "o"),
    ("gpt2-medium", "GPT-2 m", "#2a78d6", "o"),
    ("gpt2-large", "GPT-2 l", "#5b93d6", "o"),
    ("gpt2-xl", "GPT-2 xl", "#7ea6c9", "o"),
    ("EleutherAI_pythia-70m", "Pythia 70m", "#7a2f0f", "s"),
    ("EleutherAI_pythia-160m", "Pythia 160m", "#a03f16", "s"),
    ("EleutherAI_pythia-410m", "Pythia 410m", "#c94f1d", "s"),
    ("EleutherAI_pythia-1b", "Pythia 1b", "#e37438", "s"),
    ("EleutherAI_pythia-1.4b", "Pythia 1.4b", "#ef8f5e", "s"),
    ("Qwen_Qwen2-0.5B", "Qwen2 0.5B", "#6d31a5", "D"),
    ("Qwen_Qwen2-1.5B", "Qwen2 1.5B", "#9b59c9", "D"),
    ("meta-llama_Llama-3.2-1B", "Llama-3.2 1B", "#1a8f6a", "^"),
]
import os
for _extra, _lab, _col, _mk in [
    ("EleutherAI_pythia-2.8b", "Pythia 2.8b", "#f4b48e", "s"),
    ("meta-llama_Llama-3.2-3B", "Llama-3.2 3B", "#57b592", "^"),
]:
    if os.path.exists(f"{GROOT}/m5w_{_extra}.npz"):
        MODELS.append((_extra, _lab, _col, _mk))

rows = []
for key, label, color, mk in MODELS:
    z = np.load(f"{GROOT}/m5w_{key}.npz", allow_pickle=True)
    base = float(z["base_score"][0])
    t, c = z["width_treat"], z["width_ctrl"]
    n_w = len(t)
    drop_t, drop_c = base - t, base - c
    dmax = drop_t[-1]
    prev = z["prev_token_scores"]
    lad = np.sort(np.nan_to_num(prev).ravel())[::-1]
    rows.append(dict(key=key, label=label, color=color, mk=mk, n_w=n_w,
                     x=np.arange(1, n_w + 1) / n_w,
                     y=drop_t / dmax, yc=drop_c / dmax,
                     frac1=float(drop_t[0] / dmax),
                     cliff=float(lad[0] / lad[1])))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.9, 2.55),
                               gridspec_kw=dict(width_ratios=[1.45, 1.0], wspace=0.32))

# ---- (a) width curves (models with a single writer have no curve) ----
curves = [r for r in rows if r["n_w"] > 1]
for r in curves:
    axA.plot(r["x"], r["yc"], color=GRID, lw=0.9, zorder=2)
for r in curves:
    axA.plot(r["x"], r["y"], color=r["color"], lw=1.7, marker=r["mk"], ms=3.6,
             mew=0, zorder=4, label=f"{r['label']}  (n={r['n_w']})")
    axA.plot(r["x"][0], r["y"][0], marker=r["mk"], ms=6.2, mfc=SURFACE,
             mec=r["color"], mew=1.4, zorder=6)
axA.set_xlabel("fraction of writers ablated  $k/n_w$", labelpad=2)
axA.set_ylabel("fraction of joint effect realized", labelpad=3)
axA.set_xlim(0, 1.03)
axA.set_ylim(-0.06, 1.06)
axA.grid(axis="y", color=GRID, lw=0.5, zorder=0)
axA.set_axisbelow(True)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
leg = axA.legend(loc="upper left", ncol=2, columnspacing=0.8, fontsize=5.2, frameon=True, handlelength=1.6,
                 labelspacing=0.26, borderpad=0.35, bbox_to_anchor=(-0.01, 1.03),
                 facecolor=SURFACE, edgecolor="none", framealpha=0.92)
for t_ in leg.get_texts():
    t_.set_color(INK)
leg.set_zorder(7)
axA.annotate("matched controls", xy=(0.60, 0.085), color=MUTED, fontsize=6.4)
axA.set_title("(a) writer-ablation width curves", fontsize=7.6,
              loc="left", pad=4, color=INK)

# ---- (b) cliff ratio predicts single-head success ----
from scipy.stats import spearmanr
for r in rows:
    axB.plot(r["cliff"], r["frac1"], marker=r["mk"], ms=6.0, mfc=r["color"],
             mec=SURFACE, mew=0.8, zorder=4)
# label only the informative extremes
LAB = {"gpt2": (-0.05, 0.06, "right"),
       "EleutherAI_pythia-70m": (-0.05, -0.06, "right"),
       "EleutherAI_pythia-410m": (0.05, 0.05, "left"),
       "Qwen_Qwen2-1.5B": (0.05, -0.02, "left"),
       "EleutherAI_pythia-160m": (0.05, -0.06, "left")}
for r in rows:
    if r["key"] in LAB:
        dx, dy, ha = LAB[r["key"]]
        axB.annotate(r["label"], xy=(r["cliff"], r["frac1"]),
                     xytext=(r["cliff"] + dx, r["frac1"] + dy),
                     fontsize=6.0, color=INK, ha=ha, va="center")
rho, pv = spearmanr([r["cliff"] for r in rows], [r["frac1"] for r in rows])
axB.annotate(f"Spearman $\\rho={rho:.2f}$\n$p={pv:.3f}$, $n={len(rows)}$",
             xy=(1.62, 0.30), fontsize=6.6, color=INK, linespacing=1.5)
axB.set_xlabel("writer-ladder cliff (top-1/top-2)", labelpad=2)
axB.set_ylabel("joint effect realized at $k{=}1$", labelpad=3)
axB.set_xlim(0.93, 2.55)
axB.set_ylim(-0.07, 1.09)
axB.grid(axis="y", color=GRID, lw=0.5, zorder=0)
axB.set_axisbelow(True)
for sp_ in ("top", "right"):
    axB.spines[sp_].set_visible(False)
axB.set_title("(b) the ladder cliff predicts single-unit tests", fontsize=7.6,
              loc="left", pad=4, color=INK)

fig.savefig(f"{OUT}/fig2.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}/fig2.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote fig2.pdf/png")
print(f"{'model':>14} {'n_w':>4} {'frac@k=1':>9} {'cliff':>6}")
for r in rows:
    print(f"{r['label']:>14} {r['n_w']:>4} {r['frac1']:>9.3f} {r['cliff']:>6.2f}")
sp = [(r["cliff"], r["frac1"]) for r in rows]
from scipy.stats import spearmanr
rho, p = spearmanr([a for a, _ in sp], [b for _, b in sp])
print(f"spearman(cliff, frac@1): rho={rho:.3f} p={p:.4f}")
