#!/usr/bin/env python3
"""Figure 4 — the readout boundary as a curve (all data saved; re-plot only).

Two stacked panels, shared x = rank K (log scale), PaliGemma tower L24:
  (top)    the representation moves: rep_closed (VLM run, pg_subspace.json) and
           the within-tower probe (so400m_tower_steer.json) both rise to ~0.8.
  (bottom) the consumer ignores it: Delta p(target word) stays ~0 at every rank —
           for SIZE it moves the WRONG way — while the prompted baseline gap
           (p_tgt - p_src ~ +0.32) proves the frozen LM can read the attribute.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VROOT = "/path/to/Documents/workspace/vision-model-steering/results/metrics"
OUT = "/path/to/Documents/workspace/interp4discovery-neurips2026/paper/figures"

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

pg = json.load(open(f"{VROOT}/pg_subspace.json"))
tw = json.load(open(f"{VROOT}/so400m_tower_steer.json"))
pr = json.load(open(f"{VROOT}/exp2_prompted.json"))

K_pg = np.array(pg["size"]["ranks"], float)
K_tw = np.array(tw["ranks"] if "ranks" in tw else tw["size"]["ranks"], float)

fig, (axT, axB) = plt.subplots(1, 2, figsize=(6.4, 1.95),
                               gridspec_kw=dict(wspace=0.30))

# ---------- top: the representation moves ----------
axT.axhline(1.0, color=GRID, lw=0.7)
axT.plot(K_pg, pg["size"]["rep_closed"], color=PURPLE, lw=1.7, marker="D", ms=3.4,
         mew=0, label="size · rep. gap closed (in VLM)")
axT.plot(K_tw, tw["size"]["closed"], color=PURPLE, lw=1.2, ls=(0, (4, 2)),
         marker="D", ms=2.8, mew=0, alpha=0.65,
         label="size · within-tower probe")
axT.plot(K_pg, np.clip(pg["orientation"]["rep_closed"], None, 1.25), color=GREEN,
         lw=1.7, marker="^", ms=3.6, mew=0, label="orientation · rep. gap closed")
axT.set_ylabel("representation moved", labelpad=3)
axT.set_xscale("log")
axT.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
axT.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
axT.tick_params(axis="x", which="minor", bottom=False)
axT.set_xlabel("intervention rank $K$ (log)", labelpad=2)
axT.set_ylim(-0.05, 1.3)
axT.set_yticks([0, 0.5, 1.0])
axT.grid(axis="y", color=GRID, lw=0.5)
axT.set_axisbelow(True)
for sp in ("top", "right"):
    axT.spines[sp].set_visible(False)
leg = axT.legend(loc="lower right", fontsize=5.9, frameon=True, facecolor=SURFACE,
                 edgecolor="none", framealpha=0.9, handlelength=1.8,
                 labelspacing=0.3, borderpad=0.25)
for t in leg.get_texts():
    t.set_color(INK)
axT.set_title("(i) the edit lands", fontsize=7.4, loc="left", pad=4, color=INK)

# ---------- bottom: the consumer does not ----------
p0_s = pg["size"]["p_base"]
p0_o = pg["orientation"]["p_base"]
dp_s = np.array(pg["size"]["p_target"]) - p0_s
dp_o = np.array(pg["orientation"]["p_target"]) - p0_o
gap_prompted = pr["size"]["p_tgt_base"] - pr["size"]["p_src_base"]

axB.axhline(0.0, color=GRID, lw=0.7)
axB.axhspan(gap_prompted - 0.008, gap_prompted + 0.008, color="#efe7d9", lw=0)
axB.plot(K_pg, dp_s, color=PURPLE, lw=1.7, marker="D", ms=3.4, mew=0)
axB.plot(K_pg, dp_o, color=GREEN, lw=1.7, marker="^", ms=3.6, mew=0)
axB.annotate("what the LM reports when simply ASKED\n(prompted src→tgt gap, no steering)",
             xy=(1.15, gap_prompted - 0.045), fontsize=6.3, color="#7a6a4a",
             va="top", linespacing=1.35)
axB.annotate("size: wrong direction", xy=(32, dp_s[5]), xytext=(10.5, -0.115),
             fontsize=6.4, color=PURPLE,
             arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.6, shrinkA=1, shrinkB=2))
axB.annotate("orientation: flat", xy=(64, dp_o[6]), xytext=(24, 0.075),
             fontsize=6.4, color=GREEN,
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6, shrinkA=1, shrinkB=2))
axB.set_ylabel("Δ p(target word)", labelpad=3)
axB.set_xlabel("intervention rank $K$ (log)", labelpad=2)
axB.set_xscale("log")
axB.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
axB.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
axB.set_ylim(-0.16, 0.42)
axB.grid(axis="y", color=GRID, lw=0.5)
axB.set_axisbelow(True)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.tick_params(axis="x", which="minor", bottom=False)
axB.set_title("(ii) the reader ignores it", fontsize=7.4, loc="left", pad=4,
              color=INK)

fig.savefig(f"{OUT}/fig4.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}/fig4.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote fig4")
print(f"size: rep_closed max {max(pg['size']['rep_closed']):.3f}, "
      f"dp range [{dp_s.min():+.3f}, {dp_s.max():+.3f}], prompted gap {gap_prompted:+.3f}")
