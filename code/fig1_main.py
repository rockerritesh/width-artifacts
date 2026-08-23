#!/usr/bin/env python3
"""Figure 1 (MAIN/hero) — the whole argument in one image.

Top strip (schematic): a mechanism is a bundle of k* units.
  (i)   width-1 intervention -> ~10% of the effect  (the field's standard test)
  (ii)  width-k* intervention -> ~95%
  (iii) readout-limited: the bundle moves, the reader is not connected -> nothing,
        at ANY width.
Bottom left (data): three granularities collapse on the normalized width axis.
Bottom right (data): PaliGemma — representation gap closes to 81% while the
  readout moves ~0 (as fraction of the prompted-readout gap), at every rank.

Palette #2a78d6/#eb6834/#1a8f6a/#8b3fb0 — validated (dataviz six checks, light).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = "/path/to/Documents/workspace"
VM = f"{ROOT}/vision-model-steering/results/metrics"
NLP = f"{ROOT}/nlp-remove-jspace-layer/blog/figures"
G = f"{ROOT}/gpt2-mech-interp/results"
OUT = f"{ROOT}/interp4discovery-neurips2026/paper/figures"

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1a8f6a", "#8b3fb0"
INK, MUTED, GRID, SURFACE = "#1b1a17", "#55534d", "#d8d6d1", "#fcfcfb"
STRAND, PINCH = "#b9b5ac", "#c0392b"
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.4,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

fig = plt.figure(figsize=(6.9, 3.35))
gs = fig.add_gridspec(2, 7, height_ratios=[0.62, 1.22], hspace=0.38,
                      width_ratios=[1, 1, 1, 0.28, 1, 1, 1],
                      left=0.075, right=0.985, top=0.985, bottom=0.115)
axS = [fig.add_subplot(g) for g in
       (gs[0, 0:2], gs[0, 2:4], gs[0, 4:7])]
axA = fig.add_subplot(gs[1, 0:3])
axB = fig.add_subplot(gs[1, 4:7])

# ================= top strip: the bundle schematic =================
N_STRANDS = 7


def bundle(ax, pinch_n, connected, fill, label, sub):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ys = np.linspace(0.30, 0.88, N_STRANDS)
    x0, x1 = 0.04, 0.60
    # reader box
    bx = FancyBboxPatch((0.76, 0.34), 0.17, 0.50, boxstyle="round,pad=0.012",
                        fc="#eceae5", ec=MUTED, lw=0.8)
    ax.add_patch(bx)
    ax.text(0.845, 0.59, "reader", ha="center", va="center", fontsize=6.3,
            color=INK, rotation=90)
    # strands
    xe = 0.755 if connected else 0.655
    for y in ys:
        ax.plot([x0, xe], [y, y], color=STRAND, lw=1.4, solid_capstyle="round")
    if not connected:
        for y in ys[::2]:
            ax.plot([0.685, 0.72], [y, y], color=GRID, lw=0.9, ls=(0, (1.4, 1.6)))
        ax.text(0.702, 0.16, "not read", ha="center", fontsize=5.8, color=PINCH)
    # the intervention: a pinch bar across the top `pinch_n` strands
    top, bot = ys[-1], ys[-pinch_n]
    ax.add_patch(Rectangle((0.30, bot - 0.055), 0.045, (top - bot) + 0.11,
                           fc=PINCH, ec="none", alpha=0.85))
    # effect meter
    ax.add_patch(Rectangle((0.955, 0.30), 0.038, 0.58, fc="none", ec=MUTED, lw=0.7))
    if fill > 0:
        ax.add_patch(Rectangle((0.955, 0.30), 0.038, 0.58 * fill,
                               fc=GREEN if fill > 0.5 else ORANGE, ec="none"))
    ax.text(0.974, 0.20, f"{int(round(fill * 100))}%", ha="center", fontsize=6.0,
            color=INK)
    ax.text(0.32, 1.00, label, ha="center", va="top", fontsize=7.0, color=INK)
    ax.text(0.32, 0.055, sub, ha="center", va="center", fontsize=6.1, color=MUTED)


bundle(axS[0], 1, True, 0.10, "intervene on 1 unit", "1 direction · 1 layer · 1 head")
bundle(axS[1], N_STRANDS, True, 0.95, "intervene on $k^*$ units", "width matches the mechanism")
bundle(axS[2], N_STRANDS, False, 0.02, "readout-limited", "no width helps — nothing is reading")
axS[0].text(0.02, 1.22, "a mechanism is $k^*$ units wide:", fontsize=7.4,
            color=INK, style="italic", va="top", transform=axS[0].transAxes)

# ================= bottom left: the collapse =================
TAU = 0.95


def kstar_eff(w, e):
    w, e = np.asarray(w, float), np.asarray(e, float)
    i = np.nonzero(e >= TAU * e.max())[0][0]
    if i == 0:
        return w[0]
    return w[i - 1] + (TAU * e.max() - e[i - 1]) * (w[i] - w[i - 1]) / (e[i] - e[i - 1])


v = json.load(open(f"{VM}/subspace_steer_dinov2.json"))
m = json.load(open(f"{NLP}/metrics_100.json"))
cum = [c for c in m["cumulative"] if c["k"] > 0]
w_lay = np.array([c["k"] for c in cum], float)
max_lay = 1.0 - next(s for s in m["sweep"] if s["layer"] == 0)["top1_mean"]
e_lay = (1.0 - np.array([c["top1_mean"] for c in cum])) / max_lay
z = np.load(f"{G}/r1b_lattice_greedy.npz", allow_pickle=True)
frac = (z["coalition_value"] - z["v_empty"][0]) / (z["v_full"][0] - z["v_empty"][0])
sz = z["coalition_size"]
w_h = np.arange(1, sz.max() + 1, dtype=float)
e_h = np.array([frac[sz == s].max() for s in w_h])

SER = [("GPT-2 · heads", w_h, e_h, BLUE, "o"),
       ("Llama-8B · layers", w_lay, e_lay, ORANGE, "s"),
       ("DINOv2 · orientation", np.array(v["ranks"], float),
        np.array(v["orientation"]["closed"], float), GREEN, "^")]

axA.axhspan(0.08, 0.12, color="#eeede9", zorder=0, lw=0)
axA.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
axA.axhline(TAU, color=GRID, lw=0.7)
for lab, w, e, c, mk in SER:
    ks = kstar_eff(w, e)
    x = w / ks
    keep = x <= 1.75
    axA.plot(x[keep], e[keep], color=c, lw=1.7, marker=mk, ms=3.8, mew=0,
             zorder=4, label=f"{lab} ($k^*{{=}}{ks:.0f}$)")
    axA.plot(x[0], e[0], marker=mk, ms=6.2, mfc=SURFACE, mec=c, mew=1.5, zorder=6)
axA.set_xlim(-0.04, 1.78)
axA.set_ylim(-0.03, 1.13)
axA.set_yticks([0, 0.25, 0.5, 0.75, 0.95])
axA.set_yticklabels(["0", ".25", ".50", ".75", ".95"])
axA.set_xlabel("intervention width / $k^*$", labelpad=2)
axA.set_ylabel("fraction of achievable effect", labelpad=3)
axA.grid(axis="y", color=GRID, lw=0.5)
axA.set_axisbelow(True)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
leg = axA.legend(loc="lower left", fontsize=5.8, frameon=False, handlelength=1.2,
                 handletextpad=0.5, labelspacing=0.32, borderpad=0.0,
                 bbox_to_anchor=(0.575, 0.03))
for t in leg.get_texts():
    t.set_color(INK)
axA.annotate("standard practice: $\\approx$10% of the effect",
             xy=(0.15, 0.095), xytext=(0.24, 0.015), fontsize=6.4, color=INK,
             va="center",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=2, shrinkB=2))
axA.set_title("(a) width sweeps at three granularities", fontsize=7.6, loc="left",
              pad=4, color=INK)

# ================= bottom right: the readout boundary =================
pg = json.load(open(f"{VM}/pg_subspace.json"))
pr = json.load(open(f"{VM}/exp2_prompted.json"))
K = np.array(pg["size"]["ranks"], float)
rep = np.array(pg["size"]["rep_closed"], float)
gap = pr["size"]["p_tgt_base"] - pr["size"]["p_src_base"]
read = (np.array(pg["size"]["p_target"]) - pg["size"]["p_base"]) / gap

axB.axhline(0, color=GRID, lw=0.7)
axB.fill_between(K, read, rep, color="#f2e8e2", zorder=1)
axB.plot(K, rep, color=PURPLE, lw=1.8, marker="D", ms=3.6, mew=0, zorder=4)
axB.plot(K, read, color=INK, lw=1.8, marker="v", ms=3.6, mew=0, zorder=4)
axB.set_xscale("log")
axB.set_xticks([1, 4, 16, 64])
axB.set_xticklabels(["1", "4", "16", "64"])
axB.tick_params(axis="x", which="minor", bottom=False)
axB.set_xlabel("intervention rank $K$ (log)", labelpad=2)
axB.set_ylabel("fraction of gap moved", labelpad=3)
axB.set_ylim(-0.22, 1.13)
axB.grid(axis="y", color=GRID, lw=0.5)
axB.set_axisbelow(True)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.annotate("representation: 81% closed", xy=(16, 0.81), xytext=(2.6, 0.97),
             fontsize=6.5, color=PURPLE,
             arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.7, shrinkA=1, shrinkB=2))
axB.annotate("readout: $\\approx$0, wrong direction\n(yet it reads size when asked)",
             xy=(2.9, 0.015), xytext=(1.05, 0.36), fontsize=6.5, color=INK,
             va="center", linespacing=1.4,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=1, shrinkB=2))
axB.text(34, 0.44, "the readout\nboundary", fontsize=6.6, color="#a0644a",
         ha="center", style="italic", linespacing=1.4)
axB.set_title("(b) when no width helps — PaliGemma, size", fontsize=7.6,
              loc="left", pad=4, color=INK)

fig.savefig(f"{OUT}/fig1_main.pdf", pad_inches=0.02)
fig.savefig(f"{OUT}/fig1_main.png", dpi=300, pad_inches=0.02)
print("wrote fig1_main")
