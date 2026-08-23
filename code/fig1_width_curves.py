"""Figure 1 for the Interp4Discovery submission.

(a) Effect realised vs intervention width, normalised by each mechanism's own k*.
    Three modalities collapse onto one curve; `size` departs -- it has a
    representation-side k* (16 directions) but no effect-side k*, the signature
    of a readout limit rather than a width limit.
(b) The readout boundary: at PaliGemma's frozen LM the matched random control
    moves the answer MORE than the CAV does.

Two k* definitions, kept explicitly separate (this is the paper's diagnostic):
  k*_repr  -- width reaching 95% of the attribute signal in the representation,
              measured with NO intervention (vision only; from size_cav_summary).
  k*_eff   -- width reaching 95% of the maximum achievable *interventional*
              effect, from the width sweep itself.
Agreement => width-limited. k*_repr exists but k*_eff does not => readout-limited.

Palette #2a78d6,#eb6834,#1a8f6a,#8b3fb0: ALL CHECKS PASS under
dataviz/scripts/validate_palette.js --mode light (CVD WARN 6.4 green/orange,
licensed by secondary encoding: distinct markers + direct labels).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/path/to/Documents/workspace"
OUT = f"{ROOT}/interp4discovery-neurips2026/paper/figures"
TAU = 0.95
SATURATES = 0.90          # a mechanism is width-limited only if it gets this far

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1a8f6a", "#8b3fb0"
INK, MUTED, GRID = "#1b1a17", "#55534d", "#d8d6d1"
SURFACE, BAND = "#fcfcfb", "#eeede9"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.4,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.labelsize": 6.9, "ytick.labelsize": 6.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def kstar_eff(widths, eff):
    """Smallest width reaching TAU of max effect; None if it never saturates."""
    widths, eff = np.asarray(widths, float), np.asarray(eff, float)
    if eff.max() < SATURATES:
        return None
    i = np.nonzero(eff >= TAU * eff.max())[0][0]
    if i == 0:
        return widths[0]
    w0, w1, e0, e1 = widths[i - 1], widths[i], eff[i - 1], eff[i]
    return w0 + (TAU * eff.max() - e0) * (w1 - w0) / (e1 - e0) if e1 > e0 else w1


# ---------------- the three width sweeps ----------------
v = json.load(open(f"{ROOT}/vision-model-steering/results/metrics/subspace_steer_dinov2.json"))
ranks = np.array(v["ranks"], float)

m = json.load(open(f"{ROOT}/nlp-remove-jspace-layer/blog/figures/metrics_100.json"))
cum = [c for c in m["cumulative"] if c["k"] > 0]
w_lay = np.array([c["k"] for c in cum], float)
max_lay = 1.0 - next(s for s in m["sweep"] if s["layer"] == 0)["top1_mean"]
eff_lay = (1.0 - np.array([c["top1_mean"] for c in cum])) / max_lay

z = np.load(f"{ROOT}/gpt2-mech-interp/results/r1b_lattice_greedy.npz", allow_pickle=True)
frac = (z["coalition_value"] - z["v_empty"][0]) / (z["v_full"][0] - z["v_empty"][0])
sz = z["coalition_size"]
w_head = np.arange(1, sz.max() + 1, dtype=float)
eff_head = np.array([frac[sz == s].max() for s in w_head])

SERIES = [
    dict(lab="GPT-2 · heads", unit="heads", w=w_head, e=eff_head,
         c=BLUE, mk="o", ls="-", k_repr=None),
    dict(lab="Llama-8B · layers", unit="layers", w=w_lay, e=eff_lay,
         c=ORANGE, mk="s", ls="-", k_repr=None),
    dict(lab="DINOv2 · orientation", unit="dirs", w=ranks,
         e=np.array(v["orientation"]["closed"], float), c=GREEN, mk="^", ls="-",
         k_repr=v["orientation"]["kstar"]),
    dict(lab="DINOv2 · size", unit="dirs", w=ranks,
         e=np.array(v["size"]["closed"], float), c=PURPLE, mk="D", ls="--",
         k_repr=v["size"]["kstar"]),
]
for s in SERIES:
    s["k_eff"] = kstar_eff(s["w"], s["e"])
    # normalise by the effect-side k* where it exists, else the representation-side
    s["k_norm"] = s["k_eff"] if s["k_eff"] is not None else float(s["k_repr"])
    s["x"] = s["w"] / s["k_norm"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.9, 2.62),
                               gridspec_kw=dict(width_ratios=[1.46, 1.0], wspace=0.29))

# ================= panel A =================
axA.axhspan(0.08, 0.12, color=BAND, zorder=0, lw=0)
axA.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
axA.axhline(TAU, color=GRID, lw=0.7, zorder=1)

for s in SERIES:
    keep = s["x"] <= 2.05
    axA.plot(s["x"][keep], s["e"][keep], color=s["c"], lw=1.7, ls=s["ls"],
             marker=s["mk"], ms=3.8, mew=0, zorder=4)
    axA.plot(s["x"][0], s["e"][0], marker=s["mk"], ms=6.4, mfc=SURFACE,
             mec=s["c"], mew=1.5, zorder=6)          # width-1 = standard practice

axA.set_xlim(-0.05, 2.06)
axA.set_ylim(-0.04, 1.20)
axA.set_xlabel("intervention width / $k^*$", labelpad=2)
axA.set_ylabel("fraction of achievable effect", labelpad=3)
axA.set_yticks([0, 0.25, 0.5, 0.75, 0.95])
axA.set_yticklabels(["0", ".25", ".50", ".75", ".95"])
axA.grid(axis="y", color=GRID, lw=0.5, zorder=0)
axA.set_axisbelow(True)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)

axA.text(1.03, 0.60, "width $=k^*$", color=MUTED, fontsize=6.5,
         rotation=90, va="center", ha="left")
axA.annotate("standard practice (width 1)\nrecovers $\\approx$10% of the effect",
             xy=(0.155, 0.098), xytext=(0.60, 0.015),
             color=INK, fontsize=6.6, va="bottom", linespacing=1.35,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                             shrinkA=2, shrinkB=2))
axA.annotate("$k^*_{\\mathrm{repr}}{=}16$ dirs, but the effect\n"
             "saturates at .40 — readout-limited",
             xy=(1.05, 0.383), xytext=(1.16, 0.60), color=PURPLE, fontsize=6.6,
             va="center", linespacing=1.4,
             arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.7,
                             shrinkA=2, shrinkB=2))


def leg_label(s):
    if s["k_eff"] is None:
        return f"{s['lab']}  (none)"
    return f"{s['lab']}  ({s['k_eff']:.0f})"


leg = axA.legend([plt.Line2D([], [], color=s["c"], lw=1.7, ls=s["ls"],
                             marker=s["mk"], ms=3.8, mew=0) for s in SERIES],
                 [leg_label(s) for s in SERIES],
                 loc="upper left", fontsize=6.4, frameon=False,
                 handlelength=1.9, labelspacing=0.30, borderpad=0.0,
                 title="$k^*_{\\mathrm{eff}}$ (units in caption)",
                 alignment="left", bbox_to_anchor=(-0.015, 1.045))
leg.get_title().set_fontsize(6.4)
leg.get_title().set_color(MUTED)
for t in leg.get_texts():
    t.set_color(INK)
axA.set_title("(a) one law, three granularities", fontsize=7.6, loc="left",
              pad=4, color=INK)

# ================= panel B =================
p = json.load(open(f"{ROOT}/vision-model-steering/results/metrics/steer_paligemma_l10.json"))
al = np.array(p["alphas"], float)
axB.axhline(p["p_large_cav"][p["alphas"].index(0.0)], color=GRID, lw=0.7, zorder=1)
axB.plot(al, p["p_large_rand"], color=ORANGE, lw=1.7, ls="--", marker="s",
         ms=3.8, mew=0, zorder=3)
axB.plot(al, p["p_large_cav"], color=GREEN, lw=1.7, marker="^", ms=3.8,
         mew=0, zorder=4)

axB.set_xlabel(r"steering coefficient $\alpha$", labelpad=2)
axB.set_ylabel(r'p("large")', labelpad=3)
axB.set_xlim(-6.9, 6.9)
axB.set_ylim(0.06, 0.75)
axB.set_xticks([-6, -4, -2, 0, 2, 4, 6])
axB.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
axB.grid(axis="y", color=GRID, lw=0.5, zorder=0)
axB.set_axisbelow(True)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)

axB.annotate("matched random control", xy=(-4.0, 0.646), xytext=(-3.3, 0.706),
             color=INK, fontsize=6.6, va="center",
             arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7,
                             shrinkA=2, shrinkB=2))
axB.annotate("size CAV", xy=(4.0, 0.240), xytext=(4.15, 0.372),
             color=INK, fontsize=6.6, va="center", ha="center",
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7,
                             shrinkA=2, shrinkB=2))
axB.text(0.0, 0.088, "captions invariant across all $\\alpha$",
         color=MUTED, fontsize=6.5, va="center", ha="center")
axB.set_title("(b) the readout boundary — no width helps", fontsize=7.6,
              loc="left", pad=4, color=INK)

fig.savefig(f"{OUT}/fig1.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}/fig1.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote fig1.pdf / fig1.png\n")
print(f"{'series':34s} {'k*_repr':>8s} {'k*_eff':>7s} {'eff@1':>6s} {'max':>6s}")
for s in SERIES:
    ke = "none" if s["k_eff"] is None else f"{s['k_eff']:.1f}"
    print(f"{s['lab']:34s} {str(s['k_repr']):>8s} {ke:>7s} "
          f"{s['e'][0]:>6.3f} {s['e'].max():>6.3f}")
