#!/usr/bin/env python3
"""Generate paper/stats.tex (macros) and paper/table1.tex (diagnostic matrix)
from the committed artifacts. Every number in the paper's prose comes from here.
DO NOT EDIT the outputs by hand.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path("/path/to/Documents/workspace")
P = ROOT / "interp4discovery-neurips2026"
RES, VM, G = P / "results", ROOT / "vision-model-steering/results/metrics", \
    ROOT / "gpt2-mech-interp/results"
NLP = ROOT / "nlp-remove-jspace-layer/blog/figures"

M = {}          # macro name -> formatted value
TAU = 0.95


def kstar_eff(w, e, sat=0.90):
    w, e = np.asarray(w, float), np.asarray(e, float)
    if e.max() < sat:
        return None
    i = np.nonzero(e >= TAU * e.max())[0][0]
    if i == 0:
        return float(w[0])
    return float(w[i - 1] + (TAU * e.max() - e[i - 1]) * (w[i] - w[i - 1])
                 / (e[i] - e[i - 1]))


# ---------------- headline collapse (Fig 1a) ----------------
z = np.load(G / "r1b_lattice_greedy.npz", allow_pickle=True)
frac = (z["coalition_value"] - z["v_empty"][0]) / (z["v_full"][0] - z["v_empty"][0])
sz = z["coalition_size"]
w_h = np.arange(1, sz.max() + 1, dtype=float)
e_h = np.array([frac[sz == s].max() for s in w_h])
M["GptWidthOne"] = f"{e_h[0]:.3f}"
M["GptKstarEff"] = f"{kstar_eff(w_h, e_h):.1f}"
# necessity direction: damage from dropping ONE head from the full circuit
nec1 = 1.0 - frac[sz == 8]
M["GptNecDropOneMax"] = f"{nec1.max():.2f}"
M["GptNecDropOneMed"] = f"{np.median(nec1):.2f}"
M["GptNecDropOneMin"] = f"{nec1.min():.2f}"

m100 = json.load(open(NLP / "metrics_100.json"))
cum = [c for c in m100["cumulative"] if c["k"] > 0]
w_l = np.array([c["k"] for c in cum], float)
max_l = 1.0 - next(s for s in m100["sweep"] if s["layer"] == 0)["top1_mean"]
e_l = (1.0 - np.array([c["top1_mean"] for c in cum])) / max_l
M["LlamaWidthOne"] = f"{e_l[0]:.3f}"
M["LlamaKstarEff"] = f"{kstar_eff(w_l, e_l):.1f}"
M["LlamaTopOneSingle"] = f"{next(c['top1_mean'] for c in cum if c['k']==1)*100:.0f}\\%"
M["LlamaTopOneBand"] = f"{next(c['top1_mean'] for c in cum if c['k']==12)*100:.0f}\\%"

sub = json.load(open(VM / "subspace_steer_dinov2.json"))
e_o = np.array(sub["orientation"]["closed"], float)
M["OrientWidthOne"] = f"{e_o[0]:.3f}"
M["OrientKstarEff"] = f"{kstar_eff(np.array(sub['ranks'], float), e_o):.1f}"
M["OrientKstarRepr"] = str(sub["orientation"]["kstar"])
M["OrientAtKstar"] = f"{max(e_o):.2f}"

# ---------------- superadditivity ----------------
sa = json.load(open(RES / "superadditivity.json"))
M["SaRatioTwelve"] = f"{sa['per_k']['12']['ratio_of_means']:.1f}"
M["SaRatioFourteen"] = f"{sa['per_k']['14']['ratio_of_means']:.1f}"
M["SaCrossK"] = "10"
kl12 = sa["per_k"]["12"]
M["SaBandKL"] = f"{kl12['kl_band_mean']:.1f}"
M["SaSumKL"] = f"{kl12['kl_sum_singles_mean']:.2f}"

# ---------------- E5 random vs contiguous ----------------
e5 = json.load(open(RES / "e5_random_vs_contiguous.json"))
c12 = next(r for r in e5["contiguous"] if r["k"] == 12)
r12 = [r["kl"] for r in e5["random_region"] if r["k"] == 12]
M["EfiveContigTwelve"] = f"{c12['kl']:.2f}"
M["EfiveRandTwelve"] = f"{np.mean(r12):.2f}"
M["EfiveRandTwelveSd"] = f"{np.std(r12):.2f}"
st = e5["strided"][0]
M["EfiveStrided"] = f"{st['kl']:.2f}"
c6 = next(r for r in e5["contiguous"] if r["k"] == 6)
M["EfiveContigSix"] = f"{c6['kl']:.2f}"
sl_int = [r for r in e5["slide"] if 4 <= r["start"] <= 16]
M["EfiveSlideMin"] = f"{min(r['kl'] for r in sl_int):.1f}"
M["EfiveSlideMax"] = f"{max(r['kl'] for r in sl_int):.1f}"
M["EfiveSlideZero"] = f"{next(r['kl'] for r in e5['slide'] if r['start']==0):.1f}"
M["EfiveSlideTwenty"] = f"{next(r['kl'] for r in e5['slide'] if r['start']==20):.1f}"
M["EfivePrompts"] = str(e5["n_prompts"])

# ---------------- m5w universality ----------------
gw = json.load(open(RES / "gpt2_width.json"))
MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
          "EleutherAI_pythia-410m", "EleutherAI_pythia-1.4b",
          "meta-llama_Llama-3.2-1B"]
LABEL = {"gpt2": "GPT-2 small", "gpt2-medium": "GPT-2 medium",
         "gpt2-large": "GPT-2 large", "gpt2-xl": "GPT-2 XL",
         "EleutherAI_pythia-410m": "Pythia-410m",
         "EleutherAI_pythia-1.4b": "Pythia-1.4b",
         "meta-llama_Llama-3.2-1B": "Llama-3.2-1B"}
rows_h = []
fr1, cliffs = [], []
for key in MODELS:
    zz = np.load(G / f"m5w_{key}.npz", allow_pickle=True)
    base = float(zz["base_score"][0])
    t = zz["width_treat"]
    drop = base - t
    n_w = len(t)
    ke = None
    tgt = TAU * drop[-1]
    hit = np.nonzero(drop >= tgt)[0]
    if len(hit):
        ke = int(hit[0]) + 1
    f1 = float(drop[0] / drop[-1])
    prev = np.sort(np.nan_to_num(zz["prev_token_scores"]).ravel())[::-1]
    cliff = float(prev[0] / prev[1])
    fr1.append(f1); cliffs.append(cliff)
    rows_h.append((LABEL[key], n_w, ke, f1, cliff))
# cliff stats over ALL m5w artifacts present (dynamic: grows with the sweep)
import glob as _glob
from scipy.stats import spearmanr as _spearmanr
_all = sorted(_glob.glob(str(G / "m5w_*.npz")))
_fr1, _cliff, _succ = [], [], 0
for _f in _all:
    _z = np.load(_f, allow_pickle=True)
    _b = float(_z["base_score"][0]); _t = _z["width_treat"]; _d = _b - _t
    _f1 = float(_d[0] / _d[-1])
    _lad = np.sort(np.nan_to_num(_z["prev_token_scores"]).ravel())[::-1]
    _fr1.append(_f1); _cliff.append(float(_lad[0] / _lad[1]))
    _succ += _f1 > 0.5
_rho, _pv = _spearmanr(_cliff, _fr1)
M["CliffN"] = str(len(_all))
M["CliffRho"] = f"{_rho:.2f}"
M["CliffP"] = f"{_pv:.3f}"
M["CliffSucceedN"] = str(_succ)
M["SpearmanCliff"] = f"{_rho:.2f}"
M["HeadFracOneMedian"] = f"{np.median([f for f in fr1]):.2f}"
M["LlamaOneBCliffK"] = "3"
M["RedundancyR"] = f"{gw['redundancy']['corr_layers_induction']:+.2f}"
M["RedundancyRW"] = f"{gw['redundancy']['corr_layers_writers']:+.2f}"

# ---------------- PaliGemma readout ----------------
pg = json.load(open(VM / "pg_subspace.json"))
pr = json.load(open(VM / "exp2_prompted.json"))
rep = np.array(pg["size"]["rep_closed"])
dp = np.array(pg["size"]["p_target"]) - pg["size"]["p_base"]
M["PgRepClosed"] = f"{rep[3]*100:.0f}\\%"
M["PgDpWorst"] = f"{dp.min():+.3f}"
M["PgPromptedGap"] = f"{pr['size']['p_tgt_base']-pr['size']['p_src_base']:+.2f}"
l10 = json.load(open(VM / "steer_paligemma_l10.json"))
M["PgRandPeak"] = f"{max(l10['p_large_rand']):.2f}"
M["PgCavPeak"] = f"{l10['p_large_cav'][l10['alphas'].index(-4.0)]:.2f}"

tw = json.load(open(VM / "so400m_tower_steer.json"))
idk = tw["size"]["id_kept"]
rk = tw["ranks"] if "ranks" in tw else tw["size"]["ranks"]
M["TowerIdAtKstar"] = f"{idk[rk.index(13)]:.2f}"
M["TowerIdAtMax"] = f"{idk[-1]:.2f}"
M["TowerClosedAtKstar"] = f"{tw['size']['closed'][rk.index(13)]:.2f}"

# ---------------- E10 random-subspace control ----------------
e10 = json.load(open(RES / "e10_random_subspace.json"))
M["RandSubMean"] = f"{e10['random']['55']['mean']:.2f}"
M["RandSubSd"] = f"{e10['random']['55']['sd']:.2f}"

# ---------------- B probe-transfer width (layer-side k*_repr) ----------------
epb = json.load(open(RES / "eB_probe_transfer.json"))
M["ProbeBandMean"] = f"{epb['mean_states']['band_width_mean']:.0f}"
M["ProbeBandLo"] = str(epb['mean_states']['band_width_range'][0])
M["ProbeBandHi"] = str(epb['mean_states']['band_width_range'][1])

# ---------------- E patch-field test ----------------
epf = json.load(open(RES / "eE_patch_field.json"))
M["EPatchBase"] = f"{epf['p_base_small']:.2f}"
M["EPatchLargeRef"] = f"{epf['p_base_large']:.2f}"
M["EPatchField"] = f"{epf['field']['1.0']:.2f}"
M["EPatchShuffled"] = f"{epf['shuffled']['1.0']:.2f}"

# ---------------- E2 gate ----------------
e2 = json.load(open(RES / "e2_depth_rank.json"))
best_sz = []
for l, r in e2["models"]["dinov2-base"]["size"]["depths"].items():
    best_sz.append(float(np.interp(r["kstar"], e2["ranks"], r["closed_final"])))
M["EtwoDinoSizeBest"] = f"{max(best_sz):.2f}"
oc = []
for mname in ("dinov2-base", "clip-base", "siglip-base"):
    r = e2["models"][mname]["orientation"]["depths"]["11"]
    oc.append(float(np.interp(r["kstar"], e2["ranks"], r["closed_final"])))
M["EtwoOrientMin"] = f"{min(oc):.2f}"

# ---------------- vision table rows ----------------
ss = json.load(open(VM / "subspace_steer_all.json"))
rows_v = []
for mname in ("dinov2-base", "clip-base", "siglip-base"):
    for attr in ("size", "orientation"):
        d = ss["models"][mname][attr]
        e = np.array(d["closed"], float)
        ke = kstar_eff(np.array(ss["ranks"], float), e)
        rows_v.append((mname.replace("-base", ""), attr, d["kstar"],
                       e[0], ke, float(e.max())))

# ---------------- write stats.tex ----------------
lines = ["% AUTO-GENERATED by code/make_stats.py -- do not edit by hand"]
for k, v in M.items():
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(P / "paper/stats.tex").write_text("\n".join(lines) + "\n")
print(f"stats.tex: {len(M)} macros")

# ---------------- write table1.tex ----------------
def fnum(x, nd=2):
    return "---" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


t = [r"% AUTO-GENERATED by code/make_stats.py -- do not edit by hand",
     r"\begin{tabular}{@{}llrrrrl@{}}", r"\toprule",
     r"granularity / system & attribute & $k^*_{\mathrm{repr}}$ & "
     r"eff.\ @ width 1 & $k^*_{\mathrm{eff}}$ & max eff. & verdict \\",
     r"\midrule"]
for name, attr, kr, e1, ke, emax in rows_v:
    verdict = ("width-limited" if ke is not None and ke <= 1.4 * kr
               else ("basis-limited" if emax < 0.9 else "width-limited$^\\dagger$"))
    t.append(f"{name} L11 & {attr} & {kr} & {e1:.2f} & "
             f"{fnum(ke, 0)} & {emax:.2f} & {verdict} \\\\")
t.append(r"\midrule")
med_f1 = float(np.median([r[3] for r in rows_h]))
t.append(f"7 LMs (Fig.~2) & induction writers & 2--9 & {med_f1:.2f} (median) & "
         f"$n_w{{-}}(0..2)$ & 1.00 & width-limited \\\\")
t.append(r"\midrule")
t.append(f"GPT-2 IOI & head coalition & 9 & {e_h[0]:.2f} & "
         f"{float(M['GptKstarEff']):.0f} & {e_h.max():.2f} & width-limited \\\\")
t.append(f"Llama-3.1-8B & layer band & ${M['ProbeBandMean']}$ (transfer) & {e_l[0]:.2f} & "
         f"{float(M['LlamaKstarEff']):.0f} & {e_l.max():.2f} & width-limited \\\\")
gap_s = pr["size"]["p_tgt_base"] - pr["size"]["p_src_base"]
dp_s = np.array(pg["size"]["p_target"]) - pg["size"]["p_base"]
dp_o = np.array(pg["orientation"]["p_target"]) - pg["orientation"]["p_base"]
t.append(f"PaliGemma readout & size & {pg['size']['kstar']} & "
         f"{dp_s[0]/gap_s:.2f} & none & {dp_s.max()/gap_s:.2f} & "
         f"\\textbf{{readout-limited}} \\\\")
t.append(f"PaliGemma readout & orientation & {pg['orientation']['kstar']} & "
         f"{abs(dp_o[0]):.2f} & none & {abs(dp_o).max():.2f}$^{{\\ddagger}}$ & "
         f"\\textbf{{readout-limited}} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
(P / "paper/table1.tex").write_text("\n".join(t) + "\n")
print("table1.tex written")
print("\nhead rows (label, n_w, k*_eff, frac@1, cliff):")
for r in rows_h:
    print("  ", r)
print("vision rows (model, attr, k*_repr, eff@1, k*_eff, max):")
for r in rows_v:
    print("  ", r)
