#!/usr/bin/env python3
"""E0b — GPT-2 / M5 offline width tables (Fig 1a source, Fig 2 inset, Table 1 rows).

All inputs are committed artifacts in gpt2-mech-interp/results/. No model loads.

Produces results/gpt2_width.json with:
  - best/mean coalition-value curves per size, in BOTH normalizations
      ours:  (v - v_empty) / (v_full - v_empty)
      repo:  (v - corrupt_ld) / (baseline_ld - corrupt_ld)
    + the affine conversion constants.
  - per-model k*_repr analogues from the M5 score matrices:
      n_prev_ge_half   : prev-token heads >= 0.5 * max   (role criterion, all layers)
      n_writers        : that, AND layer < top-induction-head layer (what M5 ablated)
      n_ind_ge_half    : induction heads >= 0.5 * max
  - per-model writer ladders (sorted prev-token scores) + cliff ratio (top1/top2)
  - the single-vs-joint K-composition table (scores array, 7 models)
  - redundancy correlations recomputed from m5_universality.npz
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path("/path/to/Documents/workspace/gpt2-mech-interp/results")
OUT = Path("/path/to/Documents/workspace/interp4discovery-neurips2026/results/gpt2_width.json")

res = {}

# ---------- coalition width curves, both normalizations ----------
z = np.load(ROOT / "r1b_lattice_greedy.npz", allow_pickle=True)
v, s = z["coalition_value"], z["coalition_size"]
v_e, v_f = float(z["v_empty"][0]), float(z["v_full"][0])
b, c = float(z["baseline_ld"][0]), float(z["corrupt_ld"][0])
sizes = list(range(0, int(s.max()) + 1))
curves = {}
for name, lo, hi in [("ours", v_e, v_f), ("repo", c, b)]:
    f = (v - lo) / (hi - lo)
    curves[name] = dict(
        best=[float(f[s == k].max()) for k in sizes],
        mean=[float(f[s == k].mean()) for k in sizes],
        min=[float(f[s == k].min()) for k in sizes])
# affine conversion ours -> repo: f_repo = a * f_ours + b0
a = (v_f - v_e) / (b - c)
b0 = (v_e - c) / (b - c)
res["coalitions"] = dict(
    sizes=sizes, curves=curves,
    anchors=dict(v_empty=v_e, v_full=v_f, baseline_ld=b, corrupt_ld=c),
    conversion=dict(a=float(a), b=float(b0),
                    note="frac_repo = a*frac_ours + b; harder-bar direction: ours"),
    head_labels=json.loads(str(z["_manifest"]))["head_labels"])

# ---------- M5 per-model k*_repr + ladders + single-vs-joint ----------
MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl",
          "EleutherAI_pythia-410m", "EleutherAI_pythia-1.4b", "meta-llama_Llama-3.2-1B"]
per_model = {}
for m in MODELS:
    d = np.load(ROOT / f"m5_{m}.npz", allow_pickle=True)
    ind, prev = d["induction_scores"], d["prev_token_scores"]
    ind_flat, prev_flat = np.nan_to_num(ind).ravel(), np.nan_to_num(prev).ravel()
    top_ind = np.unravel_index(np.nanargmax(ind), ind.shape)
    ladder = np.sort(prev_flat)[::-1]
    ladder = ladder[ladder > 0.25][:12]
    scores = d["scores"]  # [base, treat, ctrl, treat_drop, ctrl_drop, joint, joint_ctrl]
    per_model[m] = dict(
        n_layers=int(ind.shape[0]), n_heads=int(ind.shape[1]),
        top_induction=dict(layer=int(top_ind[0]), head=int(top_ind[1]),
                           score=float(ind[top_ind])),
        n_prev_ge_half=int((prev_flat >= 0.5 * prev_flat.max()).sum()),
        n_writers=int(d["writers"].shape[0]),
        n_ind_ge_half=int((ind_flat >= 0.5 * ind_flat.max()).sum()),
        writer_ladder=[round(float(x), 4) for x in ladder],
        cliff_ratio=float(ladder[0] / ladder[1]) if len(ladder) > 1 else None,
        scores=dict(zip(["base", "treat_1head", "ctrl_1head", "treat_drop",
                         "ctrl_drop", "joint", "joint_ctrl"],
                        [round(float(x), 4) for x in scores])))
res["m5"] = per_model

# ---------- redundancy correlations ----------
u = np.load(ROOT / "m5_universality.npz", allow_pickle=True)
arr = u["summary"]
lay, nw, ni = arr[:, 0], arr[:, 13], arr[:, 14]
res["redundancy"] = dict(
    corr_layers_writers=float(np.corrcoef(lay, nw)[0, 1]),
    corr_layers_induction=float(np.corrcoef(lay, ni)[0, 1]),
    n_layers=[int(x) for x in lay], n_writers=[int(x) for x in nw],
    n_induction=[int(x) for x in ni])

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)

print("coalition best-of-size (ours):",
      " ".join(f"{x:.3f}" for x in curves["ours"]["best"]))
print("conversion: f_repo = {a:.6f}*f_ours + {b:.6f}".format(a=a, b=b0))
print(f"\n{'model':>26} {'prev>=.5':>8} {'writers':>8} {'ind>=.5':>8} {'cliff':>6} "
      f"{'1-head drop':>11} {'joint drop':>10}")
for m, d in per_model.items():
    sc = d["scores"]
    print(f"{m:>26} {d['n_prev_ge_half']:>8} {d['n_writers']:>8} {d['n_ind_ge_half']:>8} "
          f"{d['cliff_ratio']:>6.2f} {sc['base']-sc['treat_1head']:>11.3f} "
          f"{sc['base']-sc['joint']:>10.3f}")
print(f"\nredundancy: writers r={res['redundancy']['corr_layers_writers']:+.3f}, "
      f"induction r={res['redundancy']['corr_layers_induction']:+.3f}")
print(f"wrote {OUT}")
