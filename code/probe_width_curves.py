"""Fig 1 test: do the three modalities share a normalised width axis?

Common definition, transferred verbatim from the vision k*:
    k* = smallest intervention width reaching 95% of the maximum achievable effect.
x = width / k*   ;   y = effect realised / max achievable effect
"""
import json
import numpy as np

ROOT = "/path/to/Documents/workspace"
TAU = 0.95


def kstar_from_curve(widths, eff):
    """Smallest width reaching TAU of max effect (linear interp between samples)."""
    eff = np.asarray(eff, float)
    widths = np.asarray(widths, float)
    target = TAU * eff.max()
    hit = np.nonzero(eff >= target)[0]
    if not len(hit):
        return None
    i = hit[0]
    if i == 0:
        return widths[0]
    w0, w1, e0, e1 = widths[i - 1], widths[i], eff[i - 1], eff[i]
    return w0 + (target - e0) * (w1 - w0) / (e1 - e0) if e1 > e0 else w1


rows = []

# ---------- 1. vision: rank-K subspace steering (DINOv2-base, L11->L12) ----------
v = json.load(open(f"{ROOT}/vision-model-steering/results/metrics/subspace_steer_dinov2.json"))
ranks = np.array(v["ranks"], float)
for attr in ("orientation", "size"):
    closed = np.array(v[attr]["closed"], float)
    rows.append(dict(
        name=f"vision / {attr} (rank-K CAV)",
        widths=ranks, eff=closed,
        kstar_reported=v[attr]["kstar"],          # independent 95%-variance k*
        kstar_curve=kstar_from_curve(ranks, closed),
        unit="directions", std_practice=1.0,
    ))

# ---------- 2. Llama-3.1-8B: contiguous middle-band removal (100 queries) ----------
m = json.load(open(f"{ROOT}/nlp-remove-jspace-layer/blog/figures/metrics_100.json"))
cum = [c for c in m["cumulative"] if c["k"] > 0]
w_lay = np.array([c["k"] for c in cum], float)
# effect = departure from the intact model's own continuation
eff_lay = np.array([1.0 - c["top1_mean"] for c in cum], float)
# max achievable = the destroyed-model reference (single-layer L0 ablation)
l0 = next(s for s in m["sweep"] if s["layer"] == 0)
max_lay = 1.0 - l0["top1_mean"]
rows.append(dict(
    name="Llama-3.1-8B / layer band",
    widths=w_lay, eff=eff_lay / max_lay,
    kstar_reported=None,
    kstar_curve=kstar_from_curve(w_lay, eff_lay / max_lay),
    unit="layers", std_practice=1.0,
))

# ---------- 3. GPT-2 small: IOI head coalitions (512-coalition lattice) ----------
z = np.load(f"{ROOT}/gpt2-mech-interp/results/r1b_lattice_greedy.npz", allow_pickle=True)
val, size = z["coalition_value"], z["coalition_size"]
v_e, v_f = float(z["v_empty"][0]), float(z["v_full"][0])
frac = (val - v_e) / (v_f - v_e)          # fraction of full-circuit logit diff restored
sizes = np.arange(1, size.max() + 1)
best = np.array([frac[size == s].max() for s in sizes])
mean = np.array([frac[size == s].mean() for s in sizes])
rows.append(dict(
    name="GPT-2 small / head coalition (best)",
    widths=sizes.astype(float), eff=best,
    kstar_reported=7,                       # R3's threshold-free tau=0.95 cut
    kstar_curve=kstar_from_curve(sizes, best),
    unit="heads", std_practice=1.0,
))
rows.append(dict(
    name="GPT-2 small / head coalition (mean)",
    widths=sizes.astype(float), eff=mean,
    kstar_reported=None, kstar_curve=kstar_from_curve(sizes, mean),
    unit="heads", std_practice=1.0,
))

# ---------- report ----------
print(f"{'modality':38s} {'k* rep':>7s} {'k* curve':>9s} {'eff@w=1':>9s} {'eff@k*':>8s}")
print("-" * 78)
for r in rows:
    ks = r["kstar_curve"]
    e1 = np.interp(r["std_practice"], r["widths"], r["eff"])
    eks = np.interp(ks, r["widths"], r["eff"]) if ks else float("nan")
    rep = r["kstar_reported"]
    print(f"{r['name']:38s} {str(rep):>7s} {ks:>9.1f} {e1:>9.3f} {eks:>8.3f}")

print("\n--- normalised: x = width/k*_curve, y = effect ---")
for r in rows:
    ks = r["kstar_curve"]
    xs = r["widths"] / ks
    print(f"\n{r['name']}  (k*={ks:.1f} {r['unit']}, standard practice at x={1/ks:.3f})")
    for x, y in zip(xs, r["eff"]):
        if x <= 2.2:
            print(f"   x={x:5.2f}  y={y:6.3f}")

np.save("/private/tmp/claude-501/-Users-USER-Documents-workspace/"
        "14059083-264b-40a4-9bd5-e8ed8bc6bac0/scratchpad/width_rows.npy",
        np.array(rows, dtype=object), allow_pickle=True)
