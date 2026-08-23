#!/usr/bin/env python3
"""E1 + E0c — canonical k*_repr, per layer, all encoders x attributes, plus
intrinsic-dimension companions (stable rank / participation ratio), persisted.

CANONICAL DEFINITION (used everywhere in the paper):
  On the saved pooled bank X at layer L (centered, NOT standardized):
    SVD -> top n_pc=120 components W; Z = Xc @ W
    relevance_j = sum_t corr(Z_j, t)^2 over attribute targets
       size: t = {scale}                (1 target)
       orientation: t = {sin th, cos th} (2 targets — stated in the paper)
    k* = smallest K with cumsum(sorted relevance)/total >= 0.95
  This is exactly the definition steering used (subspace_steer.py) — so k*_repr and
  k*_eff are comparable by construction. Differences vs other repo scripts (n_pc,
  standardization) are reported in the audit block at the end.

Also per (encoder, attr, layer): bin-mean manifold stable_rank and part_ratio
(intrinsic_dim.py definitions, never persisted until now), and the top-1 relevance
fraction (monosemanticity proxy).

CPU-only, reads saved banks. Writes results/kstar_canonical.json.
"""
import json
import numpy as np
import torch
from pathlib import Path

VROOT = Path("/path/to/Documents/workspace/vision-model-steering")
OUT = Path("/path/to/Documents/workspace/interp4discovery-neurips2026/results/kstar_canonical.json")

MODELS = ["dinov2-base", "clip-base", "siglip-base"]
N_PC = 120


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den < 1e-9 else float((a * b).sum() / den)


def kstar_relevance(X, targets, n_pc=N_PC):
    mu = X.mean(0); Xc = X - mu
    n_pc = min(n_pc, Xc.shape[0] - 1, Xc.shape[1])
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:n_pc].T
    rel = np.zeros(n_pc)
    for t in targets:
        rel += np.array([corr(Z[:, j], t) ** 2 for j in range(n_pc)])
    rel = np.sort(rel)[::-1]; tot = rel.sum()
    if tot <= 1e-9:
        return n_pc, 0.0
    k = int(np.argmax(np.cumsum(rel) / tot >= 0.95)) + 1
    return k, float(rel[0] / tot)


def manifold_dims(X, bin_key):
    bins = sorted(set(bin_key.tolist()))
    M = np.stack([X[bin_key == b].mean(0) for b in bins])
    M = M - M.mean(0)
    lam = np.linalg.svd(M, compute_uv=False) ** 2
    return (float(lam.sum() / (lam.max() + 1e-12)),
            float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-12)))


def load_attr(emb_dir, attr):
    meta = sorted(json.load(open(emb_dir / "meta.json")), key=lambda r: r["global_idx"])
    if attr == "size":
        y = np.array([r["scale"] for r in meta], np.float32)
        targets, bink = [y], np.round(y, 3)
    else:
        th = np.radians(np.array([r["theta_deg"] for r in meta], np.float32))
        targets, bink = [np.sin(th), np.cos(th)], np.array([r["theta_deg"] for r in meta])
    return meta, targets, bink


res = {"definition": "centered raw SVD, n_pc=120, corr^2 relevance, 95% cumulative; "
                     "orientation uses 2 targets (sin, cos)",
       "models": {}}

for attr, emb in [("size", VROOT / "results/embeddings"),
                  ("orientation", VROOT / "results/embeddings_orient")]:
    _, targets, bink = load_attr(emb, attr)
    for m in MODELS:
        bank = torch.load(emb / m / "embeddings.pt").numpy()  # (N, 13, 768)
        L = bank.shape[1]
        ks, top1f, srs, prs = [], [], [], []
        for l in range(L):
            k, t1 = kstar_relevance(bank[:, l, :], targets)
            sr, pr = manifold_dims(bank[:, l, :], bink)
            ks.append(k); top1f.append(round(t1, 4))
            srs.append(round(sr, 3)); prs.append(round(pr, 3))
        res["models"].setdefault(m, {})[attr] = dict(
            kstar_per_layer=ks, top1_frac_per_layer=top1f,
            stable_rank_per_layer=srs, part_ratio_per_layer=prs,
            kstar_L11=ks[11], stable_rank_L11=srs[11], part_ratio_L11=prs[11])
        print(f"{m:13s} {attr:12s} k*/layer: {ks}  (L11: k*={ks[11]}, "
              f"sr={srs[11]}, pr={prs[11]})")

# audit: compare with previously saved k* values at L11
saved = json.load(open(VROOT / "results/metrics/subspace_steer_all.json"))
audit = {}
for m in MODELS:
    for attr in ("size", "orientation"):
        old = saved["models"][m][attr]["kstar"]
        new = res["models"][m][attr]["kstar_L11"]
        audit[f"{m}/{attr}"] = dict(saved_sweep=old, canonical=new, delta=new - old)
res["audit_vs_saved_sweep_L11"] = audit
print("\naudit vs subspace_steer_all.json (L11):")
for k2, v2 in audit.items():
    flag = "" if v2["delta"] == 0 else "  <-- differs"
    print(f"  {k2:26s} saved={v2['saved_sweep']:3d} canonical={v2['canonical']:3d}{flag}")

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
