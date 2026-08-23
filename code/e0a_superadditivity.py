#!/usr/bin/env python3
"""E0a — superadditivity of band ablation (Fig 3a, claim C2).

From nlp-remove-jspace-layer/blog/figures/per_query_results.json (100 queries,
Llama-3.1-8B-Instruct, teacher-forced): compare the effect of removing a contiguous
band of k layers against the additive prediction from the same layers' single-layer
ablations.

  KL:    superadditivity ratio  = KL(band) / sum_{l in band} KL(single_l)
  top-1: independence prediction = prod_{l in band} top1(single_l)   (survival model)

Band definition (verified identical in the 8-prompt and 100-query runs):
  start = max(0, 11 - k//2); idxs = range(start, start+k)

Outputs: results/superadditivity.json + printed table with bootstrap CIs.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path("/path/to/Documents/workspace")
SRC = ROOT / "nlp-remove-jspace-layer/blog/figures/per_query_results.json"
OUT = ROOT / "interp4discovery-neurips2026/results/superadditivity.json"

rows = json.load(open(SRC))
KS = [1, 2, 4, 6, 8, 10, 12, 14]
rng = np.random.default_rng(0)
N_BOOT = 10_000


def band(k):
    s = max(0, 11 - k // 2)
    return list(range(s, s + k))


def boot_ci(x, stat=np.mean):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    vals = stat(x[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


res = {"n_queries": len(rows), "ks": KS, "band_idxs": {k: band(k) for k in KS},
       "per_k": {}, "per_category": {}}

print(f"{'k':>3} {'KL(band)':>9} {'ΣKL(singles)':>13} {'ratio':>7} {'CI95':>15} "
      f"{'top1(band)':>11} {'Πtop1':>7} {'Δ':>7}")
for k in KS:
    idxs = band(k)
    kl_band, kl_sum, t1_band, t1_prod, ratios = [], [], [], [], []
    for r in rows:
        kb = r["cumulative"][str(k)]["kl"]
        ks_ = sum(r["single"][str(l)]["kl"] for l in idxs)
        tb = r["cumulative"][str(k)]["top1"]
        tp = float(np.prod([r["single"][str(l)]["top1"] for l in idxs]))
        kl_band.append(kb); kl_sum.append(ks_); t1_band.append(tb); t1_prod.append(tp)
        if ks_ > 1e-9:
            ratios.append(kb / ks_)
    ratio_mean_of_means = float(np.mean(kl_band) / np.mean(kl_sum))
    lo, hi = boot_ci(np.array(ratios))
    d = float(np.mean(t1_band) - np.mean(t1_prod))
    res["per_k"][k] = dict(
        kl_band_mean=float(np.mean(kl_band)), kl_sum_singles_mean=float(np.mean(kl_sum)),
        ratio_of_means=ratio_mean_of_means, ratio_per_query_mean=float(np.mean(ratios)),
        ratio_ci95=[lo, hi], top1_band_mean=float(np.mean(t1_band)),
        top1_independence_mean=float(np.mean(t1_prod)), top1_delta=d)
    print(f"{k:>3} {np.mean(kl_band):>9.3f} {np.mean(kl_sum):>13.3f} "
          f"{ratio_mean_of_means:>7.2f} [{lo:>6.2f},{hi:>6.2f}] "
          f"{np.mean(t1_band):>11.3f} {np.mean(t1_prod):>7.3f} {d:>+7.3f}")

# per-category ratios at the headline k=12
for cat in sorted({r["category"] for r in rows}):
    sub = [r for r in rows if r["category"] == cat]
    rr = [r["cumulative"]["12"]["kl"] / max(sum(r["single"][str(l)]["kl"] for l in band(12)), 1e-9)
          for r in sub]
    res["per_category"][cat] = dict(k12_ratio_mean=float(np.mean(rr)), n=len(sub))

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nk=12 per-category KL ratios: " +
      ", ".join(f"{c}={v['k12_ratio_mean']:.1f}x" for c, v in res["per_category"].items()))
print(f"wrote {OUT}")
