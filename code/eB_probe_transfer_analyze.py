#!/usr/bin/env python3
"""Experiment B, stage 2 — probe-transfer width as a representation-side
k*_repr for the LAYER granularity.

Idea: if a band of layers carries the same task signal redundantly (the reason
single-layer ablation is survivable), then a linear probe trained on layer i
should transfer to layer j within the band. The width of the mutual-transfer
block is measurable WITHOUT any ablation. Compare it to the interventional
k*_eff ~= 13 of the band sweep.

Protocol: unit-normalize each state vector (scale-free across layers), train a
4-way logistic probe per layer on a train split, evaluate every probe on every
layer's test split -> 33x33 transfer matrix T. For each layer i,
w(i) = #{j : T[i,j] >= 0.9 * T[i,i]}. Report w over the band region.

Input: results/eB_states_vm.npz (mean-pooled arm; last-token as robustness).
Output: results/eB_probe_transfer.json
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RES = Path("/path/to/Documents/workspace/interp4discovery-neurips2026/results")
d = np.load(RES / "eB_states_vm.npz")
labels = d["labels"]


def transfer_matrix(states):
    N, L, H = states.shape
    X = states.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3,
                                      random_state=0, stratify=labels)
    probes = []
    for l in range(L):
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(X[idx_tr, l], labels[idx_tr])
        probes.append(clf)
    T = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            T[i, j] = probes[i].score(X[idx_te, j], labels[idx_te])
    return T


def widths(T, tau=0.9):
    L = T.shape[0]
    return [int((T[i] >= tau * T[i, i]).sum()) for i in range(L)]


out = {}
for arm in ("mean_states", "last_states"):
    T = transfer_matrix(d[arm])
    w = widths(T)
    self_acc = [round(float(T[i, i]), 3) for i in range(T.shape[0])]
    # the ablation band was decoder layers 5..16 -> hidden-state indices 6..17
    band = w[6:18]
    out[arm] = dict(self_acc=self_acc, transfer_width=w,
                    band_width_mean=round(float(np.mean(band)), 1),
                    band_width_range=[int(min(band)), int(max(band))],
                    tau=0.9)
    print(f"[{arm}] self-acc L0={self_acc[0]:.2f} L11={self_acc[11]:.2f} "
          f"L32={self_acc[32]:.2f}")
    print(f"[{arm}] transfer width per layer: {w}")
    print(f"[{arm}] width over the band (L6-17): mean {np.mean(band):.1f}, "
          f"range {min(band)}-{max(band)}  (interventional k*_eff ~ 13)")

json.dump(out, open(RES / "eB_probe_transfer.json", "w"), indent=1)
print("wrote eB_probe_transfer.json")
