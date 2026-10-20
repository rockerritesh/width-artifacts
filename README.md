# Your Intervention Was Too Narrow — code and artifacts

**Mechanism Width as a Precondition for Interpretability-Driven Discovery**
Sumit Yadav, Basanta Joshi — Pulchowk Campus, Institute of Engineering, Tribhuvan University, Nepal
*NeurIPS 2026 Workshop on Interpretability for Discovery (Interp4Discovery), non-archival.*

Paper: [PDF](https://openreview.net/pdf?id=aDQHyo9kyh) · [OpenReview](https://openreview.net/forum?id=aDQHyo9kyh) · [NeurIPS poster](https://neurips.cc/virtual/2026/loc/atlanta/164229) · Project page: [`docs/index.html`](docs/index.html)

> The field's standard intervention unit (one direction, one layer, one attention head) recovers
> only ≈10% of the achievable causal effect in three unrelated testbeds. A negative result at
> standard width is an underpowered test, not evidence of absence. Sweep width, compare
> k\*<sub>repr</sub> to k\*<sub>eff</sub>, and run matched controls before claiming a discovery.

## Layout

- `code/` — all analysis and experiment scripts. `make_stats.py` regenerates every number in the
  paper (`stats.tex` + `table1.tex`) from `results/`. `fig1_main.py` / `fig2..4_*.py` regenerate
  every figure.
- `results/` — one JSON/NPZ per experiment; `results/upstream/` holds the raw sweep artifacts
  (coalition lattice, width sweeps, steering sweeps) the analyses read.
- `figures/` — the compiled figure PDFs.
- `docs/` — project page.

## Camera-ready additions (reviewer feedback)

| What | Script | Output |
|---|---|---|
| Threshold sensitivity: all four analytic constants on a 3⁴ grid; verdict class never flips | `code/e11_threshold_sensitivity.py` | `results/e11_threshold_sensitivity.json`, Table 2 |
| Criterion-free head widths: joint greedy selection with no role score, 7 LMs | `code/m5y_joint_greedy_batched.py` (driver `m5y_all.py`) | `results/upstream/m5y_*.npz` |
| Single-head-ranking comparison (a width-1 ranking; reported only for contrast) | `code/m5x_criterion_free.py` | `results/upstream/m5x_*.npz` |
| IOI width with no size cap (greedy elimination over all 144 heads) | — | `results/upstream/r3_greedy_discovery.npz` |
| Head-width table, macros, cliff vs criterion-free width-1 (ρ = 0.89, p = 0.007, n = 7) | `code/e12_head_width_criterion_free.py` | `results/e12_head_width.json`, Table 3 |
| Fig 1: k\*<sub>repr</sub> ticks; axis normalized by k\*<sub>eff</sub> | `code/fig1_main.py` | `figures/fig1_main.pdf` |

`m5y_joint_greedy_batched.py` evaluates 48 candidate heads per forward pass and stops at the
reader's layer; it reproduces the unbatched `m5y_joint_greedy.py` exactly on GPT-2 small (0 diff)
and medium (max |Δ| = 1.8e-7). The `m5*` scripts import the `interp` package of the companion
mechanistic-interpretability repository (ablation hooks, metrics, artifact manifests).

## Reproducing

Model checkpoints are public on HuggingFace. Paths in scripts are placeholders
(`/path/to/Documents/workspace`); point them at your checkout. Rerun `code/make_stats.py` after
changing any result.

## Citation

```bibtex
@inproceedings{yadav2026narrow,
  title     = {Your Intervention Was Too Narrow: Mechanism Width as a Precondition
               for Interpretability-Driven Discovery},
  author    = {Yadav, Sumit and Joshi, Basanta},
  booktitle = {NeurIPS 2026 Workshop on Interpretability for Discovery},
  year      = {2026},
  note      = {Non-archival}
}
```
