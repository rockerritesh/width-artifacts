# Your Intervention Was Too Narrow — code and artifacts

Anonymized artifact bundle for the Interp4Discovery 2026 submission.

- `code/` — all analysis and experiment scripts. `make_stats.py` regenerates
  every number in the paper (stats.tex + table1.tex) from `results/`.
  `fig1_main.py`/`fig2..4_*.py` regenerate every figure.
- `results/` — one JSON/NPZ per experiment; `results/upstream/` holds the
  raw sweep artifacts (coalition lattice, width sweeps, steering sweeps)
  the analyses read.
- `figures/` — the compiled figure PDFs.

Model checkpoints are public (HuggingFace); paths in scripts are
placeholders. Full de-anonymized repositories will be released on
acceptance.
