"""Milestone 5X: criterion-free width sweep (camera-ready check, reviewer vd7n).

M5W sweeps only heads that pass the previous-token criterion, so its agreement
between n_writers and k*_eff is partly circular. Here the criterion is dropped:
every head upstream of the reader is ranked by its OWN single-head ablation
effect on the reader's induction score, and the top-K by that causal ranking are
ablated jointly for K = 1 .. K_MAX. No role score enters selection or ordering.

Reported: k*_eff (smallest K reaching 95% of the max drop over the sweep), the
width-1 fraction, and the overlap of the causal top-n_w with the M5W criterion
writers. Same eval, seed, batch and reference batch as m5w_width_sweep.py.

Usage: `PYTHONPATH=. python experiments/m5x_criterion_free.py <tl_model_name>`
Writes results/m5x_<model>.npz.
"""

import sys

import numpy as np

from experiments.m5w_width_sweep import BATCH, SEED, SEQ_LEN, score_matrices
from interp.ablation import build_ablation_hooks, precompute_head_means
from interp.artifacts import build_manifest, save_artifact
from interp.data import make_repeated_tokens
from interp.loading import device_or_default, load_model
from interp.roles import BY_ROLE, select

K_MAX = 30
TAU = 0.95


def main(model_name: str) -> int:
    print(f"=== Milestone 5X (criterion-free width): {model_name} ===")
    device = device_or_default()
    model = load_model(model_name, device=device, seed=SEED, processed=True)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    bos = model.tokenizer.bos_token_id
    if bos is None:
        bos = model.tokenizer.eos_token_id
    tokens = make_repeated_tokens(BATCH, SEQ_LEN, model.cfg.d_vocab, bos, SEED).to(device)
    reference = make_repeated_tokens(
        BATCH, SEQ_LEN, model.cfg.d_vocab, bos, SEED + 1000
    ).to(device)

    ind, prev = score_matrices(model, tokens)
    reader = tuple(int(x) for x in np.unravel_index(int(np.argmax(ind)), ind.shape))
    base = float(ind[reader])
    print(f"  reader: L{reader[0]}H{reader[1]} induction {base:.4f}")

    crit = BY_ROLE["previous_token"]
    writers = [h for h in select(prev, crit.frac, crit.sign) if h[0] < reader[0]]
    n_w = len(writers)

    means = precompute_head_means(model, reference)

    def reader_score(heads):
        a_ind, _ = score_matrices(model, tokens,
                                  fwd_hooks=build_ablation_hooks(list(heads), means))
        return float(a_ind[reader])

    upstream = [(l, h) for l in range(reader[0]) for h in range(n_heads)]
    single = np.array([base - reader_score([lh]) for lh in upstream])
    order = [upstream[i] for i in np.argsort(-single)]
    print(f"  upstream heads: {len(upstream)}; top-5 by single-head drop: "
          f"{[(f'L{l}H{h}', round(float(single[upstream.index((l, h))]), 3)) for l, h in order[:5]]}")

    k_max = min(K_MAX, len(order))
    drop = np.array([base - reader_score(order[:k]) for k in range(1, k_max + 1)])
    tgt = TAU * drop.max()
    k_eff = int(np.nonzero(drop >= tgt)[0][0]) + 1
    overlap = len(set(order[:n_w]) & set(writers)) if n_w else 0
    print(f"  n_w (criterion) {n_w}  k*_eff (criterion-free) {k_eff}  "
          f"width-1 {drop[0] / drop.max():.3f}  overlap top-n_w {overlap}/{n_w}")

    path = f"results/m5x_{model_name.replace('/', '_')}.npz"
    save_artifact(
        path,
        {
            "base_score": np.array([base]),
            "single_drop": single,
            "upstream": np.array(upstream, dtype=np.int64),
            "causal_order": np.array(order[:k_max], dtype=np.int64),
            "width_drop": drop,
            "criterion_writers": np.array(writers, dtype=np.int64).reshape(-1, 2),
            "k_eff": np.array([k_eff]),
            "n_w": np.array([n_w]),
            "overlap": np.array([overlap]),
        },
        build_manifest(model_name, SEED, extra={
            "milestone": "5X",
            "experiment": "m5x_criterion_free",
            "reader": f"L{reader[0]}H{reader[1]}",
            "selection": "all heads with layer < reader layer, ranked by single-head "
                         "mean-ablation drop in reader induction score; no role criterion",
            "k_max": k_max,
            "tau": TAU,
        }),
    )
    print(f"artifact written: {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: m5x_criterion_free.py <transformer_lens_model_name>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
