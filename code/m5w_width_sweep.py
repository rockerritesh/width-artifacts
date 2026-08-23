"""Milestone 5W: the ablation WIDTH curve, per model.

M5 measured two endpoints per model: ablate the single top previous-token head,
and ablate every head passing the previous-token criterion jointly. Between those
endpoints nothing was measured, and the gap is exactly the question a width claim
needs answered: how does the reader's induction score fall as the top-k writers
are removed together, k = 1 .. n_writers?

Usage: `PYTHONPATH=. python experiments/m5w_width_sweep.py <transformer_lens_name>`
Writes results/m5w_<model>.npz. The m5_<model>.npz endpoint artifacts are left
untouched and serve as the cross-check: this sweep's k=1 and k=n values must
reproduce them (same eval, same seed, same reference batch).

Design, unchanged from m5_one_model.py so the numbers compare like with like:
  - writers = heads passing the previous-token criterion (>= 0.5 of max) AND
    earlier than the top induction head, ranked by prev-token score descending.
  - control at every k: size- AND layer-matched — for each of the top-k writers,
    the lowest-prev-scoring unclaimed head in that same layer.
  - mean-ablation reference is a SEPARATE batch (defect-9 guard retained: if the
    k=1 treatment changes no induction score at all, the run aborts).
  - readout = the top induction head's induction score under each ablation.

One deliberate improvement over m5_one_model.py: run_with_cache carries a
names_filter for hook_pattern only — score_matrices reads nothing else, and the
unfiltered cache on gpt2-xl is ~1.5 GB.
"""

import sys

import numpy as np
import torch

from interp.ablation import build_ablation_hooks, precompute_head_means
from interp.artifacts import build_manifest, save_artifact
from interp.data import make_repeated_tokens
from interp.loading import device_or_default, load_model
from interp.metrics import induction_score, prev_token_score
from interp.roles import BY_ROLE, select

SEED = 0
BATCH = 8
SEQ_LEN = 50


def artifact_path(model_name: str) -> str:
    return f"results/m5w_{model_name.replace('/', '_')}.npz"


def score_matrices(model, tokens, fwd_hooks=None):
    """Induction and previous-token scores for every head, pattern hooks only."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    ind = np.zeros((n_layers, n_heads), dtype=np.float64)
    prev = np.zeros((n_layers, n_heads), dtype=np.float64)
    names_filter = lambda n: n.endswith("attn.hook_pattern")

    import contextlib
    ctx = model.hooks(fwd_hooks=fwd_hooks) if fwd_hooks else contextlib.nullcontext()
    with ctx:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names_filter)
        for layer in range(n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
            ind[layer] = induction_score(pattern, SEQ_LEN).cpu().numpy()
            prev[layer] = prev_token_score(pattern).cpu().numpy()
    return ind, prev


def matched_control(prev, chosen_writers, already=()):
    """For each chosen writer, the lowest-prev-scoring unclaimed head in that layer."""
    control: list[tuple[int, int]] = list(already)
    for layer, _ in chosen_writers:
        row = prev[layer].copy()
        for l, h in chosen_writers:
            if l == layer:
                row[h] = np.inf
        for l, h in control:
            if l == layer:
                row[h] = np.inf
        control.append((layer, int(np.argmin(row))))
    return control[len(already):]


def main(model_name: str) -> int:
    print(f"=== Milestone 5W (width sweep): {model_name} ===")
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
    ind_head = tuple(int(x) for x in np.unravel_index(int(np.argmax(ind)), ind.shape))
    base_score = float(ind[ind_head])
    print(f"  reader: L{ind_head[0]}H{ind_head[1]} induction {base_score:.4f}")

    crit = BY_ROLE["previous_token"]
    writers = [h for h in select(prev, crit.frac, crit.sign) if h[0] < ind_head[0]]
    writers.sort(key=lambda lh: -float(prev[lh[0], lh[1]]))
    n_w = len(writers)
    print(f"  writers ({n_w}, ranked): "
          f"{[(f'L{l}H{h}', round(float(prev[l, h]), 3)) for l, h in writers]}")
    if n_w == 0:
        print("  no writers pass the criterion; nothing to sweep. Recorded as such.")
        return 0

    # one cached pass for all head means; reused by every ablation below
    means = precompute_head_means(model, reference)

    def ablated_reader_score(heads):
        hooks = build_ablation_hooks(list(heads), means)
        a_ind, _ = score_matrices(model, tokens, fwd_hooks=hooks)
        return float(a_ind[ind_head]), a_ind

    width_treat = np.full(n_w, np.nan)
    width_ctrl = np.full(n_w, np.nan)
    control_sets: list[list[tuple[int, int]]] = []
    for k in range(1, n_w + 1):
        chosen = writers[:k]
        ctrl = matched_control(prev, chosen)
        control_sets.append(ctrl)
        t_score, t_ind = ablated_reader_score(chosen)
        c_score, _ = ablated_reader_score(ctrl)
        if k == 1 and np.allclose(t_ind, ind, atol=1e-8):
            raise RuntimeError(
                "ablating the top writer changed NO induction score at all. The "
                "intervention did not happen -- check hooks and reference batch."
            )
        width_treat[k - 1] = t_score
        width_ctrl[k - 1] = c_score
        print(f"  k={k:>2}  treat {t_score:.4f} (drop {base_score - t_score:+.4f})   "
              f"ctrl {c_score:.4f} (drop {base_score - c_score:+.4f})")

    path = artifact_path(model_name)
    save_artifact(
        path,
        {
            "induction_scores": ind,
            "prev_token_scores": prev,
            "base_score": np.array([base_score]),
            "width_treat": width_treat,
            "width_ctrl": width_ctrl,
            "writer_order": np.array(writers, dtype=np.int64).reshape(-1, 2),
            "control_sets_flat": np.array(
                [h for s in control_sets for h in s], dtype=np.int64
            ).reshape(-1, 2),
            "control_set_sizes": np.array([len(s) for s in control_sets], dtype=np.int64),
        },
        build_manifest(model_name, SEED, extra={
            "milestone": "5W",
            "experiment": "m5w_width_sweep",
            "n_layers": n_layers,
            "n_heads": n_heads,
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "reader": f"L{ind_head[0]}H{ind_head[1]}",
            "base_score": base_score,
            "n_writers": n_w,
            "writer_order": [f"L{l}H{h}" for l, h in writers],
            "criterion": "prev-token >= 0.5 of max AND layer < reader layer, "
                         "ranked by prev-token score descending",
            "control": "per k: size- and layer-matched lowest-prev heads",
            "ablation": "mean over a separate reference batch, batch-dim only",
        }),
    )
    print(f"artifact written: {path}")
    print(f"=== {model_name} COMPLETE ===")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: m5w_width_sweep.py <transformer_lens_model_name>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
