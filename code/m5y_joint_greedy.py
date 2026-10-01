"""Milestone 5Y: criterion-free width by JOINT greedy forward selection.

M5X ranked upstream heads by single-head ablation effect. That ranking is itself
a width-1 measurement: on gpt2-medium the six criterion writers ablated jointly
drop the reader more (0.367) than M5X's best 30 single-ranked heads (0.300), i.e.
single-head attribution misses heads that matter only together.

Here each step adds the upstream head whose ablation, JOINTLY with the heads
already chosen, drops the reader's induction score most. No role score enters.
Cost is ~U * K_MAX forward passes (U = heads upstream of the reader).

Usage: `PYTHONPATH=. python experiments/m5y_joint_greedy.py <tl_model_name>`
Writes results/m5y_<model>.npz.
"""

import sys
import time

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
    print(f"=== Milestone 5Y (joint greedy width): {model_name} ===", flush=True)
    device = device_or_default()
    model = load_model(model_name, device=device, seed=SEED, processed=True)
    n_heads = model.cfg.n_heads

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
    crit = BY_ROLE["previous_token"]
    writers = [h for h in select(prev, crit.frac, crit.sign) if h[0] < reader[0]]
    n_w = len(writers)
    print(f"  reader L{reader[0]}H{reader[1]} base {base:.4f}  n_w {n_w}", flush=True)

    means = precompute_head_means(model, reference)

    def reader_score(heads):
        a_ind, _ = score_matrices(model, tokens,
                                  fwd_hooks=build_ablation_hooks(list(heads), means))
        return float(a_ind[reader])

    upstream = [(l, h) for l in range(reader[0]) for h in range(n_heads)]
    k_max = min(K_MAX, len(upstream))
    chosen: list[tuple[int, int]] = []
    drop = np.zeros(k_max)
    t0 = time.time()
    for k in range(k_max):
        best, best_d = None, -np.inf
        for lh in upstream:
            if lh in chosen:
                continue
            d = base - reader_score(chosen + [lh])
            if d > best_d:
                best, best_d = lh, d
        chosen.append(best)
        drop[k] = best_d
        print(f"  k={k + 1:>2} +L{best[0]}H{best[1]}  drop {best_d:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    k_eff = int(np.nonzero(drop >= TAU * drop.max())[0][0]) + 1
    overlap = len(set(chosen[:n_w]) & set(writers)) if n_w else 0
    crit_drop = base - reader_score(writers) if n_w else 0.0
    print(f"  n_w {n_w}  k*_eff (joint greedy) {k_eff}  width-1 {drop[0] / drop.max():.3f}  "
          f"overlap top-n_w {overlap}/{n_w}  criterion-set drop {crit_drop:.3f} "
          f"vs greedy max {drop.max():.3f}", flush=True)

    path = f"results/m5y_{model_name.replace('/', '_')}.npz"
    save_artifact(
        path,
        {
            "base_score": np.array([base]),
            "greedy_order": np.array(chosen, dtype=np.int64),
            "width_drop": drop,
            "criterion_writers": np.array(writers, dtype=np.int64).reshape(-1, 2),
            "criterion_drop": np.array([crit_drop]),
            "k_eff": np.array([k_eff]),
            "n_w": np.array([n_w]),
            "overlap": np.array([overlap]),
        },
        build_manifest(model_name, SEED, extra={
            "milestone": "5Y",
            "experiment": "m5y_joint_greedy",
            "reader": f"L{reader[0]}H{reader[1]}",
            "selection": "greedy forward: add the upstream head maximizing the JOINT "
                         "mean-ablation drop in reader induction score; no role criterion",
            "k_max": k_max,
            "tau": TAU,
        }),
    )
    print(f"artifact written: {path}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: m5y_joint_greedy.py <transformer_lens_model_name>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
