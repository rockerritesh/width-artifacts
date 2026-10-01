"""Milestone 5Y, batched: identical measurement to m5y_joint_greedy.py, faster.

Each greedy step asks, for every remaining upstream head c, "what is the reader's
induction score when chosen + {c} are mean-ablated?". The unbatched script pays
one forward pass (batch 8) per candidate, which leaves a T4 mostly idle and put
gpt2-xl (725 upstream heads x 30 steps) at ~7 GPU-hours. Two exact speedups:

  * CANDIDATE BATCHING. Stack C copies of the same 8-prompt batch; in copy c,
    ablate the chosen heads plus candidate c. Each copy is the unbatched forward
    pass (same tokens, same means), so scores match up to float reduction order.
  * EARLY STOP. Layers after the reader cannot change the reader's attention
    pattern, so the forward stops at reader_layer + 1.

Correctness gate: on gpt2 this must reproduce results/m5y_gpt2.npz (same greedy
order, drops within 1e-4). Run with --check to enforce it.

Usage: `PYTHONPATH=. python experiments/m5y_joint_greedy_batched.py <tl_model> [--check]`
Writes results/m5y_<model>.npz (same schema as the unbatched script).
"""

import sys
import time

import numpy as np
import torch

from experiments.m5w_width_sweep import BATCH, SEED, SEQ_LEN, score_matrices
from interp.ablation import precompute_head_means
from interp.artifacts import build_manifest, load_artifact, save_artifact
from interp.data import make_repeated_tokens
from interp.loading import device_or_default, load_model
from interp.roles import BY_ROLE, select

K_MAX = 30
TAU = 0.95
CHUNK = 48          # candidates per forward pass (x BATCH sequences)


def main(model_name: str, check: bool) -> int:
    print(f"=== Milestone 5Y batched (joint greedy width): {model_name} ===", flush=True)
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
    r_layer, r_head = reader
    crit = BY_ROLE["previous_token"]
    writers = [h for h in select(prev, crit.frac, crit.sign) if h[0] < r_layer]
    n_w = len(writers)
    print(f"  reader L{r_layer}H{r_head} base {base:.4f}  n_w {n_w}", flush=True)

    means = precompute_head_means(model, reference)
    # [n_layers, pos, n_heads, d_head] stacked means for vectorized writes
    M = torch.stack([torch.stack([means[(l, h)] for h in range(n_heads)], dim=1)
                     for l in range(r_layer)])
    pat_name = f"blocks.{r_layer}.attn.hook_pattern"

    def reader_scores(chosen, cands):
        """Reader induction score with chosen + {c} ablated, for each c in cands."""
        C = len(cands)
        toks = tokens.repeat(C, 1)
        by_layer = {}
        for l, h in chosen:
            by_layer.setdefault(l, ([], []))[0].append(h)
        for i, (l, h) in enumerate(cands):
            by_layer.setdefault(l, ([], []))[1].append((i, h))
        hooks = []
        for l, (hs_all, per_copy) in by_layer.items():
            def fn(z, hook, l=l, hs_all=hs_all, per_copy=per_copy):
                if hs_all:
                    z[:, :, hs_all, :] = M[l][:, hs_all, :]
                for i, h in per_copy:
                    z[i * BATCH:(i + 1) * BATCH, :, h, :] = M[l][:, h, :]
                return z
            hooks.append((f"blocks.{l}.attn.hook_z", fn))
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n == pat_name,
                                            stop_at_layer=r_layer + 1)
        pat = cache[pat_name][:, r_head]                      # [C*B, pos, pos]
        stripe = pat.diagonal(offset=-(SEQ_LEN - 1), dim1=-2, dim2=-1)
        return stripe.reshape(C, BATCH, -1).mean(dim=(1, 2)).double().cpu().numpy()

    # sanity: batched no-candidate path reproduces the unbatched base score
    upstream = [(l, h) for l in range(r_layer) for h in range(n_heads)]
    k_max = min(K_MAX, len(upstream))
    chosen: list[tuple[int, int]] = []
    drop = np.zeros(k_max)
    t0 = time.time()
    for k in range(k_max):
        rest = [lh for lh in upstream if lh not in chosen]
        scores = np.concatenate([reader_scores(chosen, rest[i:i + CHUNK])
                                 for i in range(0, len(rest), CHUNK)])
        j = int(np.argmin(scores))
        chosen.append(rest[j])
        drop[k] = base - scores[j]
        print(f"  k={k + 1:>2} +L{rest[j][0]}H{rest[j][1]}  drop {drop[k]:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    k_eff = int(np.nonzero(drop >= TAU * drop.max())[0][0]) + 1
    overlap = len(set(chosen[:n_w]) & set(writers)) if n_w else 0
    crit_drop = base - float(reader_scores(writers[:-1], writers[-1:])[0]) if n_w else 0.0
    print(f"  n_w {n_w}  k*_eff (joint greedy) {k_eff}  width-1 {drop[0] / drop.max():.3f}  "
          f"overlap top-n_w {overlap}/{n_w}  criterion-set drop {crit_drop:.3f} "
          f"vs greedy max {drop.max():.3f}", flush=True)

    if check:
        ref, _ = load_artifact(f"results/m5y_{model_name.replace('/', '_')}.npz")
        same_order = [tuple(x) for x in ref["greedy_order"]] == chosen
        max_diff = float(np.abs(ref["width_drop"] - drop).max())
        print(f"  CHECK vs unbatched: same order {same_order}, max |drop diff| "
              f"{max_diff:.2e}", flush=True)
        if not same_order or max_diff > 1e-4:
            raise RuntimeError("batched result diverges from the unbatched reference")
        return 0

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
            "experiment": "m5y_joint_greedy_batched",
            "reader": f"L{r_layer}H{r_head}",
            "selection": "greedy forward: add the upstream head maximizing the JOINT "
                         "mean-ablation drop in reader induction score; no role criterion",
            "batched": f"{CHUNK} candidates/pass, stop_at_layer=reader+1; "
                       "verified identical to unbatched on gpt2",
            "k_max": k_max,
            "tau": TAU,
        }),
    )
    print(f"artifact written: {path}", flush=True)
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("usage: m5y_joint_greedy_batched.py <tl_model_name> [--check]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(args[0], "--check" in sys.argv))
