#!/usr/bin/env python3
"""Rank-K SUBSPACE STEERING across encoders — does matching intervention rank to k* break the
distinctness-vs-causal-potency dissociation, and is the law universal across training objectives?
(Local / MPS. DINOv2 self-distill · CLIP contrastive · SigLIP sigmoid.)

Generalize the rank-1 CAV to a rank-K subspace SWAP over ALL tokens:
    h' = h + P_K (t* - h),   P_K = U_K U_K^T   (top-K attribute directions = the corr-ranked
                                                 components that define k*),  t* = target prototype.
Metric: fraction of source->target readout gap CLOSED, + object-shape identity preserved, vs K.
Prediction: success peaks at K ~= k* (small for size, large for orientation); identity holds <= k*.

    .venv-steer/bin/python code/subspace_steer.py \
        --size-emb results/embeddings --size-data data/size_v1 \
        --orient-emb results/embeddings_orient --orient-data data/orient_v1 \
        --out results/metrics/subspace_steer_all.json
"""
import argparse, gc, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoModel

SIZE = 224
L_INJ, L_READ = 11, 12
NSRC = 40
RANKS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

# repo, mean, std, pool, bank-subdir
MODELS = {
    "dinov2-base": dict(repo="facebook/dinov2-base", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                        pool="cls", tower="self"),
    "clip-base":   dict(repo="openai/clip-vit-base-patch16", mean=[0.4814, 0.4578, 0.4082],
                        std=[0.2686, 0.2613, 0.2758], pool="cls", tower="vision_model"),
    "siglip-base": dict(repo="google/siglip-base-patch16-224", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                        pool="mean", tower="vision_model"),
}


def dev():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(pil, mean, std, device):
    m, s = np.array(mean, np.float32), np.array(std, np.float32)
    a = [((np.asarray(im.convert("RGB").resize((SIZE, SIZE), PILImage.BICUBIC), np.float32) / 255 - m) / s).transpose(2, 0, 1) for im in pil]
    return torch.from_numpy(np.stack(a)).float().to(device)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den < 1e-9 else float((a * b).sum() / den)


def attr_subspace(X, targets, n_pc=120):
    mu = X.mean(0); Xc = X - mu
    n_pc = min(n_pc, Xc.shape[0] - 1, Xc.shape[1])
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_pc].T                                  # (H, n_pc) orthonormal
    Z = Xc @ W
    rel = np.zeros(n_pc)
    for t in targets:
        rel += np.array([corr(Z[:, j], t) ** 2 for j in range(n_pc)])
    order = np.argsort(-rel); Wo = W[:, order]; relo = rel[order]; tot = relo.sum()
    kstar = n_pc if tot <= 1e-9 else int(np.argmax(np.cumsum(relo) / tot >= 0.95)) + 1
    return Wo.astype(np.float32), int(kstar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-emb", required=True, type=Path)
    ap.add_argument("--size-data", required=True, type=Path)
    ap.add_argument("--orient-emb", required=True, type=Path)
    ap.add_argument("--orient-data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    device = dev(); torch.set_grad_enabled(False)
    print(f"device={device}")
    out = {"ranks": RANKS, "l_inj": L_INJ, "l_read": L_READ, "models": {}}

    for mname, cfg in MODELS.items():
        print(f"\n===== {mname} ({cfg['repo']}) =====")
        model = AutoModel.from_pretrained(cfg["repo"]).to(device).eval()
        tower = model if cfg["tower"] == "self" else getattr(model, cfg["tower"])
        layers = tower.encoder.layer if hasattr(tower.encoder, "layer") else tower.encoder.layers
        inj = layers[L_INJ - 1]
        state = {"on": False, "P": None, "t": None}

        def hook(mod, i, o):
            if not state["on"]:
                return o
            hs = (o[0] if isinstance(o, tuple) else o).clone()
            diff = state["t"][None, None, :] - hs            # (B,T,H) toward target prototype
            hs = hs + diff @ state["P"]                      # edit ALL tokens, confined to K-subspace
            return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
        h = inj.register_forward_hook(hook)

        def pool(hsq):                                       # (B,T,H) -> (B,H)
            return hsq[:, 0, :] if cfg["pool"] == "cls" else hsq.mean(1)

        def run_attr(emb_dir, data_dir, attr):
            meta = sorted(json.load(open(emb_dir / "meta.json")), key=lambda r: r["global_idx"])
            bank = torch.load(emb_dir / mname / "embeddings.pt").numpy()   # (N, L+1, H) pooled per cfg
            Xinj, Xread = bank[:, L_INJ, :], bank[:, L_READ, :]
            if attr == "size":
                y = np.array([r["scale"] for r in meta], np.float32)
                tgt, src, targets = y == y.max(), y == y.min(), [y]
            else:
                th = np.radians(np.array([r["theta_deg"] for r in meta], np.float32))
                tgt, src, targets = np.cos(th) < -0.5, np.cos(th) > 0.5, [np.sin(th), np.cos(th)]
            W, kstar = attr_subspace(Xinj, targets)
            t_star = torch.tensor(Xinj[tgt].mean(0), dtype=torch.float32, device=device)
            axis = Xread[tgt].mean(0) - Xread[src].mean(0); axis = axis / (np.linalg.norm(axis) + 1e-9)
            pf = float((Xread[src] @ axis).mean()); pb = float((Xread[tgt] @ axis).mean()); gap = pb - pf
            shapes = np.array([r.get("asset_id") or r.get("category") or "?" for r in meta])
            classes = sorted(set(shapes.tolist()))
            C = np.stack([Xread[shapes == c].mean(0) for c in classes])
            def pshape(H):
                return np.array(classes)[(((H[:, None, :] - C[None]) ** 2).sum(-1)).argmin(1)]
            si = np.where(src)[0]; rng = np.random.default_rng(0)
            si = rng.choice(si, size=min(NSRC, len(si)), replace=False)
            ssh = shapes[si]
            pv = preprocess([PILImage.open(data_dir / meta[i]["file"]) for i in si],
                            cfg["mean"], cfg["std"], device)
            print(f"  [{attr}] k*={kstar} gap={gap:.2f}")
            closed, idk = [], []
            for K in RANKS:
                Uk = torch.tensor(W[:, :K], dtype=torch.float32, device=device)
                state["P"], state["t"], state["on"] = Uk @ Uk.t(), t_star, True
                hsr = tower(pixel_values=pv, output_hidden_states=True).hidden_states[L_READ]
                state["on"] = False
                hp = pool(hsr).float().cpu().numpy()
                closed.append(float(((hp @ axis).mean() - pf) / (gap + 1e-9)))
                idk.append(float((pshape(hp) == ssh).mean()))
            print(f"    closed: " + " ".join(f"{x:.2f}" for x in closed))
            print(f"    id    : " + " ".join(f"{x:.2f}" for x in idk))
            return dict(kstar=kstar, closed=closed, id_kept=idk)

        out["models"][mname] = {
            "size": run_attr(args.size_emb, args.size_data, "size"),
            "orientation": run_attr(args.orient_emb, args.orient_data, "orientation"),
        }
        h.remove(); del model, tower; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
