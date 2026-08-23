#!/usr/bin/env python3
"""E2 (DECISION GATE) — depth x rank steering surface, 3 base encoders x 2 attributes.

Question that gates the paper's claim wording: does DINOv2-size fail to close at
K~=k* because L11 is the wrong injection depth (its best-decodability layer is L6),
or is size readout-limited in these encoders regardless of depth?

Protocol = subspace_steer.py exactly (rank-K swap toward target prototype, all
tokens), extended over injection depth. Each steered forward reports the gap
closed at BOTH readouts (same pass):
  local: hidden_states[L_inj + 1]   (matches the saved L11->L12 sweep)
  final: hidden_states[12]          (does the edit survive to the end?)
Identity (nearest shape centroid) measured at the final layer.

Runs on MPS. ~660 steered forwards ~= 5-8 min compute.
Writes results/e2_depth_rank.json incrementally (per model) so partial runs survive.
"""
import gc, json
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage
from transformers import AutoModel

VROOT = Path("/path/to/Documents/workspace/vision-model-steering")
OUT = Path("/path/to/Documents/workspace/interp4discovery-neurips2026/results/e2_depth_rank.json")

SIZE = 224
NSRC = 40
RANKS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
DEPTHS = list(range(1, 12))          # L_inj; hook layers[L_inj-1]; read L_inj+1 and 12
L_FINAL = 12
N_PC = 120

MODELS = {
    "dinov2-base": dict(repo="facebook/dinov2-base", mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225], pool="cls", tower="self"),
    "clip-base": dict(repo="openai/clip-vit-base-patch16", mean=[0.4814, 0.4578, 0.4082],
                      std=[0.2686, 0.2613, 0.2758], pool="cls", tower="vision_model"),
    "siglip-base": dict(repo="google/siglip-base-patch16-224", mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5], pool="mean", tower="vision_model"),
}


def dev():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def preprocess(pil, mean, std, device):
    m, s = np.array(mean, np.float32), np.array(std, np.float32)
    a = [((np.asarray(im.convert("RGB").resize((SIZE, SIZE), PILImage.BICUBIC),
                      np.float32) / 255 - m) / s).transpose(2, 0, 1) for im in pil]
    return torch.from_numpy(np.stack(a)).float().to(device)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den < 1e-9 else float((a * b).sum() / den)


def attr_subspace(X, targets, n_pc=N_PC):
    mu = X.mean(0); Xc = X - mu
    n_pc = min(n_pc, Xc.shape[0] - 1, Xc.shape[1])
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_pc].T
    Z = Xc @ W
    rel = np.zeros(n_pc)
    for t in targets:
        rel += np.array([corr(Z[:, j], t) ** 2 for j in range(n_pc)])
    order = np.argsort(-rel); Wo = W[:, order]; relo = rel[order]; tot = relo.sum()
    kstar = n_pc if tot <= 1e-9 else int(np.argmax(np.cumsum(relo) / tot >= 0.95)) + 1
    return Wo.astype(np.float32), int(kstar)


def main():
    device = dev()
    torch.set_grad_enabled(False)
    print(f"device={device}")
    out = {"ranks": RANKS, "depths": DEPTHS, "l_final": L_FINAL, "nsrc": NSRC,
           "models": {}}
    if OUT.exists():                                   # resume per-model
        out = json.load(open(OUT))
    ATTRS = [("size", VROOT / "results/embeddings", VROOT / "data/size_v1"),
             ("orientation", VROOT / "results/embeddings_orient", VROOT / "data/orient_v1")]

    for mname, cfg in MODELS.items():
        if mname in out["models"]:
            print(f"skip {mname} (already in output)")
            continue
        print(f"\n===== {mname} =====", flush=True)
        model = AutoModel.from_pretrained(cfg["repo"]).to(device).eval()
        tower = model if cfg["tower"] == "self" else getattr(model, cfg["tower"])
        layers = tower.encoder.layer if hasattr(tower.encoder, "layer") else tower.encoder.layers
        state = {"on": False, "P": None, "t": None}
        hooks = []

        def make_hook():
            def hook(mod, i, o):
                if not state["on"]:
                    return o
                hs = (o[0] if isinstance(o, tuple) else o).clone()
                hs = hs + (state["t"][None, None, :] - hs) @ state["P"]
                return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
            return hook

        def pool(hsq):
            return hsq[:, 0, :] if cfg["pool"] == "cls" else hsq.mean(1)

        mres = {}
        for attr, emb_dir, data_dir in ATTRS:
            meta = sorted(json.load(open(emb_dir / "meta.json")),
                          key=lambda r: r["global_idx"])
            bank = torch.load(emb_dir / mname / "embeddings.pt").numpy()
            if attr == "size":
                y = np.array([r["scale"] for r in meta], np.float32)
                tgt, src, targets = y == y.max(), y == y.min(), [y]
            else:
                th = np.radians(np.array([r["theta_deg"] for r in meta], np.float32))
                tgt, src = np.cos(th) < -0.5, np.cos(th) > 0.5
                targets = [np.sin(th), np.cos(th)]
            shapes = np.array([r.get("asset_id") or r.get("category") or "?" for r in meta])
            classes = sorted(set(shapes.tolist()))
            si = np.where(src)[0]
            si = np.random.default_rng(0).choice(si, size=min(NSRC, len(si)), replace=False)
            ssh = shapes[si]
            pv = preprocess([PILImage.open(data_dir / meta[i]["file"]) for i in si],
                            cfg["mean"], cfg["std"], device)

            # readout axes + shape centroids, precomputed per layer from the bank
            axes, pfs, gaps, cents = {}, {}, {}, {}
            for L in set([d + 1 for d in DEPTHS] + [L_FINAL]):
                Xr = bank[:, L, :]
                ax = Xr[tgt].mean(0) - Xr[src].mean(0)
                ax = ax / (np.linalg.norm(ax) + 1e-9)
                pf = float((Xr[src] @ ax).mean()); pb = float((Xr[tgt] @ ax).mean())
                axes[L], pfs[L], gaps[L] = ax, pf, pb - pf
                cents[L] = np.stack([Xr[shapes == c].mean(0) for c in classes])

            attr_res = {"depths": {}}
            for d in DEPTHS:
                Xinj = bank[:, d, :]
                W, kstar = attr_subspace(Xinj, targets)
                t_star = torch.tensor(Xinj[tgt].mean(0), dtype=torch.float32, device=device)
                h = layers[d - 1].register_forward_hook(make_hook())
                cl_loc, cl_fin, idk = [], [], []
                for K in RANKS:
                    Uk = torch.tensor(W[:, :K], dtype=torch.float32, device=device)
                    state["P"], state["t"], state["on"] = Uk @ Uk.t(), t_star, True
                    hs = tower(pixel_values=pv, output_hidden_states=True).hidden_states
                    state["on"] = False
                    hl = pool(hs[d + 1]).float().cpu().numpy()
                    hf = pool(hs[L_FINAL]).float().cpu().numpy()
                    cl_loc.append(round(float(((hl @ axes[d + 1]).mean() - pfs[d + 1])
                                              / (gaps[d + 1] + 1e-9)), 4))
                    cl_fin.append(round(float(((hf @ axes[L_FINAL]).mean() - pfs[L_FINAL])
                                              / (gaps[L_FINAL] + 1e-9)), 4))
                    pred = np.array(classes)[
                        (((hf[:, None, :] - cents[L_FINAL][None]) ** 2).sum(-1)).argmin(1)]
                    idk.append(round(float((pred == ssh).mean()), 4))
                h.remove()
                attr_res["depths"][d] = dict(kstar=kstar, closed_local=cl_loc,
                                             closed_final=cl_fin, id_kept=idk)
                print(f"  [{attr}] L{d:>2} k*={kstar:>3} "
                      f"loc@k*~{np.interp(kstar, RANKS, cl_loc):.2f} "
                      f"fin@k*~{np.interp(kstar, RANKS, cl_fin):.2f} "
                      f"best_fin={max(cl_fin):.2f}@K={RANKS[int(np.argmax(cl_fin))]}",
                      flush=True)
            mres[attr] = attr_res

        out["models"][mname] = mres
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"  saved {mname} -> {OUT}", flush=True)
        del model, tower
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    print("\nDONE")


if __name__ == "__main__":
    main()
