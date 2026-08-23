#!/usr/bin/env python3
"""E10 (referee finding 10) — matched random-SUBSPACE control for the DINOv2
orientation headline curve.

Objection to close: "any 53-dimensional shove toward the target prototype
works." Control: replace the attribute subspace (top-K correlation-ranked
PCs) with a random K-dimensional orthonormal subspace, same K, same swap
h' = h + P_K(t* - h), same injection/readout (L11 -> L12). If the lift is
generic, random matches attribute; if it is the attribute directions,
random stays near zero.

5 seeds x K in {13, 55}. MPS, ~2 min. Writes results/e10_random_subspace.json.
"""
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage
from transformers import AutoModel

VROOT = Path("/path/to/Documents/workspace/vision-model-steering")
OUT = Path("/path/to/Documents/workspace/interp4discovery-neurips2026/results/e10_random_subspace.json")

SIZE, NSRC, L_INJ, L_READ = 224, 40, 11, 12
KS = [13, 55]
SEEDS = range(5)
CFG = dict(repo="facebook/dinov2-base", mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225])

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_grad_enabled(False)

emb_dir = VROOT / "results/embeddings_orient"
meta = sorted(json.load(open(emb_dir / "meta.json")), key=lambda r: r["global_idx"])
bank = torch.load(emb_dir / "dinov2-base/embeddings.pt").numpy()
th = np.radians(np.array([r["theta_deg"] for r in meta], np.float32))
tgt, src = np.cos(th) < -0.5, np.cos(th) > 0.5

Xinj, Xread = bank[:, L_INJ, :], bank[:, L_READ, :]
t_star = torch.tensor(Xinj[tgt].mean(0), dtype=torch.float32, device=device)
axis = Xread[tgt].mean(0) - Xread[src].mean(0)
axis = axis / (np.linalg.norm(axis) + 1e-9)
pf = float((Xread[src] @ axis).mean())
gap = float((Xread[tgt] @ axis).mean()) - pf

si = np.random.default_rng(0).choice(np.where(src)[0], size=NSRC, replace=False)
m, sd = np.array(CFG["mean"], np.float32), np.array(CFG["std"], np.float32)
pv = torch.from_numpy(np.stack(
    [((np.asarray(PILImage.open(VROOT / "data/orient_v1" / meta[i]["file"])
                  .convert("RGB").resize((SIZE, SIZE), PILImage.BICUBIC),
       np.float32) / 255 - m) / sd).transpose(2, 0, 1) for i in si])).float().to(device)

model = AutoModel.from_pretrained(CFG["repo"]).to(device).eval()
layers = model.encoder.layer
state = {"on": False, "P": None}


def hook(mod, i, o):
    if not state["on"]:
        return o
    hs = (o[0] if isinstance(o, tuple) else o).clone()
    hs = hs + (t_star[None, None, :] - hs) @ state["P"]
    return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs


h = layers[L_INJ - 1].register_forward_hook(hook)


def closed_with(P):
    state["P"], state["on"] = P, True
    hs = model(pixel_values=pv, output_hidden_states=True).hidden_states[L_READ]
    state["on"] = False
    hp = hs[:, 0, :].float().cpu().numpy()
    return float(((hp @ axis).mean() - pf) / (gap + 1e-9))


res = {"model": "dinov2-base", "attr": "orientation", "l_inj": L_INJ,
       "l_read": L_READ, "random": {}, "note": "random orthonormal K-subspace, "
       "same swap toward target prototype as the attribute-subspace sweep"}
H = Xinj.shape[1]
for K in KS:
    vals = []
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(H, K, generator=g))
        Q = Q.to(device)
        vals.append(round(closed_with(Q @ Q.T), 4))
    res["random"][K] = dict(seeds=vals, mean=round(float(np.mean(vals)), 4),
                            sd=round(float(np.std(vals)), 4))
    print(f"K={K}: random-subspace closed = {np.mean(vals):.3f} ± {np.std(vals):.3f}  {vals}")

h.remove()
OUT.parent.mkdir(exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
print("(attribute-subspace reference: 0.22 at K=13, 0.96 at K=55)")
