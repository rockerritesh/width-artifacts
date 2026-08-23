#!/usr/bin/env python3
"""Experiment E — does PaliGemma's LM read object size from SPATIAL layout
(patch extent) rather than from the pooled subspace we steered?

Design (2 arms + baseline, tower layer 24, same site as pg_subspace):
  field:    h'_p = h_p + beta * (T_large[p] - h_p)   per patch position p,
            where T_large is the large-class MEAN TOKEN FIELD (256, 1152).
            Preserves spatial structure of "a large object was here".
  shuffled: same swap, but T_large's patch positions randomly permuted.
            IDENTICAL pooled content, spatial structure destroyed.
If the consumer reads extent/layout: field moves p("large"), shuffled does
not. If only pooled content matters: the two match (and both stay flat,
like the pooled-prototype pushes already measured).

Runs on the T4 (bf16). Reads Kubric size images + meta.json (pushed).
Writes eE_patch_field.json.
"""
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/size_v1"
META = HERE / "meta.json"
OUT = HERE / "eE_patch_field.json"

MODEL = "google/paligemma-3b-mix-224"
L_INJ = 24
BETAS = [0.25, 0.5, 1.0]
NSRC = 24
PROMPT = "answer en is the object small or large?"

device = "cuda"
dtype = torch.bfloat16
proc = AutoProcessor.from_pretrained(MODEL)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL, dtype=dtype, device_map="cuda", low_cpu_mem_usage=True).eval()
torch.set_grad_enabled(False)

meta = sorted(json.load(open(META)), key=lambda r: r["global_idx"])
scales = np.array([r["scale"] for r in meta], np.float32)
files = [r["file"] for r in meta]
big, small = scales == scales.max(), scales == scales.min()
print(f"{len(meta)} images; {big.sum()} large, {small.sum()} small", flush=True)

def _find_layers(m):
    for path in ("vision_tower.vision_model.encoder.layers",
                 "model.vision_tower.vision_model.encoder.layers",
                 "model.vision_tower.encoder.layers",
                 "model.vision_tower.model.encoder.layers",
                 "vision_tower.encoder.layers"):
        obj = m
        for att in path.split("."):
            obj = getattr(obj, att, None)
            if obj is None:
                break
        if obj is not None:
            print(f"encoder layers at: {path} ({len(obj)} blocks)", flush=True)
            return obj
    raise AttributeError("could not locate vision encoder layers")


layers = _find_layers(model)
state = {"mode": None, "beta": 0.0, "field": None, "cap": []}


def hook(mod, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    if state["mode"] == "capture":
        state["cap"].append(hs.detach().float().cpu())
    elif state["mode"] in ("field", "shuffled"):
        f = state["field"].to(hs.device, hs.dtype)
        hs = hs + state["beta"] * (f[None, :, :] - hs)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    return out


h = layers[L_INJ].register_forward_hook(hook)


def pixel_batch(idxs):
    ims = [Image.open(DATA / files[i]).convert("RGB") for i in idxs]
    return proc(images=ims, text=[PROMPT] * len(ims),
                return_tensors="pt", padding=True).to(device)


def p_large(idxs):
    """P(first generated token == ' large') vs ' small', renormalized."""
    tl = proc.tokenizer(" large", add_special_tokens=False).input_ids[0]
    ts = proc.tokenizer(" small", add_special_tokens=False).input_ids[0]
    ps = []
    for i in range(0, len(idxs), 8):
        b = pixel_batch(idxs[i:i + 8])
        out = model(**b)
        logits = out.logits[:, -1, :].float()
        pr = torch.softmax(logits, -1)
        pl, psm = pr[:, tl], pr[:, ts]
        ps += (pl / (pl + psm + 1e-9)).tolist()
    return float(np.mean(ps))


# ---- capture the large-class mean token field at L24 ----
state["mode"] = "capture"
big_idx = np.where(big)[0]
for i in range(0, len(big_idx), 8):
    _ = model(**pixel_batch(big_idx[i:i + 8]))
T_large = torch.cat(state["cap"]).mean(0)          # (n_tokens, 1152)
state["cap"] = []
print(f"large-class field: {tuple(T_large.shape)}", flush=True)

src = np.where(small)[0][:NSRC].tolist()
state["mode"] = None
base = p_large(src)
ref_large = p_large(big_idx[:NSRC].tolist())
print(f"baseline p(large): small-objects {base:.3f}, large-objects {ref_large:.3f}",
      flush=True)

res = dict(model=MODEL, l_inj=L_INJ, n_src=len(src), prompt=PROMPT,
           p_base_small=round(base, 4), p_base_large=round(ref_large, 4),
           field={}, shuffled={})
rng = np.random.default_rng(0)
perm = rng.permutation(T_large.shape[0])
for beta in BETAS:
    state.update(mode="field", beta=beta, field=T_large)
    pf = p_large(src)
    state.update(mode="shuffled", field=T_large[perm])
    psh = p_large(src)
    state["mode"] = None
    res["field"][beta] = round(pf, 4)
    res["shuffled"][beta] = round(psh, 4)
    print(f"beta={beta}: field {pf:.3f} | shuffled {psh:.3f}", flush=True)

h.remove()
json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT}", flush=True)
