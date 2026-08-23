#!/usr/bin/env python3
"""E5/E6 — random k-subset vs contiguous k-band layer removal (Fig 3b), plus the
strided-band (width vs span) and band-position controls. Llama-3.1-8B base on NDIF
remote — laptop orchestrates, no local GPU.

Metric (matches metrics_remote.json): next-token KL(ablated||baseline) and top-1
agreement at the last prompt position, over 32 completion prompts, one batched
remote trace per condition.

Ablation of an ARBITRARY layer set S: for each removed L (ascending), rebind
  layers[L].output = output of the nearest kept layer j < L (embeddings if none).
The old contiguous-only shortcut (all removed layers -> layers[min-1].output)
is wrong for non-contiguous sets: it would also wipe the kept layers in between.

Arms:
  baseline                       1 trace
  contiguous k in KS             7   (start = max(0, 11 - k//2), the published band)
  random-region k x 5 seeds      35  (k layers sampled from 2..29 — excludes the two
                                      load-bearing early layers, as the contiguous
                                      mid-bands do; the clean contiguity control)
  random-anywhere k x 5 seeds    35  (k layers from 0..31 — the naive control)
  strided k=6 over span 5..16    1   (same SPAN as k=12 contiguous, half the WIDTH)
  slide: k=12 at starts 0..20    6   (band position control)
Total ~85 traces. Results written incrementally after every condition.
"""
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG

load_dotenv("/path/to/Documents/workspace/.env")
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

OUT = Path("/path/to/Documents/workspace/interp4discovery-neurips2026/results/e5_random_vs_contiguous.json")
MODEL = "meta-llama/Meta-Llama-3.1-8B"
KS = [2, 4, 6, 8, 10, 12, 14]
SEEDS = range(5)
REGION = list(range(2, 30))          # survivable-singles region

PROMPTS = [
    "The capital of France is", "The capital of Japan is",
    "The chemical symbol for gold is", "Two plus two equals",
    "The opposite of hot is", "The sun rises in the",
    "Water is composed of hydrogen and", "The first President of the United States was",
    "The largest planet in the solar system is", "Roses are red, violets are",
    "The speed of light is approximately", "The Eiffel Tower is located in the city of",
    "The boiling point of water in Celsius is", "The author of Romeo and Juliet is",
    "The currency of the United Kingdom is", "A triangle has this many sides:",
    "The largest ocean on Earth is the", "Photosynthesis converts sunlight into",
    "The freezing point of water in Fahrenheit is", "The square root of sixty-four is",
    "Mount Everest is located in the country of", "The human heart has this many chambers:",
    "The longest river in the world is the", "Lightning is followed by the sound of",
    "The primary language spoken in Brazil is", "An octagon has this many sides:",
    "The planet closest to the sun is", "Ice floats on water because it is less",
    "The capital of Italy is", "A decade is a period of this many years:",
    "The gas plants absorb from the air is called carbon", "The opposite of expensive is",
]


def val(p):
    return p.value if hasattr(p, "value") else p


model = LanguageModel(MODEL)
model.tokenizer.pad_token = model.tokenizer.eos_token
model.tokenizer.padding_side = "left"
NL = model.config.num_hidden_layers
print(f"model={MODEL} layers={NL} prompts={len(PROMPTS)}", flush=True)


def logits_last(removed):
    import time
    removed = sorted(set(removed or []))
    for attempt in range(6):
        if attempt:
            time.sleep(15 * attempt)
        try:
            with model.trace(PROMPTS, remote=True):
                if removed:
                    rem = set(removed)
                    srcs = {}                      # one .output access per source
                    for L in removed:              # (re-access after later layers
                        j = L - 1                  #  raises OutOfOrderError)
                        while j >= 0 and j in rem:
                            j -= 1
                        if j not in srcs:
                            srcs[j] = (model.model.embed_tokens.output if j < 0
                                       else model.model.layers[j].output)
                        model.model.layers[L].output = srcs[j]
                lg = model.output.logits[:, -1, :].save()
            return val(lg).float()
        except Exception as e:
            print(f"   retry {attempt + 1} ({type(e).__name__}: {str(e)[:90]})", flush=True)
    raise RuntimeError("remote call failed after 6 attempts")


print(">>> baseline", flush=True)
base = logits_last(None)
lpb = F.log_softmax(base, -1)


def metrics(removed):
    la = logits_last(removed)
    lpa = F.log_softmax(la, -1)
    kl = (lpa.exp() * (lpa - lpb)).sum(-1)
    top = (base.argmax(-1) == la.argmax(-1)).float()
    return round(kl.mean().item(), 4), round(top.mean().item(), 4)


res = {"model": MODEL, "n_prompts": len(PROMPTS), "metric": "next-token KL/top1, last position",
       "region": [REGION[0], REGION[-1]], "contiguous": [], "random_region": [],
       "random_anywhere": [], "strided": [], "slide": []}
if OUT.exists():
    res = json.load(open(OUT))


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)


def have(arm, **kw):
    return any(all(r.get(k2) == v2 for k2, v2 in kw.items()) for r in res[arm])


for k in KS:                                             # contiguous (published band)
    if have("contiguous", k=k):
        continue
    start = max(0, 11 - k // 2)
    idxs = list(range(start, start + k))
    kl, top = metrics(idxs)
    res["contiguous"].append(dict(k=k, idxs=idxs, kl=kl, top1=top)); save()
    print(f"contig  k={k:2d} {idxs[0]}..{idxs[-1]}  KL={kl:7.3f} top1={top:.3f}", flush=True)

for k in KS:                                             # random within region
    for seed in SEEDS:
        if have("random_region", k=k, seed=seed):
            continue
        idxs = sorted(random.Random(1000 + seed).sample(REGION, k))
        kl, top = metrics(idxs)
        res["random_region"].append(dict(k=k, seed=seed, idxs=idxs, kl=kl, top1=top)); save()
        print(f"rand-rg k={k:2d} s={seed}  KL={kl:7.3f} top1={top:.3f}  {idxs}", flush=True)

for k in KS:                                             # random anywhere
    for seed in SEEDS:
        if have("random_anywhere", k=k, seed=seed):
            continue
        idxs = sorted(random.Random(2000 + seed).sample(range(NL), k))
        kl, top = metrics(idxs)
        res["random_anywhere"].append(dict(k=k, seed=seed, idxs=idxs, kl=kl, top1=top)); save()
        print(f"rand-an k={k:2d} s={seed}  KL={kl:7.3f} top1={top:.3f}  {idxs}", flush=True)

if not res["strided"]:                                   # width vs span
    idxs = [5, 7, 9, 11, 13, 15]
    kl, top = metrics(idxs)
    res["strided"].append(dict(k=6, span=[5, 16], idxs=idxs, kl=kl, top1=top)); save()
    print(f"strided k=6 span 5..16  KL={kl:7.3f} top1={top:.3f}", flush=True)

for start in [0, 4, 8, 12, 16, 20]:                      # band position, k=12
    if have("slide", start=start):
        continue
    idxs = list(range(start, start + 12))
    kl, top = metrics(idxs)
    res["slide"].append(dict(start=start, k=12, idxs=idxs, kl=kl, top1=top)); save()
    print(f"slide   start={start:2d}  KL={kl:7.3f} top1={top:.3f}", flush=True)

print(">>> E5/E6 COMPLETE", flush=True)
