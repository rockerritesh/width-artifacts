#!/usr/bin/env python3
"""
Layer-ablation sweep on FULL-PRECISION Llama-3.1-8B (base) via NDIF remote (nnsight).
Runs on the laptop (orchestration only); the model executes on NDIF's servers.

Metric: next-token distribution at the last prompt position, ablated vs baseline —
mean KL(ablated‖baseline) and top-1 agreement over a set of completion prompts.
Ablation = make a decoder layer identity: layer.output[0][:] = layer.input.

Outputs: blog/figures/remote_sweep.png, remote_cumulative.png, metrics_remote.json
"""
import os, json, statistics as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG

load_dotenv()
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])   # stays in-process
def val(p): return p.value if hasattr(p, "value") else p

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "blog", "figures")
os.makedirs(OUT, exist_ok=True)
CRIMSON, BLUE, SURFACE, CONCEPT, OUTPUT = "#c0392b", "#2f6fb0", "#eaf0ea", "#fbe7c8", "#e4edf8"

MODEL = "meta-llama/Meta-Llama-3.1-8B"    # base 8B is the pinned/hosted model on NDIF
model = LanguageModel(MODEL)
model.tokenizer.pad_token = model.tokenizer.eos_token
model.tokenizer.padding_side = "left"
NL = model.config.num_hidden_layers
print(f"model={MODEL}  layers={NL}", flush=True)

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The chemical symbol for gold is",
    "Two plus two equals",
    "The opposite of hot is",
    "The sun rises in the",
    "Water is composed of hydrogen and",
    "The first President of the United States was",
    "The largest planet in the solar system is",
    "Roses are red, violets are",
    "The speed of light is approximately",
    "The Eiffel Tower is located in the city of",
]

def logits_last(layer_idxs):
    """One remote forward over the batch; return last-position logits [B, vocab].
    Ablation = identity layer via REBIND: a layer's input is the previous layer's
    output (residual stream), so set layers[L].output = layers[start-1].output
    (embeddings for start==0). In-place [:]= is not captured remotely; rebind is."""
    for attempt in range(2):
        try:
            with model.trace(PROMPTS, remote=True):
                if layer_idxs:
                    start = min(layer_idxs)
                    entry = (model.model.embed_tokens.output if start == 0
                             else model.model.layers[start - 1].output)
                    for L in layer_idxs:
                        model.model.layers[L].output = entry
                lg = model.output.logits[:, -1, :].save()
            return val(lg).float()
        except Exception as e:
            print(f"   retry {attempt+1} ({type(e).__name__}: {str(e)[:80]})", flush=True)
    raise RuntimeError("remote call failed")

print(">>> baseline", flush=True)
base = logits_last(None)

def metrics(layer_idxs):
    la = logits_last(layer_idxs)
    lpb = F.log_softmax(base, -1); lpa = F.log_softmax(la, -1)
    kl = (lpa.exp() * (lpa - lpb)).sum(-1)
    top = (base.argmax(-1) == la.argmax(-1)).float()
    return kl.mean().item(), top.mean().item()

# --- single-layer sweep ---
print(">>> single-layer sweep (remote)", flush=True)
sweep = []
for L in range(NL):
    try:
        kl, top = metrics([L])
        sweep.append({"layer": L, "kl": kl, "top1": top})
        print(f"   L{L:2d}: KL={kl:.3f} top1={top:.2f}", flush=True)
    except Exception as e:
        print(f"   L{L:2d}: SKIP ({type(e).__name__})", flush=True)

# --- cumulative middle-band ---
print(">>> cumulative middle-band (remote)", flush=True)
KS = [0, 1, 2, 4, 6, 8, 10, 12, 14]
cum = []
for k in KS:
    start = max(0, 11 - k // 2); idxs = list(range(start, start + k))
    try:
        kl, top = metrics(idxs if k else None)
        cum.append({"k": k, "kl": kl, "top1": top})
        print(f"   k={k:2d}: KL={kl:.2f} top1={top:.2f}", flush=True)
    except Exception as e:
        print(f"   k={k:2d}: SKIP ({type(e).__name__})", flush=True)

# --- try to overlay our local 4-bit sweep for comparison ---
local = None
lp = os.path.join(OUT, "metrics_100.json")
if os.path.exists(lp):
    ld = json.load(open(lp))
    local = {s["layer"]: s["top1_mean"] for s in ld.get("sweep", [])}

# --- plots ---
xs = [s["layer"] for s in sweep]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
for ax in (a1, a2):
    ax.axvspan(-0.5, 4.5, color=SURFACE); ax.axvspan(4.5, 16.5, color=CONCEPT); ax.axvspan(16.5, NL - 0.5, color=OUTPUT)
a1.plot(xs, [s["kl"] for s in sweep], "o-", color=CRIMSON, lw=2)
a1.set_yscale("log"); a1.set_ylabel("mean KL (log)")
a1.set_title(f"Full-precision {MODEL} via NDIF remote — single-layer ablation (next-token metric)")
a2.plot(xs, [s["top1"] * 100 for s in sweep], "s-", color=BLUE, lw=2, label="remote fp16 base-8B (this run)")
if local:
    a2.plot(list(local.keys()), [v * 100 for v in local.values()], "^--", color="#888", lw=1.5,
            label="local 4-bit Instruct-8B (Exp 5, continuation metric)")
    a2.legend(fontsize=8, loc="lower center")
a2.set_ylabel("top-1 agreement %"); a2.set_xlabel("decoder layer removed (0..31)"); a2.set_ylim(0, 103)
fig.tight_layout(); fig.savefig(f"{OUT}/remote_sweep.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(">>> saved remote_sweep.png", flush=True)

ks = [c["k"] for c in cum]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(ks, [c["kl"] for c in cum], "o-", color=CRIMSON, lw=2)
ax.set_yscale("symlog"); ax.set_ylabel("mean KL (symlog)", color=CRIMSON)
ax.set_xlabel("# contiguous middle layers removed (centered ~L11)")
axb = ax.twinx(); axb.plot(ks, [c["top1"] * 100 for c in cum], "s--", color=BLUE, lw=1.8)
axb.set_ylabel("top-1 agreement %", color=BLUE); axb.set_ylim(0, 103)
ax.set_title(f"Full-precision {MODEL} via NDIF — removing the middle band")
fig.tight_layout(); fig.savefig(f"{OUT}/remote_cumulative.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(">>> saved remote_cumulative.png", flush=True)

json.dump({"model": MODEL, "backend": "NDIF remote (nnsight)", "n_prompts": len(PROMPTS),
           "metric": "next-token KL/top1 at last position", "sweep": sweep, "cumulative": cum},
          open(f"{OUT}/metrics_remote.json", "w"), indent=2)
print(">>> REMOTE SWEEP COMPLETE", flush=True)
