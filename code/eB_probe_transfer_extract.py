#!/usr/bin/env python3
"""Experiment B, stage 1 — extract per-layer last-token states for the 20NG
topic set from base Llama-3.1-8B via NDIF remote (free tier; laptop only).

Same 4 topic classes and size as the local Instruct run (space, baseball,
graphics, mideast; 100 each). Left padding makes the last position the real
text end, so we save layers[i].output[:, -1] for all 33 residual points
(embeddings + 32 layers) and PERSIST them this time.

~50 remote traces (batch 8). Output: results/eB_states.npz
  states (400, 33, 4096) float16, labels (400,), texts sha1 manifest.
"""
import hashlib
import json
import os

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.datasets import fetch_20newsgroups

load_dotenv("/path/to/Documents/workspace/.env")
from nnsight import LanguageModel, CONFIG

CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

OUT = "/path/to/Documents/workspace/interp4discovery-neurips2026/results/eB_states.npz"
CATS = ["sci.space", "rec.sport.baseball", "comp.graphics", "talk.politics.mideast"]
PER_CLASS = 100
BATCH = 8
MAXCHARS = 600          # ~128 tokens

data = fetch_20newsgroups(subset="train", categories=CATS,
                          remove=("headers", "footers", "quotes"), shuffle=False)
texts, labels = [], []
for ci in range(len(CATS)):
    idx = [i for i, t in enumerate(data.target) if t == ci
           and len(data.data[i].strip()) > 200][:PER_CLASS]
    assert len(idx) == PER_CLASS, f"only {len(idx)} usable texts for {CATS[ci]}"
    texts += [data.data[i].strip()[:MAXCHARS] for i in idx]
    labels += [ci] * PER_CLASS

model = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
model.tokenizer.pad_token = model.tokenizer.eos_token
model.tokenizer.padding_side = "left"
NL = model.config.num_hidden_layers
print(f"layers={NL}, texts={len(texts)}", flush=True)


def val(p):
    return p.value if hasattr(p, "value") else p


def extract(batch):
    for attempt in range(6):
        if attempt:
            import time
            time.sleep(15 * attempt)
        try:
            saves = []
            with model.trace(batch, remote=True):
                saves.append(model.model.embed_tokens.output[:, -1].save())
                for L in range(NL):
                    saves.append(model.model.layers[L].output[:, -1].save())
            return torch.stack([val(x).float() for x in saves], dim=1)  # (B, 33, H)
        except Exception as e:
            print(f"  retry {attempt+1} ({type(e).__name__}: {str(e)[:80]})", flush=True)
    raise RuntimeError("extraction failed after 6 attempts")


chunks = []
done = 0
if os.path.exists(OUT):                      # resume
    prev = np.load(OUT)
    if prev["states"].shape[0] < len(texts):
        chunks = [torch.tensor(prev["states"].astype(np.float32))]
        done = prev["states"].shape[0]
        print(f"resuming from {done}", flush=True)
    else:
        print("already complete")
        raise SystemExit

for i in range(done, len(texts), BATCH):
    batch = texts[i:i + BATCH]
    chunks.append(extract(batch).cpu())
    if (i // BATCH) % 5 == 0 or i + BATCH >= len(texts):
        states = torch.cat(chunks).numpy().astype(np.float16)
        np.savez_compressed(OUT, states=states,
                            labels=np.array(labels[:states.shape[0]]),
                            cats=np.array(CATS),
                            sha=hashlib.sha1("".join(texts).encode()).hexdigest())
        print(f"{i + len(batch)}/{len(texts)} saved ({states.shape})", flush=True)

print("EXTRACTION COMPLETE", flush=True)
