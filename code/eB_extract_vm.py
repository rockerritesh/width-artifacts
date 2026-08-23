#!/usr/bin/env python3
"""Experiment B extraction, VM variant — per-layer states for the 20NG topic
set from Llama-3.1-8B-INSTRUCT in 4-bit, i.e. exactly the model and precision
the layer-band k*_eff was measured on.

Saves BOTH mean-pooled (the original separability convention) and last-token
states for all 33 residual points. Output: ~/e3-vision/eB_states_vm.npz
"""
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CATS = ["sci.space", "rec.sport.baseball", "comp.graphics", "talk.politics.mideast"]
PER_CLASS = 100
BATCH = 8
MAXLEN = 128
OUT = "eB_states_vm.npz"

data = fetch_20newsgroups(subset="train", categories=CATS,
                          remove=("headers", "footers", "quotes"), shuffle=False)
texts, labels = [], []
for ci in range(len(CATS)):
    idx = [i for i, t in enumerate(data.target) if t == ci
           and len(data.data[i].strip()) > 200][:PER_CLASS]
    assert len(idx) == PER_CLASS
    texts += [data.data[i].strip() for i in idx]
    labels += [ci] * PER_CLASS
print(f"{len(texts)} texts", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="cuda").eval()
torch.set_grad_enabled(False)

mean_states, last_states = [], []
for i in range(0, len(texts), BATCH):
    enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
              truncation=True, max_length=MAXLEN).to("cuda")
    hs = model(**enc, output_hidden_states=True).hidden_states  # 33 x (B,T,H)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    lengths = mask.sum(1)
    last_idx = enc["attention_mask"].sum(1) - 1
    mp, lt = [], []
    for h in hs:
        h = h.float()
        mp.append(((h * mask).sum(1) / lengths).cpu())
        lt.append(h[torch.arange(h.shape[0]), last_idx].cpu())
    mean_states.append(torch.stack(mp, dim=1))
    last_states.append(torch.stack(lt, dim=1))
    if (i // BATCH) % 10 == 0:
        print(f"{i + BATCH}/{len(texts)}", flush=True)

np.savez_compressed(
    OUT,
    mean_states=torch.cat(mean_states).numpy().astype(np.float16),
    last_states=torch.cat(last_states).numpy().astype(np.float16),
    labels=np.array(labels), cats=np.array(CATS))
print(f"wrote {OUT}", flush=True)
