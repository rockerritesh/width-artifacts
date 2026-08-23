import os
import traceback

from dotenv import load_dotenv

load_dotenv("/path/to/Documents/workspace/.env")
import nnsight
from nnsight import LanguageModel, CONFIG

print("nnsight", nnsight.__version__)
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
model = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
model.tokenizer.pad_token = model.tokenizer.eos_token
model.tokenizer.padding_side = "left"

# arm 1: the exact pattern the old (working, July 2026) script used
try:
    with model.trace(["The capital of France is", "Two plus two equals"], remote=True):
        ent = model.model.layers[9].output
        model.model.layers[10].output = ent
        model.model.layers[11].output = ent
        lg = model.output.logits[:, -1, :].save()
    v = lg.value if hasattr(lg, "value") else lg
    print("ARM1 OK", tuple(v.shape))
except Exception:
    print("ARM1 FAIL")
    traceback.print_exc()

# arm 2: no intervention (sanity)
try:
    with model.trace(["The capital of France is", "Two plus two equals"], remote=True):
        lg2 = model.output.logits[:, -1, :].save()
    v2 = lg2.value if hasattr(lg2, "value") else lg2
    print("ARM2 OK", tuple(v2.shape))
except Exception:
    print("ARM2 FAIL")
    traceback.print_exc()
