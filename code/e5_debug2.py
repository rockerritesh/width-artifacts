import os
import traceback

from dotenv import load_dotenv

load_dotenv("/path/to/Documents/workspace/.env")
from nnsight import LanguageModel, CONFIG

CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
model = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
model.tokenizer.pad_token = model.tokenizer.eos_token
model.tokenizer.padding_side = "left"
P = ["The capital of France is", "Two plus two equals"]

# arm 3: double separate access to the same source layer (E5's pattern)
try:
    with model.trace(P, remote=True):
        model.model.layers[10].output = model.model.layers[9].output
        model.model.layers[11].output = model.model.layers[9].output
        lg = model.output.logits[:, -1, :].save()
    print("ARM3 (double access) OK")
except Exception:
    print("ARM3 FAIL")
    traceback.print_exc()

# arm 4: non-contiguous removal {5, 10} with one access per source
try:
    with model.trace(P, remote=True):
        model.model.layers[5].output = model.model.layers[4].output
        model.model.layers[10].output = model.model.layers[9].output
        lg2 = model.output.logits[:, -1, :].save()
    print("ARM4 (non-contiguous) OK")
except Exception:
    print("ARM4 FAIL")
    traceback.print_exc()

# arm 5: E5's exact block — closure list, set, while loop inside the trace
removed = [10, 11]
try:
    with model.trace(P, remote=True):
        rem = set(removed)
        for L in removed:
            j = L - 1
            while j >= 0 and j in rem:
                j -= 1
            src = (model.model.embed_tokens.output if j < 0
                   else model.model.layers[j].output)
            model.model.layers[L].output = src
        lg3 = model.output.logits[:, -1, :].save()
    print("ARM5 (E5 exact block) OK")
except Exception:
    print("ARM5 FAIL")
    traceback.print_exc()
