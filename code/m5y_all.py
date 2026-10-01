"""Driver: run m5y_joint_greedy over the seven M5W models, one subprocess each.

Reuses m5_universality's guards verbatim (project-local HF cache, evict after each
model, GPU-free check, HF_TOKEN from .env) so disk and failure behaviour match the
sweep whose writer counts this check audits. Models with an artifact are skipped.
"""

import subprocess
import sys

from experiments.m5_universality import (CACHE, HEADROOM_GB, MODELS, REPO,
                                         assert_gpu_is_free, child_env, evict, free_gb)

# m5w/m5x artifacts name GPT-2 small "gpt2"; keep that so files pair up
NAME = {"gpt2-small": "gpt2"}


def main() -> int:
    print("=== M5Y driver: joint greedy width, seven models ===", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    assert_gpu_is_free()
    env = child_env()
    status = {}
    for model_name, size_gb, _ in MODELS:
        tl_name = NAME.get(model_name, model_name)
        out = REPO / f"results/m5y_{tl_name.replace('/', '_')}.npz"
        print(f"\n##### {tl_name}", flush=True)
        if out.exists():
            status[tl_name] = "cached"
            continue
        if free_gb(REPO) < size_gb + HEADROOM_GB:
            status[tl_name] = "skipped: disk"
            print("  SKIPPED -- insufficient disk", flush=True)
            continue
        proc = subprocess.run(
            [sys.executable, "-u", str(REPO / "experiments/m5y_joint_greedy_batched.py"), tl_name],
            cwd=str(REPO), env=env, text=True, capture_output=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            status[tl_name] = "FAILED"
            print("  FAILED:\n" + "\n".join(proc.stderr.strip().splitlines()[-12:]),
                  flush=True)
        else:
            status[tl_name] = "ok"
        evict(model_name if model_name != "gpt2-small" else "gpt2")
    print("\nSTATUS", status, flush=True)
    return 0 if all(v in ("ok", "cached") for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
