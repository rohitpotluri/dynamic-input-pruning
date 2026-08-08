"""
diag_hook2.py — Did our hook-strip actually remove the MLP hook?
Checks _hf_hook presence AFTER build_streaming_dip, and whether the MLP's
forward is wrapped by accelerate (the wrapper is what mangles shape).
"""
import argparse
import torch
from qlinear import load_quantized_qwen3
from dip_stream import build_streaming_dip

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    args = p.parse_args()

    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()
    build_streaming_dip(model, args.ckpt, 0.32)

    print("== AFTER strip attempt ==")
    n = 0
    for name, mlp in model.named_modules():
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            has_hook = hasattr(mlp, "_hf_hook")
            # accelerate stores the ORIGINAL forward as _old_forward when it wraps
            has_oldfwd = hasattr(mlp, "_old_forward")
            print(f"{name}: _hf_hook={has_hook}  _old_forward={has_oldfwd}  "
                  f"forward={getattr(mlp.forward,'__qualname__',mlp.forward)}")
            n += 1
            if n >= 3:
                break

    # also check the DECODER LAYER — maybe IT'S the one whose hook mangles output
    print("\n== decoder layer level ==")
    for name, mod in model.named_modules():
        if name in ("model.layers.0", "model.layers.1"):
            print(f"{name}: _hf_hook={hasattr(mod,'_hf_hook')}  "
                  f"_old_forward={hasattr(mod,'_old_forward')}")

if __name__ == "__main__":
    main()