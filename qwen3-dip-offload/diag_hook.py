"""
diag_hook.py — Check whether the MLP modules (whose forward we replaced) carry
accelerate's _hf_hook, and whether that hook is what mangles the output shape.
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

    # find the MLP modules and check hooks BEFORE we replace them
    print("== BEFORE build_streaming_dip ==")
    n = 0
    for name, mlp in model.named_modules():
        if all(hasattr(mlp, p2) for p2 in ("gate_proj", "up_proj", "down_proj")):
            has_hook = hasattr(mlp, "_hf_hook")
            hook_type = type(mlp._hf_hook).__name__ if has_hook else "none"
            # is the PARENT decoder layer hooked instead?
            print(f"{name}: _hf_hook={has_hook} ({hook_type})")
            n += 1
            if n >= 4:
                break

    # also check the decoder layer level
    print("\n== decoder layer hooks ==")
    n = 0
    for name, mod in model.named_modules():
        if name.endswith("layers.0") or name.endswith("layers.1"):
            has_hook = hasattr(mod, "_hf_hook")
            print(f"{name}: _hf_hook={has_hook} "
                  f"({type(mod._hf_hook).__name__ if has_hook else 'none'})")

    print("\n== AFTER build_streaming_dip ==")
    mods = build_streaming_dip(model, args.ckpt, 0.32)
    n = 0
    for name, mlp in model.named_modules():
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            has_hook = hasattr(mlp, "_hf_hook")
            hook_type = type(mlp._hf_hook).__name__ if has_hook else "none"
            print(f"{name}: _hf_hook={has_hook} ({hook_type}), "
                  f"forward={mlp.forward.__qualname__ if hasattr(mlp.forward,'__qualname__') else mlp.forward}")
            n += 1
            if n >= 4:
                break

if __name__ == "__main__":
    main()