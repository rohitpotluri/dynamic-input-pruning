"""
diag_layer_hook.py — Is the DECODER-LAYER hook the shape-mangler?

Tests, on the UNMODIFIED offloaded model (no DIP at all), whether a decoder
layer's AlignDevicesHook reshapes/mangles multi-token output. If the plain
model works fine (it does -> baseline is coherent), then the layer hook is NOT
inherently a mangler, and the mangling comes specifically from our MLP change.

We directly call one decoder layer's MLP with a 15-token input, both:
  (a) through the normal hooked path (what happens in real forward)
  (b) checking what shape comes out
to see if the layer/MLP hook itself distorts shape.
"""
import argparse
import torch
from qlinear import load_quantized_qwen3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    args = p.parse_args()

    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()

    # grab first decoder layer's MLP (UNMODIFIED - still has accelerate hooks)
    layer0 = model.model.layers[0]
    mlp = layer0.mlp

    print("MLP _hf_hook:", hasattr(mlp, "_hf_hook"))
    print("MLP _old_forward:", hasattr(mlp, "_old_forward"))

    # feed a 15-token hidden state directly to the UNMODIFIED mlp
    x = torch.randn(1, 15, 5120, dtype=torch.bfloat16, device="cuda:0")
    with torch.no_grad():
        out = mlp(x)
    print(f"\nUNMODIFIED mlp: in {tuple(x.shape)} -> out {tuple(out.shape)}")
    print("If out is (1,15,5120): the hook does NOT mangle shape ->")
    print("  the bug is in OUR StreamingDIPMlp, not accelerate's hook.")
    print("If out is wrong (e.g. (1,1,76800)): the hook itself mangles ->")
    print("  pre-dispatch won't help and we need a different structure.")

if __name__ == "__main__":
    main()