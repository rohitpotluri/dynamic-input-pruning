"""
diag_devices.py — Where do MLP weights live during forward on offloaded 32B?

Checks, for the first few MLP layers, what device qweight/scales are on when
the DIP forward runs. Tells us whether offloaded layers hand us CPU/meta
tensors (which would explain the garbage output).
"""
import argparse
import torch
from qlinear import load_quantized_qwen3, QuantLinear

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    args = p.parse_args()

    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()

    seen = {"count": 0}
    def hook(mod, inp, out):
        if seen["count"] < 8 and isinstance(mod.up_proj, QuantLinear):
            qd = mod.up_proj.qweight.device
            sd = mod.up_proj.scales.device
            print(f"MLP up_proj.qweight device={qd}  scales device={sd}")
            seen["count"] += 1

    for name, mod in model.named_modules():
        if all(hasattr(mod, p) for p in ("gate_proj","up_proj","down_proj")):
            mod.register_forward_hook(hook)

    ids = tok.apply_chat_template(
        [{"role":"user","content":"hi"}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    ids = ids.to("cuda:0")
    with torch.no_grad():
        model.generate(ids, max_new_tokens=2, do_sample=False,
                       use_cache=True, pad_token_id=tok.eos_token_id)

if __name__ == "__main__":
    main()