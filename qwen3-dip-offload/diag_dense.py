"""
diag_dense.py — Print actual tensor shapes inside one StreamingDIPMlp._dense
call, to find exactly where the wrong 76800 shape appears.
"""
import argparse
import torch
import torch.nn.functional as F
from qlinear import load_quantized_qwen3, dequantize
from dip_stream import build_streaming_dip, StreamingDIPMlp

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    args = p.parse_args()

    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()
    mods = build_streaming_dip(model, args.ckpt, 0.32)
    m = mods[0]

    print("== stored weight shapes ==")
    print("up_qw   :", tuple(m.up_qw.shape))     # expect [inter, in/2] = [25600, 2560]
    print("up_sc   :", tuple(m.up_sc.shape))     # [25600, 20]
    print("down_t  :", tuple(m.down_t.shape))    # SHOULD be [inter, out] = [25600, 5120]
    print("out_features:", m.out_features)
    print("inter (up rows):", m.inter)

    # simulate a prefill input: [batch=1, seq=15, hidden=5120]
    x = torch.randn(1, 15, 5120, dtype=torch.bfloat16, device="cuda:0")
    print("\n== _dense internal shapes ==")
    g = F.silu(m.gate_proj(x)); print("gate out g:", tuple(g.shape))
    up_w = dequantize(m.up_qw.to("cuda:0"), m.up_sc.to("cuda:0")).to(x.dtype)
    print("up_w dequant:", tuple(up_w.shape))
    u = F.linear(x, up_w); print("u = up(x):", tuple(u.shape))
    h = g * u; print("h = g*u:", tuple(h.shape))
    dt = m.down_t.to("cuda:0").to(x.dtype)
    print("down_t on gpu:", tuple(dt.shape))
    y = h @ dt
    print("y = h @ down_t:", tuple(y.shape), "<-- should be [1,15,5120]")

if __name__ == "__main__":
    main()