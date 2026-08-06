"""
diag_shapes.py — Check MLP proj shapes + dtype on offloaded 32B, and directly
compare our DIP kernel MLP output vs the dense MLP output for ONE layer, to
localize the numerical bug.
"""
import argparse
import torch
import torch.nn.functional as F
from qlinear import load_quantized_qwen3, QuantLinear, dequantize
from dip_mlp import _use_gather

def _w(proj):
    return dequantize(proj.qweight, proj.scales) if isinstance(proj, QuantLinear) else proj.weight

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/qwen3-32b-int4")
    p.add_argument("--gpu_mem", type=int, default=10)
    p.add_argument("--cpu_mem", type=int, default=60)
    args = p.parse_args()

    model, tok = load_quantized_qwen3(args.ckpt, args.gpu_mem, args.cpu_mem)
    model.eval()

    # grab the first MLP
    target = None
    for name, mod in model.named_modules():
        if all(hasattr(mod, p) for p in ("gate_proj","up_proj","down_proj")):
            target = mod; break

    up, gp, dn = target.up_proj, target.gate_proj, target.down_proj
    print("up_proj  qweight", tuple(up.qweight.shape), "scales", tuple(up.scales.shape))
    print("gate_proj qweight", tuple(gp.qweight.shape), "scales", tuple(gp.scales.shape))
    print("down_proj qweight", tuple(dn.qweight.shape), "scales", tuple(dn.scales.shape))

    dim_in = up.qweight.shape[1] * 2
    inter = up.qweight.shape[0]
    print(f"\ndim_in={dim_in}  inter={inter}")

    # make a random input on GPU, bf16
    x = torch.randn(dim_in, dtype=torch.bfloat16, device="cuda:0")

    # dense reference MLP
    g = F.silu(F.linear(x.unsqueeze(0), _w(gp).to(torch.bfloat16))).squeeze(0)
    u = F.linear(x.unsqueeze(0), _w(up).to(torch.bfloat16)).squeeze(0)
    h = g * u
    y_dense = F.linear(h.unsqueeze(0), _w(dn).to(torch.bfloat16)).squeeze(0)

    # DIP path (keep 0.32) using kernel for up_proj
    keep = 0.32
    k = max(1, int(round(inter * keep)))
    idx = g.abs().topk(k).indices.to(torch.int32)
    up_sel = torch.ops.int4_gather_ops.int4_gather_gemv(up.qweight, up.scales, x, idx)
    # compare up_sel vs dense u at idx
    u_ref_sel = u[idx.long()]
    err_up = (up_sel.float() - u_ref_sel.float()).abs()
    print(f"\nup_proj kernel vs dense (selected rows): max err {err_up.max():.4f}, "
          f"mean {err_up.mean():.5f}, rel {err_up.max()/ (u_ref_sel.abs().max()+1e-6):.4%}")

    h_sel = g[idx.long()] * up_sel.to(torch.bfloat16)
    y_dip = F.linear(h_sel.unsqueeze(0), _w(dn).to(torch.bfloat16)[:, idx.long()]).squeeze(0)

    err_y = (y_dip.float() - y_dense.float()).abs()
    print(f"final MLP DIP vs dense: max err {err_y.max():.3f}, "
          f"rel {err_y.max()/(y_dense.abs().max()+1e-6):.2%}")
    print("(some error expected from pruning; garbage would be huge/NaN)")
    print("y_dense sample:", y_dense[:5].float().tolist())
    print("y_dip   sample:", y_dip[:5].float().tolist())

if __name__ == "__main__":
    main()