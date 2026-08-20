"""
topk_kernel.py — Loads the top-k selection kernel and provides select_topk(),
used by the DIP MLPs to pick which channels survive pruning.

The kernel returns exactly k valid indices per call (deterministic tie-break),
so select_topk() returns its output directly. If the kernel is not built, it
falls back to torch.topk.
"""
import ctypes, os
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_so_dir = os.path.join(_here, "kernels", "topk_select")
_use_topk = False
if os.path.isdir(_so_dir):
    sos = [f for f in os.listdir(_so_dir)
           if f.startswith("topk_ops") and f.endswith(".so")]
    if sos:
        ctypes.CDLL(os.path.join(_so_dir, sos[0]))
        try:
            @torch.library.register_fake("topk_ops::topk_select")
            def _(s, k):
                return torch.empty(k, dtype=torch.int32, device=s.device)
            _use_topk = True
        except Exception:
            pass


def select_topk(s, k):
    """
    Select k channels by |s| using the CUDA kernel. Returns int64 idx [k].

    The kernel returns exactly k valid indices, so its output is returned
    directly with no data-dependent post-processing. Data-dependent ops
    (unique / boolean-filter) break PyTorch's fake-tensor tracing inside
    generate(), so they are deliberately avoided here.
    """
    if not _use_topk:
        return s.abs().topk(k).indices
    return torch.ops.topk_ops.topk_select(s.contiguous(), k).to(torch.long)