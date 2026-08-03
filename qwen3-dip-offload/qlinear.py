"""
qlinear.py — Custom INT4 (g128, symmetric) linear layer + Qwen3 loader.

Strategy: let transformers own the Qwen3 architecture (QK-norm, GQA, RoPE, etc).
We only replace the 7 projection nn.Linear modules per layer with QuantLinear,
which reads our packed format:

    qweight : int8  [out_features, in_features // 2]   (two int4 per byte)
    scales  : bf16  [out_features, in_features // 128]

Forward (for now) does a NAIVE dequant in PyTorch — unpack nibbles, scale,
F.linear. This is the baseline path. Later, the fused gather+dequant+GEMV
CUDA kernel slots into QuantLinear.forward for the decode (seqlen==1) case,
exactly like the SmolLM3 WeightOnlyInt8Linear pattern.

The whole point: build this as a normal nn.Module tree so accelerate's
device_map="auto" can place/offload layers across GPU + CPU.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_SIZE = 128


def _unpack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """
    qweight: int8 [out_f, in_f//2], two signed int4 packed per byte
             (low nibble = even col, high nibble = odd col).
    Returns int8 [out_f, in_f] with values in [-8, 7].
    """
    q = qweight.to(torch.uint8)
    low = q & 0x0F            # even columns
    high = (q >> 4) & 0x0F    # odd columns

    # sign-extend 4-bit -> 8-bit: values >= 8 are negative
    def sext(x):
        x = x.to(torch.int8)
        return torch.where(x >= 8, x - 16, x)

    low = sext(low)
    high = sext(high)

    out_f, half = q.shape
    out = torch.empty(out_f, half * 2, dtype=torch.int8, device=q.device)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out


def dequantize(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct bf16 weight [out_f, in_f] from packed int4 + group scales.
    scales: bf16 [out_f, in_f//GROUP_SIZE]
    """
    q = _unpack_int4(qweight)                       # int8 [out_f, in_f]
    out_f, in_f = q.shape
    n_groups = in_f // GROUP_SIZE
    qf = q.float().reshape(out_f, n_groups, GROUP_SIZE)
    s = scales.float().reshape(out_f, n_groups, 1)
    w = (qf * s).reshape(out_f, in_f)
    return w.to(torch.bfloat16)


class QuantLinear(nn.Module):
    """INT4 g128 symmetric linear. Naive dequant forward for now."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert in_features % GROUP_SIZE == 0
        # buffers, so accelerate moves/offloads them with the module
        self.register_buffer(
            "qweight",
            torch.empty(out_features, in_features // 2, dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.empty(out_features, in_features // GROUP_SIZE, dtype=torch.bfloat16),
        )
        self.bias = None  # Qwen3 projections are bias-free

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NAIVE PATH: dequantize full weight, then matmul.
        # (kernel will replace this for the seqlen==1 decode case later)
        w = dequantize(self.qweight, self.scales)        # [out_f, in_f] bf16
        return F.linear(x, w.to(x.dtype))


def _replace_linears_with_quant(model):
    """
    Walk the Qwen3 model and swap the 7 projection Linears per layer for
    QuantLinear (empty buffers; weights loaded separately).
    """
    quant_substrings = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")
    replaced = []
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(s in full for s in quant_substrings):
                ql = QuantLinear(child.in_features, child.out_features)
                setattr(module, child_name, ql)
                replaced.append(full)
    return replaced


def load_quantized_qwen3(ckpt_dir: str, gpu_mem_gib: int = 13, cpu_mem_gib: int = 60):
    """
    Build the Qwen3 model with QuantLinear layers and our int4 weights,
    placed/offloaded across GPU+CPU via accelerate.
    Returns (model, tokenizer).
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file

    ckpt_dir = Path(ckpt_dir)

    # sanity: confirm our quant format
    qcfg = json.load(open(ckpt_dir / "quant_config.json"))
    assert qcfg["group_size"] == GROUP_SIZE, qcfg

    config = AutoConfig.from_pretrained(ckpt_dir)

    # 1) build the architecture with empty (meta) weights — nothing allocated yet
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    # 2) swap projection Linears -> QuantLinear (still on meta)
    replaced = _replace_linears_with_quant(model)
    print(f"Swapped {len(replaced)} Linear layers -> QuantLinear")

    # 3) decide the GPU/CPU split on the EMPTY model. Since weights aren't
    #    materialized yet, this can't overfill the GPU.
    max_memory = {0: f"{gpu_mem_gib}GiB", "cpu": f"{cpu_mem_gib}GiB"}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Qwen3DecoderLayer"],
        dtype=torch.bfloat16,
    )
    from collections import Counter
    print(f"Device map distribution: {dict(Counter(str(v) for v in device_map.values()))}")

    # 4) load our int4 weights and place each tensor DIRECTLY onto its assigned
    #    device (per the map above). The GPU only ever receives the tensors
    #    that belong on it — no over-fill, real offload to CPU for the rest.
    state = load_file(str(ckpt_dir / "model_int4.safetensors"), device="cpu")

    def device_for(param_name: str):
        # find the longest module prefix in device_map that matches this param
        best = None
        for mod_name, dev in device_map.items():
            if param_name == mod_name or param_name.startswith(mod_name + "."):
                if best is None or len(mod_name) > len(best[0]):
                    best = (mod_name, dev)
        return best[1] if best else "cpu"

    for pname, tensor in state.items():
        dev = device_for(pname)
        # offloaded ("disk"/"cpu") tensors stay on CPU; GPU ones move to cuda:0
        target = "cuda:0" if dev == 0 or dev == "cuda:0" else "cpu"
        set_module_tensor_to_device(model, pname, target, value=tensor)

    # 5) attach accelerate's offload hooks so CPU-resident layers stream to GPU
    #    on demand during forward (this is the naive offload we will optimize).
    model = dispatch_model(model, device_map=device_map, offload_buffers=True)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    return model, tokenizer