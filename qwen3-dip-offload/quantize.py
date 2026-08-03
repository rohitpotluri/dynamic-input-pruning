"""
quantize.py — Self-quantize bf16 Qwen3-32B to symmetric INT4, group-128.

We control the layout so our fused gather+dequant+GEMV kernel can read it
directly. Design choices (kept deliberately simple for a lean kernel):

  * Symmetric quant: value = round(w / scale), no zero-point.
    scale = max(|w_group|) / 7   (int4 signed range is [-8, 7]; we use 7)
  * Group size 128 along the in_features (columns) axis.
  * Two int4 values packed per int8 byte (low nibble = even col, high = odd).
  * Per Linear we store:
        <name>.qweight : int8,  [out_features, in_features // 2]   (packed)
        <name>.scales  : bf16,  [out_features, in_features // 128]
  * lm_head ("output") and embeddings are left in bf16 (accuracy-sensitive,
    and lm_head is huge but only run once per token at the very end).

This mirrors the SmolLM3 quant flow but extends int8 -> packed int4 groupwise.

Usage:
    python quantize.py --model_dir checkpoints/Qwen/Qwen3-32B \
                       --out_dir   checkpoints/qwen3-32b-int4
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

GROUP_SIZE = 128
QMAX = 7  # symmetric signed int4 uses [-8, 7]; divide by 7


def quantize_tensor_int4_groupwise(w: torch.Tensor):
    """
    w: [out_features, in_features] float/bf16 weight.
    Returns:
        qweight_packed: int8 [out_features, in_features // 2]
        scales:         bf16 [out_features, in_features // GROUP_SIZE]
    """
    out_f, in_f = w.shape
    assert in_f % GROUP_SIZE == 0, f"in_features {in_f} not divisible by {GROUP_SIZE}"
    assert in_f % 2 == 0, "in_features must be even to pack two int4 per byte"

    w = w.float()
    n_groups = in_f // GROUP_SIZE

    # reshape into groups: [out_f, n_groups, GROUP_SIZE]
    wg = w.reshape(out_f, n_groups, GROUP_SIZE)

    # symmetric scale per group: max abs / 7
    max_abs = wg.abs().amax(dim=2, keepdim=True)           # [out_f, n_groups, 1]
    scales = (max_abs / QMAX).clamp(min=1e-8)              # avoid div by zero

    # quantize -> integers in [-8, 7]
    q = torch.round(wg / scales).clamp(-8, 7).to(torch.int8)  # [out_f, n_groups, GS]
    q = q.reshape(out_f, in_f)                                # [out_f, in_f]

    # pack two int4 per byte. low nibble = even column, high nibble = odd column.
    q_low = q[:, 0::2]   # even cols
    q_high = q[:, 1::2]  # odd cols
    # mask to 4 bits (two's complement of negatives -> low nibble)
    q_low_u = (q_low & 0x0F).to(torch.uint8)
    q_high_u = (q_high & 0x0F).to(torch.uint8)
    packed = (q_low_u | (q_high_u << 4)).to(torch.int8)   # [out_f, in_f//2]

    scales = scales.squeeze(-1).to(torch.bfloat16)        # [out_f, n_groups]
    return packed, scales


def should_quantize(name: str) -> bool:
    """Quantize Linear weights in attention + mlp. Skip embeddings, norms, lm_head."""
    if not name.endswith(".weight"):
        return False
    lname = name.lower()
    if "embed_tokens" in lname:
        return False
    if "lm_head" in lname:
        return False
    if "norm" in lname:          # RMSNorm weights (1D) — never quantize
        return False
    # Qwen3 projections we DO quantize:
    quantizable = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")
    return any(p in lname for p in quantizable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Dir with bf16 safetensors (downloaded Qwen3-32B).")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output dir for the int4 checkpoint.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # locate safetensors shards
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    print(f"Found {len(shard_files)} shard(s). Quantizing INT4 g{GROUP_SIZE} symmetric.")

    new_state = {}
    n_quant, n_kept = 0, 0

    for shard in shard_files:
        path = model_dir / shard
        print(f"  loading {shard} ...")
        sd = load_file(str(path), device="cpu")
        for name, tensor in sd.items():
            if should_quantize(name):
                base = name[:-len(".weight")]   # strip ".weight"
                packed, scales = quantize_tensor_int4_groupwise(tensor)
                new_state[base + ".qweight"] = packed
                new_state[base + ".scales"] = scales
                n_quant += 1
            else:
                # keep as-is (embeddings, norms, lm_head, biases)
                new_state[name] = tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
                n_kept += 1
        del sd

    print(f"Quantized {n_quant} weight tensors, kept {n_kept} as-is.")

    out_path = out_dir / "model_int4.safetensors"
    print(f"Saving to {out_path} ...")
    save_file(new_state, str(out_path))

    # copy config + tokenizer so the checkpoint is self-contained / loadable
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "special_tokens_map.json",
                  "vocab.json", "merges.txt"):
        src = model_dir / fname
        if src.exists():
            import shutil
            shutil.copy(src, out_dir / fname)

    # record our quant metadata for the kernel / loader to read
    meta = {
        "quant_method": "self_int4_symmetric_groupwise",
        "group_size": GROUP_SIZE,
        "bits": 4,
        "symmetric": True,
        "packing": "two_int4_per_int8_low_even_high_odd",
        "qmax": QMAX,
    }
    with open(out_dir / "quant_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()