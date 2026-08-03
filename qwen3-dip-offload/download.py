"""
download.py — Fetch the bf16 Qwen3-32B checkpoint from HuggingFace.

This is the full-precision model (~65GB). We download it once, then quantize
it ourselves to int4 group-128 (quantize.py) in a layout our kernels control.

Usage:
    python download.py --repo_id Qwen/Qwen3-32B
    # private/gated repo:
    python download.py --repo_id Qwen/Qwen3-32B --hf_token hf_xxx
"""

import os
from typing import Optional

from requests.exceptions import HTTPError


def hf_download(repo_id: str, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download

    local_dir = f"checkpoints/{repo_id}"
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns=["*.pth", "*.pt", "consolidated*"],
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print("401: pass a valid --hf_token for gated/private repos.")
        else:
            raise e
    print(f"Done. Model at {local_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a model from HF Hub.")
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-32B",
                        help="HF repo id of the bf16 model.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HF API token (for gated/private repos).")
    args = parser.parse_args()
    hf_download(args.repo_id, args.hf_token)