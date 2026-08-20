"""
gen_utils.py — Shared generation + timing helpers for the run and benchmark
scripts.

Centralizes:
  * prompt and chat-template construction (thinking disabled for Qwen3)
  * warmup-excluded steady-state decode timing (mean tok/s over several runs)

so the prompt, thinking mode, and timing methodology are identical everywhere.
"""
import time
import torch

# Generic prompt: short factual answer, no reasoning required.
DEFAULT_PROMPT = "What is the lowest point on Earth's surface?"


def build_inputs(tok, prompt, thinking=False):
    """Build input_ids with the Qwen3 chat template; thinking off by default."""
    try:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=thinking)       # Qwen3: controls <think> blocks
    except TypeError:
        # tokenizer/template without enable_thinking support -> plain template
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    return ids.to("cuda:0")


def generate_text(model, tok, prompt=DEFAULT_PROMPT, tokens=40, thinking=False):
    """Single generation; returns (text, n_tokens, seconds)."""
    ids = build_inputs(tok, prompt, thinking)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return txt, out.shape[-1] - ids.shape[-1], dt


def benchmark_decode(model, tok, prompt=DEFAULT_PROMPT, warmup_tokens=8,
                     timed_tokens=64, runs=3, thinking=False, verbose=True):
    """
    Warmup-excluded steady-state decode benchmark:

    1. one warmup generation (discarded) to pay one-time costs (CUDA init,
       first-token allocation, etc.)
    2. `runs` timed generations of `timed_tokens` each
    3. report mean and std tok/s across runs

    Returns a dict with mean/std tok/s, per-run values, and a sample output.
    """
    ids = build_inputs(tok, prompt, thinking)
    prompt_len = ids.shape[-1]

    if verbose:
        print(f"  warmup ({warmup_tokens} tok) ...")
    with torch.no_grad():
        model.generate(ids, max_new_tokens=warmup_tokens, do_sample=False,
                       use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    tps = []
    sample = None
    for r in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=timed_tokens, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = out.shape[-1] - prompt_len
        tps.append(n / dt)
        if sample is None:
            sample = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        if verbose:
            print(f"  run {r+1}/{runs}: {n} tok in {dt:.1f}s -> {n/dt:.3f} tok/s")

    t = torch.tensor(tps)
    return {
        "mean_tps": t.mean().item(),
        "std_tps": t.std().item() if len(tps) > 1 else 0.0,
        "runs": tps,
        "sample": sample,
    }