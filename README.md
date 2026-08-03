# dynamic-input-pruning
Custom Kernels for Dynamic Input Pruning, and cache aware Dynamic Input pruning

1. Install deps (from here):

bash
uv pip install -r requirements.txt

2. Download the model (~62GB, the slow part — kick it off now):

bash
python download.py --repo_id Qwen/Qwen3-32B

3. Quantize to int4 (once download finishes):

bash
python quantize.py --model_dir checkpoints/Qwen/Qwen3-32B --out_dir checkpoints/qwen3-32b-int4

4. Then the actual milestone — profile:

bash
python profile_baseline.py --ckpt checkpoints/qwen3-32b-int4 --gpu_mem 10 --tokens 4