"""
offload/prefetch.py — Host-side streaming manager for CPU-offloaded layers.

The profiler showed ~60% of time is aten::copy_ + HtoD memcpy, plus ~12s of
cudaMalloc/cudaFree churn: accelerate allocates & frees a GPU buffer for every
offloaded layer, every token, and copies from *pageable* CPU memory.

This manager kills all three problems:
  1. PINNED memory   — CPU weights are page-locked once, so HtoD is ~2x faster
                       and can be truly async.
  2. REUSED buffers  — a small pool of persistent GPU buffers; no per-layer
                       cudaMalloc/cudaFree.
  3. DOUBLE-BUFFER   — layer N+1's weights are copied on a side stream while
                       layer N computes, overlapping PCIe with compute.

Design: we own the offloaded layers ourselves (no accelerate hooks). Each
offloaded decoder layer's parameters live pinned on CPU. Before the layer runs,
its weights are already on GPU (prefetched during the previous layer). After it
runs, its GPU buffer is handed back to the pool.

This is deliberately a first working version — correctness and buffer reuse
first, then we tune overlap. It is NOT a CUDA kernel: it's host-side stream
orchestration (cudaMemcpyAsync + streams), which is the right tool for moving
weights across PCIe.
"""

import torch


class LayerWeightStreamer:
    """
    Manages streaming a set of CPU-resident (pinned) layers to GPU on demand,
    with a reusable buffer pool and a prefetch stream for overlap.

    Usage sketch:
        streamer = LayerWeightStreamer(offloaded_layers, device="cuda:0", n_buffers=2)
        streamer.pin_all()                      # one-time: page-lock CPU weights
        # during forward, for each offloaded layer i in order:
        gpu_state = streamer.get(i)             # blocks only if not already prefetched
        streamer.prefetch(i + 1)                # kick off next layer's copy async
        # ... run layer i using gpu_state ...
        streamer.release(i)                     # return buffer to pool
    """

    def __init__(self, layer_param_dicts, device="cuda:0", n_buffers=2):
        """
        layer_param_dicts: list, one entry per offloaded layer. Each entry is a
            dict {param_name: cpu_tensor}. These are the tensors to stream.
        """
        self.device = torch.device(device)
        self.layers = layer_param_dicts
        self.n_layers = len(layer_param_dicts)
        self.n_buffers = n_buffers

        # a dedicated stream for prefetch copies, separate from the compute stream
        self.copy_stream = torch.cuda.Stream(device=self.device)

        # buffer pool: for each buffer slot, a dict {param_name: gpu_tensor}
        # allocated lazily on first use, sized to the largest layer, then reused.
        self._buffers = [None] * n_buffers
        self._slot_for_layer = {}       # layer_idx -> buffer slot in use
        self._ready_event = {}          # layer_idx -> cuda event marking copy done
        self._free_slots = list(range(n_buffers))

    # ---- one-time setup ----
    def pin_all(self):
        """Page-lock all CPU weights so async HtoD works and is faster."""
        for d in self.layers:
            for name, t in d.items():
                if not t.is_pinned():
                    d[name] = t.pin_memory()

    def _alloc_slot_like(self, layer_idx, slot):
        """Allocate (once) GPU tensors for a slot, shaped for this layer."""
        src = self.layers[layer_idx]
        buf = self._buffers[slot]
        if buf is None:
            buf = {}
        for name, t in src.items():
            need = (name not in buf) or (buf[name].shape != t.shape) or (buf[name].dtype != t.dtype)
            if need:
                buf[name] = torch.empty(t.shape, dtype=t.dtype, device=self.device)
        self._buffers[slot] = buf
        return buf

    # ---- streaming API ----
    def prefetch(self, layer_idx):
        """Kick off async copy of layer_idx's weights into a free buffer slot."""
        if layer_idx >= self.n_layers or layer_idx in self._slot_for_layer:
            return  # nothing to do / already in flight or resident
        if not self._free_slots:
            return  # no free buffer; caller will fetch synchronously via get()
        slot = self._free_slots.pop(0)
        buf = self._alloc_slot_like(layer_idx, slot)

        with torch.cuda.stream(self.copy_stream):
            for name, cpu_t in self.layers[layer_idx].items():
                buf[name].copy_(cpu_t, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.copy_stream)

        self._slot_for_layer[layer_idx] = slot
        self._ready_event[layer_idx] = ev

    def get(self, layer_idx):
        """
        Return {param_name: gpu_tensor} for layer_idx, blocking until its copy
        is done. If it wasn't prefetched, copy it now (synchronously-ish).
        """
        if layer_idx not in self._slot_for_layer:
            # not prefetched — do it now
            self.prefetch(layer_idx)
        if layer_idx not in self._slot_for_layer:
            # still no slot (pool exhausted) — fall back to a direct blocking copy
            slot = self._wait_and_reclaim_any()
            self._free_slots.append(slot)
            self.prefetch(layer_idx)

        # make the compute stream wait on this layer's copy event
        ev = self._ready_event[layer_idx]
        torch.cuda.current_stream(self.device).wait_event(ev)
        return self._buffers[self._slot_for_layer[layer_idx]]

    def release(self, layer_idx):
        """Return layer_idx's buffer slot to the pool for reuse."""
        slot = self._slot_for_layer.pop(layer_idx, None)
        if slot is not None:
            self._ready_event.pop(layer_idx, None)
            self._free_slots.append(slot)

    def _wait_and_reclaim_any(self):
        """Block on the oldest in-flight layer and reclaim its slot."""
        # simplest policy: reclaim the lowest layer_idx currently held
        victim = min(self._slot_for_layer.keys())
        ev = self._ready_event[victim]
        ev.synchronize()
        slot = self._slot_for_layer.pop(victim)
        self._ready_event.pop(victim, None)
        return slot