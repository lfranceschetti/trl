# PR Goal: vLLM Server Sync via LoRA Adapter Reload (avoid merge + full weight sync)

## Summary

Add a new `vllm_sync_strategy="lora_adapter"` option for TRL's vLLM **server mode** that, when using **PEFT LoRA/QLoRA**, updates the vLLM-side policy by **saving and reloading LoRA adapters** instead of **merging adapters and syncing full model weights** (the current behavior).

This is intended to make GRPO/online RL training with PEFT more reliable and efficient - especially for QLoRA-style training - while keeping `vllm_sync_strategy="weights"` as the default.

---

### Specific issues addressed

1. **Sync is slow and memory intensive** — The merge-sync-unmerge loop in iterates over all named parameters and sequentially updates each one on the vLLM server. With `vllm_sync_strategy="lora_adapter"`, only the small LoRA checkpoint is written and loaded.
   [TRL #3557](https://github.com/huggingface/trl/issues/3557)
2. **QLoRA merge causes quantization rounding errors** — Merging LoRA weights into a 4-bit quantized base model and then unmerging is lossy. The `vllm_sync_strategy="lora_adapter"` approach avoids merging entirely; vLLM applies LoRA on top of the unmodified base model at inference time.
   [TRL #3466](https://github.com/huggingface/trl/issues/3466),
3. **NCCL weight transfer is brittle over long runs** (not tested if this solves it) — Current sometimes fails after hours of training due to NCCL communication errors. The `vllm_sync_strategy="lora_adapter"` approach replaces NCCL with a simple file write + HTTP reload request, which is more robust.
   [TRL #2840](https://github.com/huggingface/trl/issues/2840)

### Other advantages

**Decoupled from parameter naming** — the `"weights"` strategy iterates `state_dict()` keys and must match each one to the vLLM-side parameter name, which is fragile across PEFT versions and model architectures (see the manual prefix-stripping in [TRL #2818](https://github.com/huggingface/trl/pull/2818)). The `"lora_adapter"` strategy saves a standard PEFT checkpoint and lets vLLM load it through its own adapter path - the two sides never need to agree on internal parameter names.

### Prior PRs and why this PR is different

This approach was originally proposed in [TRL #2730](https://github.com/huggingface/trl/pull/2730) but was superseded by [TRL #2818](https://github.com/huggingface/trl/pull/2818), which chose the merge-sync-unmerge approach because it supports **all PEFT adapter types** (DoRA, IA3, etc.), not just LoRA.

This PR addresses that objection by:

- Keeping `vllm_sync_strategy="weights"` as the **default** — full backward compatibility, works with any adapter type
- Offering `vllm_sync_strategy="lora_adapter"` as an **opt-in optimization** for the most common case (standard LoRA/rsLoRA) raising a clear for non-LoRA adapter types (IA3, Prefix Tuning, etc.), guiding users to the `vllm_sync_strategy="weights"` default

The original [TRL #2730](https://github.com/huggingface/trl/pull/2730) noted a potential vLLM memory leak from repeatedly loading adapters. This might still be an issue, however only very small memory increases have been observed during experimental runs

### Future Improvements

vLLM's `load_inplace` feature ([vllm#31326](https://github.com/vllm-project/vllm/pull/31326), merged Jan 2026) could further improve this in the future but is not required. Trl adaptation suggested in [vLLM #20149](https://github.com/vllm-project/vllm/issues/20149)

## User-Facing Behavior

### New config / flag (opt-in)

Introduce a sync strategy toggle, e.g.:

- `vllm_sync_strategy = "weights" | "lora_adapter"`

  - default: `"weights"` (current behavior)
  - `"lora_adapter"`: adapter save + server adapter reload

## Notes for My Setup (fill in)

- TRL version:
- vLLM version:
- Server mode: `trl vllm-serve` ?
- Inference base precision: FP16/BF16
- Training: QLoRA (bnb 4-bit) + LoRA adapters
- Shared adapter path available? yes/no
- Desired refresh cadence: Same as current update_named_params
