# PR Goal: vLLM Server Sync via LoRA Adapter Reload (avoid merge + full weight sync)

## Summary
Add an **opt-in** synchronization strategy for TRL’s vLLM **server mode** that, when using **PEFT LoRA/QLoRA**, updates the vLLM-side policy by **saving/reloading LoRA adapters** instead of **merging adapters + syncing full model weights**.

This is intended to make GRPO/online RL training with PEFT more reliable and efficient—especially for QLoRA-style training—while keeping the existing behavior as the default.

---


## Motivation

- Directly addresses https://github.com/vllm-project/vllm/issues/20149 

- Merging and Unmerging is incredibly slow because it loops through all named params and sequentially updates the vLLM server for each one
https://github.com/huggingface/trl/issues/3557

- Merging LoRA adapter into quantized model may lead to rounding errors
https://github.com/huggingface/trl/issues/3466


- Current weight transfer approach is brittle and sometimes fails after long hours of training (not guartanteed to be solved with this PR)
https://github.com/huggingface/trl/issues/2840

---

## Non-Goals
- Do **not** change defaults (must remain backward compatible).
- Do **not** re-architect TRL vLLM integration broadly.
- Do **not** guarantee compatibility with every vLLM version / serving backend.
- Do **not** implement a general “multi-model / multi-agent orchestrator” (keep scope to syncing the trainable policy).

---

## Proposed User-Facing Behavior
### New config / flag (opt-in)
Introduce a sync strategy toggle, e.g.:

- `vllm_sync_strategy = "weights" | "lora_adapter"`
  - default: `"weights"` (current behavior)
  - `"lora_adapter"`: adapter save + server adapter reload

Optional supporting knobs:
- `vllm_adapter_refresh_steps: int` (default: same cadence as current vLLM sync step or `save_steps`)
- `vllm_adapter_name: str` (default: derived from run name, e.g. `{run_name}-policy`)
- `vllm_adapter_dir: str` (default: output_dir subdir, e.g. `{output_dir}/vllm_adapters/`)
- `vllm_adapter_versioning: bool` (default: True; use step-stamped IDs to avoid races)

### Expected server requirements (documented)
For `vllm_sync_strategy="lora_adapter"` to work, the server must:
- support LoRA
- expose an API to load/unload or update adapters (depending on TRL’s vLLM server implementation)
- have access to adapter files (shared filesystem or equivalent mechanism)

If requirements are unmet:
- fail loudly with a clear error message, OR
- fallback to `"weights"` (decide and document)

## Design Overview
### Current flow (conceptual)
1. Trainer updates model parameters (PEFT or full)
2. Trainer “moves” model to vLLM / synchronizes policy to server
3. vLLM server generates rollouts for next step

### New flow for `"lora_adapter"`
1. Trainer updates LoRA parameters
2. On refresh:
   - save LoRA adapter checkpoint (atomic write)
   - instruct vLLM server to load the adapter (or load a new versioned adapter ID)
3. vLLM uses base model + latest adapter for rollouts

### Atomic write protocol (needs to be specified)

### Adapter versioning strategy
- **Versioned names** (recommended):
  - load `policy_step_1200`, then switch subsequent requests to that adapter id
  - optional cleanup of old versions

  ## Notes for My Setup (fill in)
- TRL version:
- vLLM version:
- Server mode: `trl vllm-serve` ?
- Inference base precision: FP16/BF16
- Training: QLoRA (bnb 4-bit) + LoRA adapters
- Shared adapter path available? yes/no
- Desired refresh cadence: Same as current update_named_params

## Checklist (Working)
