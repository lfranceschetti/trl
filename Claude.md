# CLAUDE.md — LoRA Adapter Sync for TRL vLLM Server Mode

## Goal

Add an opt-in `vllm_sync_strategy="lora_adapter"` to TRL's `GRPOTrainer` (and other online trainers that use vLLM server mode). When enabled, the trainer syncs the policy to vLLM by **saving the LoRA adapter to disk and telling the vLLM server to reload it**, instead of the current approach of merging the adapter into the base weights and pushing every parameter tensor over NCCL.

This PR touches 4 files: `grpo_config.py`, `grpo_trainer.py`, `vllm_serve.py`, and `vllm_client.py`.

---

## Why This Matters

The current `_move_model_to_vllm` method (server mode, PEFT models) does:

```
1. unwrapped_model.merge_adapter()        # merge LoRA into base
2. state_dict = unwrapped_model.state_dict()  # full merged state dict
3. for name, param in state_dict.items():
4.     vllm_client.update_named_param(name, param)  # one HTTP+NCCL call per tensor
5. unwrapped_model.unmerge_adapter()
```

Problems this causes:
- **QLoRA crash/corruption**: `merge_adapter()` on a 4-bit model raises `TypeError` (PEFT #2501) or produces silent rounding errors with a UserWarning (TRL #3466).
- **Extremely slow**: Loops through hundreds of parameter tensors sequentially. For MoE models (128 experts), this takes minutes per sync (TRL #3557).
- **Brittle over long runs**: NCCL communicator state degrades, causing crashes after hours (TRL #2840).
- **Unnecessary data transfer**: Sends ~14GB (full 7B model) when only ~50-200MB of LoRA weights actually changed.

---

## Codebase Orientation

All paths relative to the TRL repo root. Target branch: `main` (latest stable is v0.25.1).

### Files to modify

| File | What it does | What changes |
|------|-------------|--------------|
| `trl/trainer/grpo_config.py` | Defines `GRPOConfig` dataclass | Add `vllm_sync_strategy`, `vllm_lora_rank` fields |
| `trl/trainer/grpo_trainer.py` | Contains `GRPOTrainer` | Add `_move_lora_to_vllm()` method, branch in `_move_model_to_vllm()` |
| `trl/scripts/vllm_serve.py` | FastAPI server with `WeightSyncWorker` | Add `/load_lora_adapter/` and `/unload_lora_adapter/` endpoints, start LLM with `enable_lora` when strategy is `lora_adapter` |
| `trl/extras/vllm_client.py` | HTTP client wrapper used by trainer | Add `load_lora_adapter()` method |

### Files to NOT modify

- `trl/trainer/grpo_trainer.py` loss computation, advantage calculation, sampling logic — untouched.
- The colocate mode code path — this PR only targets server mode (`vllm_mode="server"`).
- Any other trainer (DPO, SFT, etc.) — only `GRPOTrainer` uses `_move_model_to_vllm`.

### Key existing methods to understand

- `GRPOTrainer._move_model_to_vllm()`: Called every `steps_per_generation * num_iterations` training steps from `_prepare_inputs()`. This is the ONLY place where we need to branch.
- `GRPOTrainer._generate_and_score_completions()`: Calls `self.vllm_client.generate()`. If using LoRA adapter sync, the generate call needs to specify the LoRA adapter name. The vllm_client already passes model-related params; check if the server's `/generate/` endpoint needs a `model` or `lora_name` parameter.
- `VLLMClient.update_named_param()`: The current NCCL-based sync. We don't remove it — it remains the default.
- `VLLMClient.init_communicator()` / `close_communicator()`: NCCL setup/teardown. When using `lora_adapter` strategy, these should be skipped entirely (no NCCL communicator needed).

---

## Implementation Plan

### Step 1: `grpo_config.py` — Add config fields

```python
vllm_sync_strategy: str = field(
    default="weights",
    metadata={
        "help": 'Strategy for syncing model to vLLM server. "weights" (default): merge adapter '
        'and push all parameters via NCCL (current behavior). "lora_adapter": save adapter to '
        "disk and tell vLLM to reload it (requires vLLM --enable-lora). Only applies to "
        'vllm_mode="server" with PEFT models.'
    },
)
vllm_lora_rank: int = field(
    default=64,
    metadata={"help": "Max LoRA rank for vLLM server (must be >= actual adapter rank). Only used with vllm_sync_strategy='lora_adapter'."},
)
```

Add validation in `GRPOTrainer.__init__()`:
- If `vllm_sync_strategy == "lora_adapter"` and `vllm_mode != "server"`, raise `ValueError`.
- If `vllm_sync_strategy == "lora_adapter"` and model is not a PEFT model, raise `ValueError`.

### Step 2: `vllm_client.py` — Add LoRA adapter HTTP methods

Add to `VLLMClient`:

```python
def load_lora_adapter(self, lora_path: str) -> None:
    """Tell the vLLM server to load/reload a LoRA adapter from disk."""
    url = f"http://{self.host}:{self.server_port}/load_lora_adapter/"
    response = self.session.post(url, json={"lora_path": lora_path})
    response.raise_for_status()
```

When `vllm_sync_strategy == "lora_adapter"`, the trainer should skip `init_communicator()` and `close_communicator()` calls. Check `GRPOTrainer.__init__()` for where these are called and gate them:

```python
if self.args.vllm_sync_strategy != "lora_adapter":
    self.vllm_client.init_communicator()
```

### Step 3: `vllm_serve.py` — Add LoRA endpoints and enable-lora flag

The `trl vllm-serve` CLI currently does not accept `--enable_lora`. Add a new flag:

```python
# In ScriptArguments
enable_lora: bool = field(
    default=False,
    metadata={"help": "Enable LoRA adapter serving. Required for vllm_sync_strategy='lora_adapter'."},
)
max_lora_rank: int = field(
    default=64,
    metadata={"help": "Maximum LoRA rank. Must be >= the rank of any adapter that will be loaded."},
)
```

In `main()`, pass to the LLM constructor:

```python
llm = LLM(
    ...
    enable_lora=script_args.enable_lora,
    max_lora_rank=script_args.max_lora_rank,
    max_loras=2,  # reasonable default
    ...
)
```

Add endpoints:

```python
class LoadLoRARequest(BaseModel):
    lora_path: str

@app.post("/load_lora_adapter/")
async def load_lora_adapter(request: LoadLoRARequest):
    from vllm.lora.request import LoRARequest as VLLMLoRARequest
    nonlocal lora_request  # or use app.state

    new_lora = VLLMLoRARequest(
        lora_name="policy",
        lora_int_id=abs(hash("policy")) % (2**31),
        lora_path=request.lora_path,
    )
    # Use vLLM's offline LLM API: pass lora_request per generate call
    lora_request = new_lora
    return {"status": "success"}
```

The `/generate/` endpoint must forward the `lora_request` to `llm.generate()`:

```python
all_outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
```

The existing `/generate/` endpoint takes a `prompts` list but does NOT currently accept a `lora_request`. The simplest approach: use the server-level `lora_request` variable (set by `/load_lora_adapter/`), not a per-request parameter. This matches the training use case where there's exactly one policy adapter.

### Step 4: `grpo_trainer.py` — Add the new sync path

Add a new method `_move_lora_to_vllm()` and branch in the existing `_move_model_to_vllm()`:

```python
@profiling_decorator
def _move_model_to_vllm(self):
    if self.args.vllm_sync_strategy == "lora_adapter":
        return self._move_lora_to_vllm()
    # ... existing merge+push code unchanged ...

def _move_lora_to_vllm(self):
    """Save LoRA adapter and tell vLLM server to reload it."""
    if self.accelerator.is_main_process:
        adapter_dir = os.path.join(self.args.output_dir, "vllm_lora_adapter")
        tmp_dir = adapter_dir + "_tmp"

        # Save adapter (only LoRA params, ~50-200MB)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(tmp_dir)

        # Atomic swap
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.rename(tmp_dir, adapter_dir)

        # Tell vLLM to reload
        self.vllm_client.load_lora_adapter(lora_path=adapter_dir)

    # Sync all processes
    if self.accelerator.num_processes > 1:
        self.accelerator.wait_for_everyone()
```

**Critical detail**: `save_pretrained` on a PEFT model saves only the adapter files (`adapter_config.json` + `adapter_model.safetensors`). This is the correct format for vLLM's LoRA loading.

**Critical detail**: With DeepSpeed ZeRO-3, adapter parameters may be sharded. Wrap the save in the same `unwrap_model_for_generation` / `gather_deepspeed3_params` context manager that the existing code already uses:

```python
with unwrap_model_for_generation(
    self.model_wrapped, self.accelerator,
    gather_deepspeed3_params=self.args.ds3_gather_for_generation
) as unwrapped_model:
    unwrapped_model.save_pretrained(tmp_dir)
```

### Step 5: Skip NCCL communicator setup

In `GRPOTrainer.__init__()`, find where `self.vllm_client.init_communicator()` is called and gate it:

```python
if self.args.vllm_sync_strategy != "lora_adapter":
    self.vllm_client.init_communicator()
```

Similarly, gate `close_communicator()` in the cleanup/teardown.

When `enable_lora=True` on the server side, the `WeightSyncWorker` custom worker class is unnecessary. But for backward compatibility, keep it — it just won't be used if no NCCL communicator is initialized. Alternatively, when `--enable_lora` is passed, don't set `worker_cls`.

---

## What NOT to Do

- **Don't touch the existing `"weights"` code path.** This PR adds a parallel path, not a replacement.
- **Don't change the generation endpoint protocol.** The server's `/generate/` already returns completions. The adapter is loaded server-side; the client doesn't need to specify it per request.
- **Don't implement adapter versioning / cleanup in v1.** Save to a fixed path, overwrite each time. Versioning is a follow-up.
- **Don't add vLLM `enable_lora` auto-detection.** The user must explicitly pass `--enable_lora` to `trl vllm-serve` and `vllm_sync_strategy="lora_adapter"` to the trainer. Explicit is better.
- **Don't try to support colocate mode.** The colocate mode has a completely different sync mechanism (in-process model sharing). This PR is server-mode only.

---

## Testing Checklist

1. **Default behavior unchanged**: Run GRPO with `vllm_sync_strategy="weights"` (default). Everything works as before.
2. **LoRA sync basic**: Run GRPO with a PEFT model, `vllm_sync_strategy="lora_adapter"`, vLLM server started with `--enable_lora --max_lora_rank 64`. Verify:
   - Adapter files appear in `{output_dir}/vllm_lora_adapter/`
   - Server logs show adapter loaded
   - Generated completions change as training progresses (not stuck on base model)
3. **QLoRA**: Same as above but with 4-bit quantized training model. Verify no `merge_adapter()` warning, no TypeError.
4. **Multi-GPU training**: Run with `accelerate` (2+ training GPUs) + separate vLLM server. Verify only main process saves/reloads.
5. **Error handling**: Start vLLM without `--enable_lora`, use `vllm_sync_strategy="lora_adapter"`. Verify clear error message.

---

## Related Issues

This PR addresses:
- **TRL #3466**: QLoRA + vLLM merge produces rounding errors → bypassed entirely (no merge)
- **TRL #3557**: `_move_model_to_vllm` loops through all named params → replaced with single adapter save + HTTP call
- **TRL #2840**: Weight transfer fails after hours → no NCCL communicator state to degrade
- **PEFT #2501**: `merge_adapter()` on 4-bit crashes with TypeError → no merge needed
- **vLLM #20149**: Request for LoRA tensor update API → uses vLLM's existing LoRA loading

---

## vLLM LoRA API Reference

vLLM supports LoRA at two levels. Both work for this use case:

**Offline LLM API** (used by `trl vllm-serve`):
```python
from vllm.lora.request import LoRARequest
lora_req = LoRARequest("policy", 1, "/path/to/adapter")
outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
```
vLLM caches the adapter internally. To force a reload from disk, create a new `LoRARequest` with a new `lora_int_id`.

**OpenAI-compatible server API** (alternative if using `vllm serve` directly):
```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve model --enable-lora
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -d '{"lora_name":"policy","lora_path":"/path","load_inplace":true}'
```
`load_inplace=True` replaces an existing adapter with the same name (designed for RL training loops per vLLM docs).

This PR uses the **offline LLM API** approach since `trl vllm-serve` already uses `vllm.LLM` directly.

### Forcing adapter reload

vLLM caches adapters by `lora_int_id`. To reload updated weights from the same path, increment the ID:

```python
self._lora_version += 1
lora_request = LoRARequest(
    lora_name=name,
    lora_int_id=self._lora_version,
    lora_path=path,
)
```

This is the simplest reliable way to ensure vLLM re-reads from disk.