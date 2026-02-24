#!/bin/bash
#SBATCH --job-name=trl-grpo-lora-sync
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MULTITURN_DIR="$WORKSPACE_ROOT/multiturn-llm-training"

# Mirror the environment bootstrap used by multiturn-llm-training/grpo_test.sh.
if [ -f "$MULTITURN_DIR/bash_variables.sh" ]; then
  chmod +x "$MULTITURN_DIR/bash_variables.sh"
  source "$MULTITURN_DIR/bash_variables.sh"
fi

# Keep the same accelerate config override from grpo_test.sh for consistency.
export ACCELERATE_CONFIG="/cluster/home/fraluca/.cache/huggingface/accelerate/default_config.yaml"

mkdir -p logs

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- User config ---
# Path to your local TRL checkout (this repo, with your PR changes)
TRL_REPO="$WORKSPACE_ROOT/trl"   # <-- adjust if needed
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_ID="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
RUN_NAME="${RUN_NAME:-grpo_lora_sync_smoke}"

# venv per job (recommended on clusters)
ENV_DIR="${SLURM_TMPDIR:-/tmp}/trl_venv"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install -U pip wheel setuptools

# Install TRL with vLLM extras, from your local repo (so it uses your modifications).
# TRL docs: pip install "trl[vllm]" for vLLM integration. :contentReference[oaicite:1]{index=1}
cd "$TRL_REPO"
pip install -e ".[vllm]"

# Pin vLLM if your cluster has surprises.
# TRL is version-sensitive; current docs list supported versions and examples. :contentReference[oaicite:2]{index=2}
# Uncomment if needed:
# pip install "vllm==0.10.2"

# Quick sanity prints
python -c "import trl, sys; print('TRL:', trl.__version__ if hasattr(trl,'__version__') else 'dev', 'at', trl.__file__); print(sys.executable)"

# --- Start vLLM server on GPU0 ---
# TRL docs show: CUDA_VISIBLE_DEVICES=0,1,... trl vllm-serve --model ... :contentReference[oaicite:3]{index=3}
# You added --enable_lora / --max_lora_rank in your PR implementation.
export CUDA_VISIBLE_DEVICES=0
export VLLM_SERVER_HOST="127.0.0.1"
export VLLM_SERVER_PORT="$VLLM_PORT"

echo "Starting TRL vLLM server on GPU0..."
trl vllm-serve \
  --model "$MODEL_ID" \
  --host "$VLLM_SERVER_HOST" \
  --port "$VLLM_SERVER_PORT" \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --enable_lora \
  --max_lora_rank 64 \
  > "logs/vllm_server_${SLURM_JOB_ID:-manual}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server readiness
echo "Waiting for vLLM server to come up..."
for i in {1..60}; do
  if curl -s --noproxy "*" "http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM server is healthy."
    break
  fi
  sleep 1
  if [ "$i" -eq 60 ]; then
    echo "vLLM server did not become healthy. Check logs/vllm_server_${SLURM_JOB_ID:-manual}.log"
    kill $VLLM_PID || true
    exit 1
  fi
done

# --- Run a tiny GRPO job on GPU1 ---
export CUDA_VISIBLE_DEVICES=1

cat > /tmp/train_grpo_smoke.py <<'PY'
import os
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# Tiny toy reward: number of unique characters in the completion.
# (TRL docs use similar "toy reward" patterns in examples.) :contentReference[oaicite:4]{index=4}
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

host = os.environ.get("VLLM_SERVER_HOST", "127.0.0.1")
port = int(os.environ.get("VLLM_SERVER_PORT", "8000"))

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# Small dataset slice for speed; just need a smoke test.
ds = load_dataset("trl-lib/tldr", split="train[:128]")  # small & fast
# Ensure the dataset has a "prompt" field expected by GRPOTrainer in your setup.
# Many TRL examples use TL;DR dataset; adjust if your local TRL expects different columns. :contentReference[oaicite:5]{index=5}

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

args = GRPOConfig(
    output_dir=f"outputs/grpo_lora_sync_smoke",
    run_name="grpo_lora_sync_smoke",
    bf16=True,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_prompt_length=256,
    max_completion_length=64,
    num_generations=2,

    use_vllm=True,
    vllm_mode="server",
    vllm_server_host=host,
    vllm_server_port=port,

    # This is your new flag:
    vllm_sync_strategy="lora_adapter",
    vllm_lora_name="policy",
    vllm_lora_rank=64,

    logging_steps=5,
    save_strategy="no",
    report_to=[],
)

trainer = GRPOTrainer(
    model=model_id,
    args=args,
    reward_funcs=reward_num_unique_chars,
    train_dataset=ds,
    peft_config=peft_config,
)

trainer.train()
print("DONE")
PY

export MODEL_ID="$MODEL_ID"
export RUN_NAME="$RUN_NAME"
echo "Starting GRPO smoke test on GPU1..."
python /tmp/train_grpo_smoke.py

# Cleanup server
echo "Stopping vLLM server..."
kill $VLLM_PID || true
wait $VLLM_PID || true

echo "Job complete."