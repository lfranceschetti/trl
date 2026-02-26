#!/bin/bash
#SBATCH --time=3:59:00          # Max runtime
#SBATCH --mem-per-cpu=50G         # Memory per CPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1          
#SBATCH --gpus=rtx_4090:2
#SBATCH --output=run_new_repetition.out
#SBATCH --error=run_new_repetition.err


set -euo pipefail

# IMPORTANT: under Slurm, BASH_SOURCE[0] points to a copied script in /var/spool.
# Resolve paths from explicit overrides or stable absolute defaults.
DEFAULT_WORKSPACE_ROOT="/cluster/home/fraluca/negotio2"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [ -n "${TRL_REPO:-}" ]; then
  TRL_REPO="$(cd "$TRL_REPO" && pwd)"
  WORKSPACE_ROOT="$(cd "$TRL_REPO/.." && pwd)"
elif [ -d "$SUBMIT_DIR/trl" ] && [ -d "$SUBMIT_DIR/multiturn-llm-training" ]; then
  WORKSPACE_ROOT="$(cd "$SUBMIT_DIR" && pwd)"
  TRL_REPO="$WORKSPACE_ROOT/trl"
elif [ -d "$SUBMIT_DIR" ] && [ "$(basename "$SUBMIT_DIR")" = "trl" ] && [ -d "$SUBMIT_DIR/../multiturn-llm-training" ]; then
  TRL_REPO="$(cd "$SUBMIT_DIR" && pwd)"
  WORKSPACE_ROOT="$(cd "$SUBMIT_DIR/.." && pwd)"
else
  WORKSPACE_ROOT="$DEFAULT_WORKSPACE_ROOT"
  TRL_REPO="$WORKSPACE_ROOT/trl"
fi

MULTITURN_DIR="$WORKSPACE_ROOT/multiturn-llm-training"

# Mirror the environment bootstrap used by multiturn-llm-training/grpo_test.sh.
if [ -f "$MULTITURN_DIR/bash_variables.sh" ]; then
  # bash_variables.sh appends to PYTHONPATH and can fail under `set -u` when unset.
  export PYTHONPATH="${PYTHONPATH:-}"
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
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_ID="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
RUN_NAME="${RUN_NAME:-vllm_sync_lora}"
WANDB_PROJECT="${WANDB_PROJECT:-grpo_lora_length_reward}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# venv per job (recommended on clusters)
# Use a unique environment per run to avoid stale/corrupted metadata from prior jobs.
ENV_DIR="${SLURM_TMPDIR:-/tmp}/trl_venv_${SLURM_JOB_ID:-$$}"
rm -rf "$ENV_DIR"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
# Keep pip on a stable release line for this cluster stack.
python -m pip install -U "pip<26" wheel setuptools

# Install TRL with vLLM extras, from your local repo (so it uses your modifications).
# TRL docs: pip install "trl[vllm]" for vLLM integration. :contentReference[oaicite:1]{index=1}
[ -d "$TRL_REPO" ] || { echo "Missing TRL_REPO: $TRL_REPO"; exit 1; }
[ -f "$TRL_REPO/pyproject.toml" ] || [ -f "$TRL_REPO/setup.py" ] || {
  echo "No pyproject.toml/setup.py in $TRL_REPO"
  exit 1
}
echo "Using WORKSPACE_ROOT=$WORKSPACE_ROOT"
echo "Using TRL_REPO=$TRL_REPO"
cd "$TRL_REPO"
pip install -e ".[vllm]"

# Align runtime deps with the repo requirements and keep vLLM-compatible versions.
# (vLLM 0.12 requires transformers<5; peft is imported directly in train_grpo_smoke.py)
python -m pip install -U \
  -r "$TRL_REPO/requirements.txt" \
  "huggingface_hub<1" \
  "peft<0.18" \
  "deepspeed" \
  wandb \
  math_verify \
  bitsandbytes



# Pin vLLM if your cluster has surprises.
# TRL is version-sensitive; current docs list supported versions and examples. :contentReference[oaicite:2]{index=2}
# Uncomment if needed:
# pip install "vllm==0.10.2"

# Quick sanity prints and import checks
python -c "import trl, peft, accelerate, datasets, transformers, vllm, wandb, sys; print('TRL:', trl.__version__ if hasattr(trl,'__version__') else 'dev', 'at', trl.__file__); print('transformers:', transformers.__version__, 'peft:', peft.__version__, 'vllm:', vllm.__version__, 'wandb:', wandb.__version__); print(sys.executable)"

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
HEALTH_WAIT_SECONDS="${HEALTH_WAIT_SECONDS:-300}"
for ((i=1; i<=HEALTH_WAIT_SECONDS; i++)); do
  # If the server process has already exited, fail immediately with useful context.
  if ! ps -p "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM server process exited before becoming healthy."
    tail -n 50 "logs/vllm_server_${SLURM_JOB_ID:-manual}.log" || true
    exit 1
  fi

  if curl -s --max-time 2 --noproxy "*" "http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/health/" 2>/dev/null | grep -q '"status"'; then
    echo "vLLM server is healthy."
    break
  fi

  # Periodic status line so long startups are visible in logs.
  if (( i % 30 == 0 )); then
    echo "Still waiting for vLLM health (${i}s/${HEALTH_WAIT_SECONDS}s)..."
  fi

  sleep 1
  if [ "$i" -eq "$HEALTH_WAIT_SECONDS" ]; then
    echo "vLLM server did not become healthy. Check logs/vllm_server_${SLURM_JOB_ID:-manual}.log"
    kill $VLLM_PID || true
    exit 1
  fi
done

# --- Run a tiny GRPO job on GPU1 ---
export CUDA_VISIBLE_DEVICES=1

export MODEL_ID="$MODEL_ID"
export RUN_NAME="$RUN_NAME"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_ENTITY="$WANDB_ENTITY"
export WANDB_NAME="$RUN_NAME"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${SLURM_JOB_ID:-manual}-${RUN_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
echo "Starting GRPO smoke test on GPU1..."
CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  "$TRL_REPO/run_test.py" \
  --vllm-sync-strategy "lora_adapter" \
  --run-name "$RUN_NAME" \
  --env "repetition" \
  --quantized 
# Cleanup server
echo "Stopping vLLM server..."
kill $VLLM_PID || true
wait $VLLM_PID || true

echo "Job complete."