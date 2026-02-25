import os

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward


def main() -> None:
    host = os.environ.get("VLLM_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("VLLM_SERVER_PORT", "8000"))
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")

    # Small dataset slice for speed; just need a smoke test.
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    args = GRPOConfig(
        output_dir="outputs/grpo_lora_sync",
        run_name="grpo_lora_sync",
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=host,
        vllm_server_port=port,
        # This is your new flag:
        vllm_sync_strategy="lora_adapter",
        vllm_lora_rank=64,
        logging_steps=5,
        save_strategy="no",
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model_id,
        args=args,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        peft_config=peft_config,
        tools=None,
    )

    trainer.train()
    print("DONE")


if __name__ == "__main__":
    main()
