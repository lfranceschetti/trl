import os

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import argparse
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def reward_function(completions, **kwargs):
    rewards = []
    for completion in completions:
        # GRPO completion format is typically a single assistant message in a one-item list.
        print("Completion:", completion)
        if isinstance(completion, str):
            content = completion
        else:
            content = completion[0].get("content", "") if completion else ""
        
        try:
            number = int(content)
        except ValueError:
            number = 0

        if number < 0 or number > 100:
            reward = 0.0
        else: 
            reward = float(number)
        rewards.append(reward)

    print("Rewards:", rewards)
    return rewards




def main(cli_args: argparse.Namespace) -> None:
    host = os.environ.get("VLLM_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("VLLM_SERVER_PORT", "8000"))
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")

    samples = [
        {
            "prompt": "Give me a random number between 0 and 100. Only respond with the number, nothing else.",
        }
        for _ in range(1000)
    ]

    dataset = Dataset.from_list(samples)

    # Small dataset slice for speed; just need a smoke test.
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    train_args = GRPOConfig(
        output_dir="outputs/grpo_lora_length_reward",
        run_name="grpo_lora_length_reward",
        learning_rate=5e-5,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=host,
        vllm_server_port=port,
        # This is your new flag:
        vllm_sync_strategy=cli_args.vllm_sync_strategy,
        logging_steps=5,
        save_strategy="no",
        report_to="wandb",
    )

    if cli_args.quantized:
        print("Loading quantized model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    else:
        bnb_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    trainer = GRPOTrainer(
        model=model,
        args=train_args,
        reward_funcs=reward_function,
        train_dataset=dataset,
        peft_config=peft_config,
        tools=None,
    )


    trainer.train()
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized", action="store_true", default=False)
    parser.add_argument("--vllm-sync-strategy", type=str, default="weights", choices=["weights", "lora_adapter"])
    cli_args = parser.parse_args()
    main(cli_args)