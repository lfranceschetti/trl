import os
import random

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

import argparse
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# ── Word-repetition environment ────────────────────────────────────────────────

WORDS = [
    "hello", "cat", "dog", "yes", "no", "blue", "red", "green", "sun", "moon",
    "fish", "bird", "tree", "rock", "fire", "water", "star", "cloud", "rain", "snow",
    "book", "lamp", "door", "ring", "bell", "cake", "ship", "leaf", "wolf", "bear",
    "gold", "iron", "silk", "drum", "frog", "seed", "wave", "hill", "sand", "pearl",
]


def make_repetition_dataset(n=750):
    samples = []
    for _ in range(n):
        word = random.choice(WORDS)
        count = random.randint(3, 50)
        samples.append({
            "prompt": [{"role": "user", "content": f'Repeat the word "{word}" exactly {count} times, separated by spaces. Output nothing else.'}],
            "target_word": word,
            "target_count": count,
        })
    return Dataset.from_list(samples)


def repetition_reward(completions, target_word, target_count, **kwargs):
    rewards = []
    for completion, word, count in zip(completions, target_word, target_count):
        content = completion if isinstance(completion, str) else completion[0].get("content", "")
        tokens = content.strip().split()
        count = int(count)

        if not tokens:
            rewards.append(0.0)
            continue

        correct_tokens = sum(1 for t in tokens if t.strip().lower().strip(".,!?\"'") == word.lower())
        word_accuracy = correct_tokens / max(len(tokens), 1)

        count_accuracy = 1.0 - min(abs(len(tokens) - count) / count, 1.0)

        exact_bonus = 0.2 if len(tokens) == count and correct_tokens == count else 0.0

        reward = (count_accuracy * 0.4 + word_accuracy * 0.4 + exact_bonus)
        rewards.append(reward)
    return rewards


# ── Random-number environment ──────────────────────────────────────────────────

def reward_function(completions, **kwargs):
    rewards = []
    for completion in completions:
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cli_args: argparse.Namespace) -> None:
    host = os.environ.get("VLLM_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("VLLM_SERVER_PORT", "8000"))
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")
    base_run_name = cli_args.run_name or os.environ.get("RUN_NAME", "grpo_lora_length_reward")
    run_name = f"{base_run_name}_quantized" if cli_args.quantized else base_run_name
    output_dir = f"outputs/{run_name}"

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

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if cli_args.env == "math":
        dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
        run_name = run_name + "_math"
        output_dir = f"outputs/{run_name}"
        train_args = GRPOConfig(
            output_dir=output_dir,
            run_name=run_name,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            steps_per_generation=16,
            num_generations=8,
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=host,
            vllm_server_port=port,
            vllm_sync_strategy=cli_args.vllm_sync_strategy,
            logging_steps=25,
            temperature=0.7,
            top_p=0.9,
            save_strategy="no",
            report_to="wandb",
        )
        trainer = GRPOTrainer(
            model=model,
            args=train_args,
            reward_funcs=accuracy_reward,
            train_dataset=dataset,
            peft_config=peft_config,
            tools=None,
        )

    elif cli_args.env == "repetition":
        dataset = make_repetition_dataset(n=3000)
        run_name = run_name + "_repetition"
        output_dir = f"outputs/{run_name}"
        train_args = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            run_name=run_name,
            num_generations=8,
            num_train_epochs=1,
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=host,
            vllm_server_port=port,
            vllm_sync_strategy=cli_args.vllm_sync_strategy,
            logging_steps=25,
            save_strategy="no",
            report_to="wandb",
        )
        trainer = GRPOTrainer(
            model=model,
            args=train_args,
            reward_funcs=repetition_reward,
            train_dataset=dataset,
            peft_config=peft_config,
            tools=None,
        )

    else:  # length
        samples = [
            {
                "prompt": "Give me a random number between 0 and 100. Only respond with the number, nothing else.",
            }
            for _ in range(750)
        ]
        dataset = Dataset.from_list(samples)

        train_args = GRPOConfig(
            output_dir=output_dir,
            run_name=run_name,
            max_steps=750,
            num_train_epochs=1,
            learning_rate=5e-5,
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=host,
            vllm_server_port=port,
            vllm_sync_strategy=cli_args.vllm_sync_strategy,
            logging_steps=5,
            save_strategy="no",
            report_to="wandb",
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
    parser.add_argument("--env", type=str, default="math", choices=["length", "math", "repetition"])
    parser.add_argument("--quantized", action="store_true", default=False)
    parser.add_argument("--vllm-sync-strategy", type=str, default="weights", choices=["weights", "lora_adapter"])
    parser.add_argument("--run-name", type=str, default=None)
    cli_args = parser.parse_args()
    main(cli_args)