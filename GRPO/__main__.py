import re
from functools import lru_cache
from rapidfuzz import fuzz
import os
from dotenv import load_dotenv
import torch
import torch.utils.checkpoint as checkpoint
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import wandb
import warnings
import tqdm
import numpy as np
import torch.distributed as dist


# -----------------------------------------------------------------------------
# Precompiled regex patterns and system prompt for XML formatting.
# -----------------------------------------------------------------------------
FORMAT_PATTERN = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)
STRICT_FORMAT_PATTERN = re.compile(
    r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$", re.DOTALL
)
REASONING_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# -----------------------------------------------------------------------------
# Original helper functions and reward functions.
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1024)
def extract_xml_answer_cached(text: str) -> str:
    """
    Cached extraction of the answer from a response using <answer> tags.
    """
    try:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    except IndexError:
        return ""

def compute_response_cache(completions, gold=None):
    """
    Precompute a cache for each completion.
    """
    cache = []
    for idx, comp in enumerate(completions):
        response = comp[0]['content']
        extracted = extract_xml_answer_cached(response)
        cache_item = {
            'response': response,
            'extracted': extracted,
            'lower_response': response.lower().strip(),
            'lower_extracted': extracted.lower().strip()
        }
        if gold is not None and idx < len(gold):
            gold_item = gold[idx].strip()
            cache_item['gold'] = gold_item
            cache_item['lower_gold'] = gold_item.lower().strip()
        cache.append(cache_item)
    return cache

def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Computes a reward based on how well the extracted answer matches the expected answer.
    """
    cache = kwargs.get("cache", None)
    if cache is None:
        cache = compute_response_cache(completions, gold=answer)

    rewards = []
    for item in cache:
        if item['extracted'] == item.get('gold', item['extracted']):
            rewards.append(2.0)
        else:
            sim_score = fuzz.ratio(item['lower_extracted'], item.get('lower_gold', item['lower_extracted']))
            if sim_score >= 80:
                rewards.append(1.0 + sim_score / 100.0)
            else:
                rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs) -> list[float]:
    """
    Awards a reward if the response follows the XML format.
    """
    cache = kwargs.get("cache", None)
    if cache is None:
        responses = [comp[0]['content'] for comp in completions]
    else:
        responses = [item['response'] for item in cache]
    return [0.5 if FORMAT_PATTERN.search(response) else 0.0 for response in responses]

def strict_format_reward(completions, **kwargs) -> list[float]:
    """
    Awards a reward if the response strictly follows the XML format (with newlines).
    """
    cache = kwargs.get("cache", None)
    if cache is None:
        responses = [comp[0]['content'] for comp in completions]
    else:
        responses = [item['response'] for item in cache]
    return [0.5 if STRICT_FORMAT_PATTERN.match(response) else 0.0 for response in responses]

def numeric_reward(completions, **kwargs) -> list[float]:
    """
    Awards a bonus if the extracted answer is purely numeric.
    """
    cache = kwargs.get("cache", None)
    if cache is None:
        responses = [comp[0]['content'] for comp in completions]
        extracted = [extract_xml_answer_cached(response) for response in responses]
    else:
        extracted = [item['extracted'] for item in cache]
    return [0.5 if r.strip().isdigit() else 0.0 for r in extracted]

def conciseness_reward(completions, **kwargs) -> list[float]:
    """
    Penalizes overly verbose reasoning.
    """
    cache = kwargs.get("cache", None)
    if cache is None:
        responses = [comp[0]['content'] for comp in completions]
    else:
        responses = [item['response'] for item in cache]

    rewards = []
    for response in responses:
        match = REASONING_PATTERN.search(response)
        if match:
            reasoning_text = match.group(1).strip()
            word_count = len(reasoning_text.split())
            penalty = max(0, (word_count - 50) * 0.01)
            penalty = min(penalty, 0.5)
            rewards.append(-penalty)
        else:
            rewards.append(0.0)
    return rewards

def repetition_penalty(completions, **kwargs) -> list[float]:
    """
    Penalize repetition in the entire response.
    """
    cache = kwargs.get("cache", None)
    if cache is None:
        responses = [comp[0]['content'] for comp in completions]
    else:
        responses = [item['response'] for item in cache]

    penalties = []
    for response in responses:
        tokens = re.findall(r"\w+", response.lower())
        if not tokens:
            penalties.append(0.0)
            continue

        unique_tokens = set(tokens)
        repeat_ratio = (len(tokens) - len(unique_tokens)) / len(tokens)
        if repeat_ratio < 0.2:
            penalty = 0.0
        elif repeat_ratio < 0.5:
            penalty = -0.25
        else:
            penalty = -0.5
        penalties.append(penalty)
    return penalties

def cot_clarity_reward(completions, **kwargs) -> list[float]:
    """
    Rewards well-structured, stepwise reasoning.
    """
    LOGIC_KEYWORDS = ["first", "second", "third", "then", "next", "therefore", "thus", "hence"]
    cache = kwargs.get("cache", None)
    if cache is None:
        responses = [comp[0]['content'] for comp in completions]
    else:
        responses = [item['response'] for item in cache]

    rewards = []
    for response in responses:
        match = REASONING_PATTERN.search(response)
        if match:
            reasoning_text = match.group(1).lower()
            keyword_count = sum(reasoning_text.count(kw) for kw in LOGIC_KEYWORDS)
            reward = min(keyword_count * 0.1, 0.5)
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

def extract_hash_answer(text: str) -> str:
    """
    Extracts the answer from a string using a hash delimiter.
    """
    if "####" not in text:
        return ""
    return text.split("####")[1].strip()

def fuzzy_match(answer1: str, answer2: str, threshold: float = 80.0) -> bool:
    """
    Returns True if the similarity between answer1 and answer2 meets the threshold.
    """
    similarity = fuzz.ratio(answer1.lower().strip(), answer2.lower().strip())
    return similarity >= threshold

def get_gsm8k_questions(split="train"):
    """
    Loads and preprocesses the GSM8K dataset.
    """
    data = load_dataset('openai/gsm8k', 'main')[split]

    def preprocess(example):
        return {
            "prompt": [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': example['question']}
            ],
            "answer": extract_hash_answer(example['answer'])
        }

    processed_data = data.map(preprocess, remove_columns=data.column_names)
    return processed_data

# -----------------------------------------------------------------------------
# Torch and warning configurations.
# -----------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    module="torch"
)
warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
    module="torch"
)

os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Monkey-patch torch.utils.checkpoint to always pass use_reentrant=False.
_original_checkpoint = checkpoint.checkpoint
def custom_checkpoint(function, *args, **kwargs):
    if "use_reentrant" not in kwargs:
        kwargs["use_reentrant"] = False
    return _original_checkpoint(function, *args, **kwargs)
checkpoint.checkpoint = custom_checkpoint

# -----------------------------------------------------------------------------
# Original evaluation method (computing average rewards over reward functions).
# -----------------------------------------------------------------------------
def evaluate_model(model, tokenizer, dataset, reward_funcs, device):
    """
    Evaluate the given model on the dataset using the specified reward functions.
    Returns a dictionary mapping reward function names to their average scores.
    """
    model.eval()
    rewards_sum = {func.__name__: 0.0 for func in reward_funcs}
    num_examples = 0

    for example in dataset:
        prompt = example['prompt']
        encoded = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded.input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=128)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        completions = [[{'content': response}]]

        for func in reward_funcs:
            if func.__name__ in ["correctness_reward", "my_combined_reward"]:
                gold = [example['answer']]
                reward_value = func(None, completions, gold)[0]
            else:
                reward_value = func(completions)[0]
            rewards_sum[func.__name__] += reward_value

        num_examples += 1

    avg_rewards = {name: total / num_examples for name, total in rewards_sum.items()}
    return avg_rewards

# -----------------------------------------------------------------------------
# New extension: helper functions for evaluation via the CustomTrainer.
# -----------------------------------------------------------------------------
def extract_xml_answer(text: str) -> str:
    """
    Extracts the final answer from the response.
    Checks for <final_answer> tags first, then falls back to <answer> tags.
    """
    if "<final_answer>" in text:
        return text.split("<final_answer>")[-1].split("</final_answer>")[0].strip()
    elif "<answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        return ""

def generate_gsm8k(model, tokenizer, tokenized_samples, batch_size, max_completion_length):
    """
    Generates responses for GSM8K evaluation and computes accuracy.
    """
    # If distributed is not available, assume rank 0.
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        device = model.device
        predictions = []
        generation_config = GenerationConfig(
            max_new_tokens=max_completion_length,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model.eval()
        count = len(tokenized_samples)
        status = tqdm.tqdm(tokenized_samples, desc=f"Correct: 0/{count}")
        for i in range(0, count, batch_size):
            batches = tokenized_samples[i:i+batch_size]
            with torch.inference_mode():
                longest = max(len(b[0]) for b in batches)
                # Pad on the left for each sample.
                padded_input_ids = torch.stack([
                    torch.tensor([tokenizer.pad_token_id] * (longest - len(ids)) + ids)
                    for ids, _ in batches
                ]).to(device)
                # Create attention mask (ignoring pad tokens).
                attn_mask = torch.stack([
                    tokens.ne(tokenizer.pad_token_id) for tokens in padded_input_ids
                ]).to(device)

                output = model.generate(
                    input_ids=padded_input_ids,
                    attention_mask=attn_mask,
                    generation_config=generation_config,
                )

                for j, generated in enumerate(output):
                    response = tokenizer.decode(
                        generated[len(padded_input_ids[j]):], skip_special_tokens=True
                    )
                    prediction = extract_xml_answer(response)
                    # Compare the generated prediction to the gold answer.
                    predictions.append(batches[j][1] == prediction)
                status.update(batch_size)
                status.set_description(f"Correct: {sum(predictions)}/{count}")
        return np.mean(predictions)
    return 0

def tokenize_validation(tokenizer, samples, max_prompt_length):
    """
    Tokenizes validation samples using the tokenizer's chat template.
    """
    tokenized_samples = []
    for sample in samples:
        prompt = sample["prompt"]
        answer = sample['answer']
        if hasattr(tokenizer, "apply_chat_template"):
            ids = tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                truncation=False,
                max_length=max_prompt_length,
            )
        else:
            # Simple fallback if apply_chat_template is not available.
            text = ""
            for message in prompt:
                text += f"{message['role']}: {message['content']}\n"
            text += "assistant: "
            ids = tokenizer.encode(text, truncation=False, max_length=max_prompt_length)
        tokenized_samples.append((ids, answer))
    return tokenized_samples

class CustomTrainer(GRPOTrainer):
    """
    Custom trainer that overrides the evaluate() method to compute an accuracy metric on GSM8K.
    """
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        tokenized_samples = tokenize_validation(
            self.processing_class, self.eval_dataset, self.args.max_prompt_length
        )
        eval_acc = generate_gsm8k(
            self.model,
            self.processing_class,
            tokenized_samples,
            self.args.per_device_eval_batch_size,
            self.args.max_completion_length
        )

        output = {
            f"{metric_key_prefix}_accuracy": eval_acc,
            "epoch": self.state.epoch,
        }

        self.log(output)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output
        )

        return output

# -----------------------------------------------------------------------------
# Main function: training and evaluation.
# -----------------------------------------------------------------------------
def main():
    # Load environment variables from the .env file
    load_dotenv()

    # Optionally, verify that the WANDB_API_KEY is loaded
    print("WandB API Key:", os.getenv("WANDB_API_KEY"))

    # Initialize wandb.
    wandb.init(project="gsm8k-grpo", name="distilgpt2-grpo-demo")

    model_name = "distilgpt2"
    output_dir = "outputs/distilgpt2-GRPO"
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare the dataset.
    dataset = get_gsm8k_questions("train")
    test_dataset = get_gsm8k_questions("test")

    # Configure quantization for memory efficiency.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Set up PEFT/LoRA configuration.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["c_attn"],  # For distilgpt2.
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # Load model with quantization.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Enable gradient checkpointing for memory savings.
    try:
        model.gradient_checkpointing_enable(use_reentrant=False)
    except TypeError:
        model.gradient_checkpointing_enable()

    # Prepare model for k-bit training and apply LoRA modifications.
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Patch the model's forward method to handle num_logits_to_keep if present.
    original_forward = model.forward
    def patched_forward(*args, num_logits_to_keep=None, **kwargs):
        if "num_logits_to_keep" in kwargs:
            del kwargs["num_logits_to_keep"]
        return original_forward(*args, **kwargs)
    model.forward = patched_forward

    # Load and configure tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"

    # Add an apply_chat_template method if not already present.
    if not hasattr(tokenizer, "apply_chat_template"):
        def apply_chat_template(prompt, add_generation_prompt=True, truncation=False, max_length=None):
            text = ""
            for message in prompt:
                text += f"{message['role']}: {message['content']}\n"
            if add_generation_prompt:
                text += "assistant: "
            return tokenizer.encode(text, truncation=truncation, max_length=max_length)
        tokenizer.apply_chat_template = apply_chat_template

    # Training configuration (with evaluation parameters added).
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=128,
        max_completion_length=128,
        num_train_epochs=1,
        logging_steps=10,
        num_generations=2,
        save_steps=50,
        fp16=True,
        report_to="wandb",
        eval_steps=20,
        per_device_eval_batch_size=256,
        do_eval=True,
        eval_strategy="steps",
        run_name="GTPO_demo"
    )

    # Use the original reward functions for training.
    reward_functions = [
        correctness_reward,
        format_reward,
        strict_format_reward,
        numeric_reward,
        conciseness_reward,
        repetition_penalty,
        cot_clarity_reward
    ]

    # Initialize the custom trainer.
    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        reward_funcs=reward_functions
    )

    # Start training.
    trainer.train()

    # ----- Saving the Fine-Tuned Model and Tokenizer -----
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model and tokenizer have been saved to {output_dir}.")

    # ----- Evaluating the Fine-Tuned Model using the Custom Accuracy Metric -----
    print("Evaluating fine-tuned model with custom accuracy metric...")
    ft_eval_results = trainer.evaluate()
    print("Fine-tuned model evaluation results:")
    for key, value in ft_eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # ----- Evaluating the Base Model for Comparison -----
    print("Evaluating base distilgpt2 model...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(next(model.parameters()).device)
    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    if not hasattr(base_tokenizer, "chat_template") or base_tokenizer.chat_template is None:
        base_tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"

    base_eval_results = evaluate_model(
        base_model, base_tokenizer, test_dataset, reward_functions, next(model.parameters()).device
    )
    print("Base model average rewards (combined):")
    for func_name, avg_reward in base_eval_results.items():
        print(f"  {func_name}: {avg_reward:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
