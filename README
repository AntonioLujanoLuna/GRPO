# README

## Overview
This project provides an implementation of a fine-tuning pipeline for a causal language model using Group Relative Policy Optimization (GRPO) techniques. The pipeline includes:

- **Data Preprocessing**: Extracting and formatting data from the GSM8K dataset.
- **Fine-tuning**: Training a `distilgpt2` model with LoRA (Low-Rank Adaptation) and quantization for efficiency.
- **Evaluation**: Computing structured response accuracy using custom-defined reward functions.
- **Logging & Monitoring**: Integration with `wandb` for experiment tracking.

## Features
- Uses **LoRA** to enable parameter-efficient training.
- Implements **reward functions** for evaluating structured response correctness.
- Supports **gradient checkpointing** to optimize memory usage.
- Enables **quantized model loading** with BitsAndBytes (`bnb`).
- Provides **evaluation tools** for structured reasoning accuracy.

## Dependencies
Ensure the following packages are installed before running the pipeline. 

```sh
pip install torch transformers peft trl datasets wandb rapidfuzz tqdm numpy dotenv
```

Or more easily, just 

```sh
pip install -e .
```

## Training & Evaluation
### 1. Load Environment Variables
Ensure that `.env` contains the necessary API keys, particularly for `wandb` logging:
```sh
WANDB_API_KEY=<your_wandb_api_key>
```

### 2. Run Training
Execute the script to start training:
```sh
python main.py
```
This will:
- Load and preprocess the GSM8K dataset.
- Fine-tune `distilgpt2` using LoRA.
- Evaluate the model using structured reasoning accuracy metrics.
- Save the fine-tuned model and tokenizer.

### 3. Evaluate Model Performance
The trained model is evaluated using a suite of reward functions, including:
- **Correctness Reward**: Ensures the response matches the ground truth.
- **Format Reward**: Checks if the response adheres to XML structure.
- **Conciseness Reward**: Penalizes overly verbose responses.
- **Repetition Penalty**: Penalizes excessive token repetition.
- **Chain-of-Thought Clarity Reward**: Rewards well-structured reasoning.

Results will be logged in the console and `wandb`.

## Saving & Loading Models
The fine-tuned model is saved in the `outputs/distilgpt2-GRPO/` directory. To load it for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "outputs/distilgpt2-GRPO"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

## Customization
### Adjusting Hyperparameters
Modify `training_args` in `main.py` to tweak learning rates, batch sizes, and training epochs.

### Adding New Reward Functions
New reward functions can be added within the script. Ensure they return a list of float values corresponding to reward scores.

## Contributors
This project is developed to optimize structured reasoning performance in language models. Contributions and feedback are welcome!

## License
This project is released under the MIT License.

