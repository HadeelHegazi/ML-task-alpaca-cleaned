
# Fine-Tuning Language Models with Unsloth and Xformers

This repository contains code for fine-tuning a language model using Unsloth and Xformers. The code is designed to facilitate efficient model training and inference on large datasets, with a focus on optimizing memory usage and training speed.

## Overview

This script demonstrates the process of installing necessary packages, loading a pre-trained model from the Unsloth repository, and fine-tuning the model using a specific dataset. The model used in this example is "Meta-Llama-3.1-8B," which has been quantized to 4-bit precision to reduce memory usage and improve training efficiency.

## Installation

The following packages are required for the fine-tuning process:

- `unsloth`: A library for fast language model fine-tuning.
- `xformers`: A library that provides efficient attention mechanisms, including Flash Attention.
- `trl`, `peft`, `accelerate`, `bitsandbytes`: Libraries for training large models.

To install these packages, use the following commands:

```bash
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip uninstall xformers
!pip install xformers
!python -m xformers.info
!pip install triton
```

## Model Configuration

The model configuration includes setting parameters for maximum sequence length, data type detection, and 4-bit quantization. The model is loaded from the Unsloth repository and configured with LoRA (Low-Rank Adaptation) for fine-tuning.

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # Auto detection
load_in_4bit = True  # Use 4bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

## Dataset Preparation

The dataset used for fine-tuning is the Alpaca dataset, which is formatted to include instructions, inputs, and expected responses. The dataset is processed to fit the model’s input requirements.

```python
from datasets import load_dataset

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
```

## Fine-Tuning Process

The model is fine-tuned using the SFTTrainer from the `trl` library. Key training parameters include batch size, learning rate, and the number of training steps.

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=120,
        learning_rate=2e-4,
        output_dir="outputs",
    ),
)
trainer.train()
```

## Evaluation

Before and after training, the model is tested using predefined prompts to compare its performance. The outputs demonstrate the model's ability to generate responses based on the given instructions and inputs.

### Example Output Before Training

```python
# Sample prompt for testing before training
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Based on the information provided, rewrite the sentence by changing its tense from past to future.",
            "She played the piano beautifully for hours and then stopped as it was midnight.",
            "",
        )
    ], return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128)
```

### Example Output After Training

```python
# Sample prompt for testing after training
outputs = model.generate(**inputs, max_new_tokens=128)
```

## Model Saving

The fine-tuned model can be saved locally or pushed to the Hugging Face Hub for public access.

```python
model.push_to_hub("HadeelHegazi/lora_model_num_train_epochs1max_steps120", token="hf_your_token")
tokenizer.push_to_hub("HadeelHegazi/lora_model_num_train_epochs1max_steps120", token="hf_your_token")
```

## Errors

### 1. **CUDA Out of Memory Error:**
   - **Cause:** Your GPU doesn't have enough memory to allocate for the current task.
   - **Solutions:**
     - **Reduce batch size:** Lowering the batch size during training or inference can help reduce memory usage.
     - **Clear unused variables:** Ensure you're clearing any unused variables or using `torch.cuda.empty_cache()` to free up memory.
     - **Use `PYTORCH_CUDA_ALLOC_CONF`:** Set the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to manage memory fragmentation. This can be done in your script or in your environment setup.
     - **Switch to mixed precision:** If applicable, you could try using mixed-precision training (e.g., with `torch.cuda.amp`) to reduce memory usage.

### 2. **Xformers Installation Error:**
   - **Cause:** Xformers was installed for a different version of PyTorch and CUDA than what you have in your environment.
   - **Solutions:**
     - **Reinstall Xformers:** Ensure that you install Xformers compatible with your current PyTorch and CUDA versions. You can do this by running:
       ```bash
       pip uninstall xformers
       pip install xformers --no-cache-dir
       ```
     - **Check your environment:** Verify that the CUDA and PyTorch versions are consistent with the requirements of Xformers. For example, make sure that the version of PyTorch matches the one Xformers was built for.
     - **Confirm installation:** After reinstalling, confirm that Xformers is installed correctly by running:
       ```bash
       python -m xformers.info
       ```

### Steps to Resolve:
1. Adjust your code to reduce GPU memory usage.
2. Reinstall Xformers with the correct versions to match your setup.
3. Use environment variables to handle memory more effectively if fragmentation is an issue.


## Conclusion

This script provides a comprehensive guide to fine-tuning a large language model using Unsloth and Xformers. The included code can be easily adapted to different datasets and models for various NLP tasks.










---
base_model: unsloth/meta-llama-3.1-8b-bnb-4bit
language:
- en
license: apache-2.0
tags:
- text-generation-inference
- transformers
- unsloth
- llama
- trl
---

# Uploaded  model

- **Developed by:** HadeelHegazi
- **License:** apache-2.0
- **Finetuned from model :** unsloth/meta-llama-3.1-8b-bnb-4bit

This llama model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
