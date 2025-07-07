# Sorachio-1B-Chat

## Overview

Sorachio-1B-Chat is a fine-tuned conversational AI model based on Gemma3, specifically optimized for Indonesian language conversations. This model was trained using Supervised Fine-Tuning (SFT) with QLoRA adapter techniques and subsequently merged for optimal performance.

## Model Details

- **Base Model**: Gemma3
- **Model Size**: 1B parameters
- **Training Method**: Supervised Fine-Tuning (SFT) with Quantized Low-Rank Adaptation (QLoRA) 
- **Training Infrastructure**: Google Colab T4 GPU (Free Tier)
- **Dataset**: [IzzulGod/gpt4o-distill-chat-v1](https://huggingface.co/datasets/IzzulGod/gpt4o-distill-chat-v1)

## Training Configuration

### Dataset Information
- **Total Samples**: 1,316 conversation samples
- **Conversation Types**: Mixed single-turn and multi-turn conversations (primarily multi-turn for enhanced conversational capabilities)
- **Data Source**: GPT-4o distilled chat dataset

### Training Details
- **Training Epochs**: 3
- **Total Steps**: 432
- **Final Training Loss**: 2.277
- **Training Runtime**: 564.55 seconds
- **Training Samples per Second**: 6.085
- **Training Steps per Second**: 0.765

### Training Hyperparameters
- **Batch Size**: 4 per device
- **Gradient Accumulation Steps**: 2
- **Effective Batch Size**: 8 (4 × 2)
- **Learning Rate**: 2e-4
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.1
- **Learning Rate Scheduler**: Cosine
- **Optimizer**: AdamW 8-bit
- **Precision**: FP16 (Mixed Precision)
- **Additional Features**: Group by length for efficiency

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "IzzulGod/Sorachio-1B-Chat"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager"
).eval()

# Prepare conversation
messages = [
    {"role": "user", "content": "Coba perkenalkan dirimu"}
]

# Tokenize input
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode output
output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(output_text)
```

### Example Output

> Halo! Aku Sorachio, asisten AI yang siap menemani kamu dalam percakapan apa pun.
Aku dirancang untuk jadi teman yang bisa kamu ajak ngobrol tanpa batas, dari topik ringan sampai diskusi mendalam.
Bahkan kalau kamu cuma pengen curhat atau sekadar ngobrol random, aku siap nemenin! 😄

## Model Downloads

### GGUF Format
- **[Q8_0 Quantized](https://huggingface.co/IzzulGod/Sorachio-1B-Chat/resolve/main/sorachio-1b-chat-q8_0.gguf?download=true)** - Recommended for most use cases (fast inference, high quality)
- **[F16 Full Precision](https://huggingface.co/IzzulGod/Sorachio-1B-Chat/resolve/main/sorachio-1b-chat-f16.gguf?download=true)** - Full precision model (maximum quality, slower inference)

## Training Progress

| Step | Training Loss |
|------|---------------|
| 40   | 4.875200     |
| 80   | 2.665700     |
| 120  | 2.344800     |
| 160  | 2.242600     |
| 200  | 2.007000     |
| 240  | 2.030400     |
| 280  | 1.973100     |
| 320  | 1.758300     |
| 360  | 1.653700     |
| 400  | 1.709000     |

**Final Training Metrics:**
- Global Step: 432
- Training Loss: 2.277
- Epoch: 3.0
- Total FLOPs: 1,371,161,336,361,216

## Technical Specifications

### Model Architecture
- **Framework**: Transformers
- **Precision**: Float16 (recommended)
- **Attention Implementation**: Eager
- **Device Mapping**: Auto

## Use Cases

- Indonesian conversational AI
- Chatbot applications
- Educational tools
- Customer service automation
- General Indonesian language tasks

## Limitations

- Optimized primarily for Indonesian language
- Model size limited to 1B parameters
- Trained on Google Colab T4 GPU constraints
- Performance may vary with different hardware configurations

## License

Please refer to the original Gemma3 license terms for usage guidelines.
