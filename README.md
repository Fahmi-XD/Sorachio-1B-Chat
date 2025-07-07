# Sorachio-1B-Chat

## Overview
Sorachio-1B-Chat is a fine-tuned conversational AI model based on Gemma3, specifically optimized for natural conversations. This multilingual model was trained using Supervised Fine-Tuning (SFT) with QLoRA adapter techniques and subsequently merged for optimal performance.

## Model Details
- **Base Model**: Gemma3
- **Model Size**: 1B parameters
- **Language Support**: Multilingual (Indonesian, English, and more)
- **Training Method**: Supervised Fine-Tuning (SFT) with Quantized Low-Rank Adaptation (QLoRA) 
- **Training Infrastructure**: Google Colab T4 GPU (Free Tier)
- **Dataset**: [IzzulGod/gpt4o-distill-chat-v1](https://huggingface.co/datasets/IzzulGod/gpt4o-distill-chat-v1)

## Training Configuration

### Dataset Information
- **Total Samples**: 1,316 conversation samples
- **Conversation Types**: Mixed single-turn and multi-turn conversations (primarily multi-turn for enhanced conversational capabilities)
- **Data Source**: GPT-4o distilled chat dataset

### Training Setup
The model was trained with carefully optimized parameters to achieve the best performance within resource constraints:

- **Batch Size**: 4 per device with gradient accumulation (effective batch size: 8)
- **Training Epochs**: 3 full passes through the dataset
- **Learning Rate**: 2e-4 with cosine decay schedule
- **Optimizer**: AdamW 8-bit for memory efficiency
- **Precision**: Mixed FP16 training for faster convergence
- **Training Time**: ~12 minutes (495 steps total)

### Training Performance
The model showed consistent improvement throughout training, with loss decreasing from 5.48 to 1.68 over 495 training steps. The final training loss of 2.24 indicates good convergence without overfitting.

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
> Aku dirancang untuk jadi teman yang bisa kamu ajak ngobrol tanpa batas, dari topik ringan sampai diskusi mendalam.
> Bahkan kalau kamu cuma pengen curhat atau sekadar ngobrol random, aku siap nemenin! 😄

## Model Downloads

### GGUF Format
- **[Q8_0 Quantized](https://huggingface.co/IzzulGod/Sorachio-1B-Chat/resolve/main/sorachio-1b-chat-q8_0.gguf?download=true)** - Recommended for most use cases (fast inference, high quality)
- **[F16 Full Precision](https://huggingface.co/IzzulGod/Sorachio-1B-Chat/resolve/main/sorachio-1b-chat-f16.gguf?download=true)** - Full precision model (maximum quality, slower inference)

## Performance & Capabilities

- **Natural Conversations**: Trained on multi-turn dialogues for more engaging and contextual responses
- **Efficient Training**: Achieved good performance with minimal computational resources
- **Quick Inference**: Optimized for fast response generation

## Limitations

- Model size limited to 1B parameters
- Trained on Google Colab T4 GPU constraints
- Performance may vary with different hardware configurations

## License

Please refer to the original Gemma3 license terms for usage guidelines.
