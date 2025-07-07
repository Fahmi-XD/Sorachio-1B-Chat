# Sorachio-1B-Chat

[![Hugging Face Sorachio-1B-Chat](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/IzzulGod/Sorachio-1B-Chat)

## Overview

This repository contains the complete source code and training pipeline for **Sorachio-1B-Chat**, a fine-tuned conversational AI model based on Gemma3. The model is specifically optimized for natural conversations in Indonesian and other languages, trained using Supervised Fine-Tuning (SFT) with QLoRA adapter techniques.

🎯 **Ready to use model**: [IzzulGod/Sorachio-1B-Chat](https://huggingface.co/IzzulGod/Sorachio-1B-Chat)

## Repository Structure

```
📁 Sorachio-1B-Chat/
├── 📄 README.md                 # This file
├── 📓 clean-data.ipynb          # Data preprocessing and cleaning
├── 📓 convert-gguf.ipynb        # Model conversion to GGUF format 
└── 📓 fine-tune.ipynb           # Main training pipeline (SFT + QLoRA)
```

## Model Details

- **Base Model**: Gemma3
- **Model Size**: 1B parameters
- **Language Support**: Multilingual 
- **Training Method**: Supervised Fine-Tuning (SFT) with Quantized Low-Rank Adaptation (QLoRA)
- **Training Infrastructure**: Google Colab T4 GPU (Free Tier)
- **Dataset**: [IzzulGod/gpt4o-distill-chat-v1](https://huggingface.co/datasets/IzzulGod/gpt4o-distill-chat-v1)

## Notebooks Overview

### 1. 📓 `clean-data.ipynb`
Data preprocessing and cleaning pipeline:
- Raw dataset loading and exploration
- **Format conversion**: Transform raw conversation data from text format to JSON structure
- Data quality filtering and validation
- Conversation format standardization for training
- Dataset splitting and preparation

**Data Format Transformation:**
```
# Input (Raw Format):
User: Coba perkenalkan dirimu
Sorachio: Dengan senang hati! 😄
Hai, aku Sorachio. Aku adalah AI Assistant...
---
User: Kamu itu semacam AI gitu?
Sorachio: Bisa dibilang begitu, tapi aku bukan AI biasa 😌
---

# Output (JSON Format):
{"messages": [{"role": "user", "content": "Coba perkenalkan dirimu"}, {"role": "assistant", "content": "Dengan senang hati! 😄\nHai, aku Sorachio..."}]}
{"messages": [{"role": "user", "content": "Kamu itu semacam AI gitu?"}, {"role": "assistant", "content": "Bisa dibilang begitu, tapi aku bukan AI biasa 😌\nAku dibuat dengan gabungan teknologi AI modern..."}]}
```

### 2. 📓 `fine-tune.ipynb` (Main Training)
Complete fine-tuning pipeline:
- Model and tokenizer setup
- QLoRA configuration and adapter initialization
- Training loop with optimized hyperparameters
- Model evaluation and validation
- Adapter merging and final model export

**Training Configuration:**
- Batch Size: 4 per device (effective: 8 with gradient accumulation)
- Training Epochs: 3
- Learning Rate: 2e-4 with cosine decay
- Training Time: ~12 minutes (495 steps)
- Final Training Loss: 2.24

### 3. 📓 `convert-gguf.ipynb`
Model conversion and optimization:
- PyTorch to GGUF format conversion
- Quantization (Q8_0 and F16 variants)
- Model size optimization for deployment
- Compatibility testing

## Quick Start

### Option 1: Run in Google Colab
1. Open any notebook in Google Colab
2. Follow the step-by-step instructions in each notebook
3. All notebooks are designed to run in Colab's free tier

### Option 2: Use the Pre-trained Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "IzzulGod/Sorachio-1B-Chat"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

# Chat with the model
messages = [{"role": "user", "content": "Halo, apa kabar?"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

## Training Results

The model achieved excellent convergence during training:
- **Initial Loss**: 5.48 → **Final Loss**: 1.68
- **Training Duration**: ~12 minutes (495 steps)
- **Memory Usage**: Optimized for T4 GPU (16GB)
- **Performance**: Smooth loss reduction without overfitting

## Model Downloads

### GGUF Format (Optimized for Inference)
- **[Q8_0 Quantized](https://huggingface.co/IzzulGod/Sorachio-1B-Chat/resolve/main/sorachio-1b-chat-q8_0.gguf?download=true)** - Recommended for most use cases
- **[F16 Full Precision](https://huggingface.co/IzzulGod/Sorachio-1B-Chat/resolve/main/sorachio-1b-chat-f16.gguf?download=true)** - Maximum quality

### HuggingFace Hub
```bash
# Download via transformers
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="IzzulGod/Sorachio-1B-Chat", filename="pytorch_model.bin")
```

## Performance & Capabilities

✅ **Natural Conversations**: Trained on multi-turn dialogues for contextual responses  
✅ **Efficient Training**: Achieved good performance with minimal computational resources  
✅ **Fast Inference**: Optimized for quick response generation  
✅ **Easy Deployment**: Available in multiple formats (PyTorch, GGUF)  

## Requirements

```bash
# Core requirements
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
peft>=0.6.0
trl>=0.7.0

# For GGUF conversion
llama-cpp-python
```

## Usage Examples

Check out the model in action:

**Indonesian:**
```
User: Ceritakan tentang Indonesia
Sorachio: Indonesia adalah negara kepulauan terbesar di dunia dengan lebih dari 17.000 pulau...
```

**English:**
```
User: Tell me about artificial intelligence
Sorachio: Artificial intelligence is a fascinating field that encompasses machine learning...
```

## License

This project follows the original Gemma3 license terms. Please refer to the [Gemma3 license](https://huggingface.co/google/gemma-2-2b/blob/main/LICENSE) for usage guidelines.

## Citation

If you use this model or code in your research, please cite:

```bibtex
@misc{sorachio-1b-chat,
  title={Sorachio-1B-Chat: A Fine-tuned Conversational AI for Indonesian},
  author={IzzulGod},
  year={2024},
  url={https://huggingface.co/IzzulGod/Sorachio-1B-Chat}
}
```

---

**🤗 Try the model**: [IzzulGod/Sorachio-1B-Chat](https://huggingface.co/IzzulGod/Sorachio-1B-Chat)  
