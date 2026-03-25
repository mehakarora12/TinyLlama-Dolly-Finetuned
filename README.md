# 🚀 TinyLlama-1.1B Dolly Instruction Finetune

A conversational chatbot built by fine-tuning **TinyLlama-1.1B** on the **Databricks Dolly 15k** dataset using **QLoRA** (Quantized Low-Rank Adaptation).

## 🔗 Model Card
The model weights and configuration files are hosted on Hugging Face:
👉 **[View Model on Hugging Face](https://huggingface.co/Mehak-123-arora/tinyllama-dolly-finetuned)**

## 💡 Project Highlights
- **Base Model:** [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- **Dataset:** `databricks/databricks-dolly-15k`
- **Method:** LoRA (Low-Rank Adaptation) for parameter-efficient finetuning.
- **Hardware:** Trained using an NVIDIA T4 GPU.
- Training Technique: QLoRA (4-bit Quantized Low-Rank Adaptation).
- Training Platform: Google Colab (T4 GPU) 
- Quantization: 4-bit NormalFloat (nf4) with Double Quantization to minimize memory usage.

## Training Details

| Parameter | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Dataset size | 500 samples |
| Epochs | 2 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Quantization | 4-bit NF4 |
| Batch size | 1 (gradient accumulation: 8) |
| Learning rate | 2e-4 |
| Training time | ~10 minutes on T4 GPU |
| Final loss | ~1.46 |

## Key Learnings

- QLoRA makes it possible to fine-tune LLMs on free hardware
- Only 0.19% of parameters need to be trained with LoRA
- Instruction format matters — using the correct chat template significantly improves results
- Even 500 samples show measurable improvement in loss





