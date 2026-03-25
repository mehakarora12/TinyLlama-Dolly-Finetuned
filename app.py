import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

# 1. Force the model to download to the Space's local storage
# and use the CPU-only settings
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_MODEL = "Mehak-123-arora/tinyllama-dolly-finetuned"

print("--- Starting Application ---")

# 2. Loading Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 3. Loading Model (Optimized for Hugging Face Free CPU)
print("Loading base model (this may take 2-5 minutes)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",       # Using "cpu" string is safer for some transformers versions
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL)
model.eval()
print("✓ Everything loaded successfully!")

def chat_function(message, history, max_tokens, temperature):
    # Constructing the prompt with TinyLlama format
    prompt = "<|system|>\nYou are a helpful assistant.</s>\n"
    for user_msg, bot_msg in history:
        prompt += f"<|user|>\n{user_msg}</s>\n<|assistant|>\n{bot_msg}</s>\n"
    prompt += f"<|user|>\n{message}</s>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# Create the interface WITHOUT the theme here
with gr.Blocks() as demo:
    gr.Markdown("# 🦙 TinyLlama Dolly-Chat (CPU)")
    gr.ChatInterface(
        fn=chat_function,
        additional_inputs=[
            gr.Slider(50, 512, value=128, label="Max Tokens"),
            gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
        ],
    )

if __name__ == "__main__":
    # demo.queue() is already called here
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_error=True
    )

