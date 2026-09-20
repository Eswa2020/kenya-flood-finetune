# merge_model.py
# Runs on the CPU Devlab. Folds the trained LoRA adapter into the base
# model, producing one standard deployable model with no PEFT overhead.
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

BASE_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./flood-llama-adapter"
MERGED_PATH  = "./flood-llama-merged"

print("Loading base model for merging (downloads ~16GB on first run)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapter weights into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_PATH)

print("Merge complete.")
print(f"Merged model saved to: {MERGED_PATH}")