import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

HF_TOKEN   = os.getenv("HF_TOKEN")
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "flood-llama-adapter"
DATA_DIR   = "."

LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

LEARNING_RATE = 2e-4
BATCH_SIZE    = 8
GRAD_ACCUM    = 2
NUM_EPOCHS    = 3
MAX_SEQ_LEN   = 512

if not HF_TOKEN:
    raise SystemExit("Set your token first: export HF_TOKEN=hf_your_token_here")
login(token=HF_TOKEN)
print("Configuration loaded. HF login successful.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto",
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}% of {total:,})")

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/val.jsonl",
    },
)

def apply_chat_template(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(apply_chat_template)
print(f"Dataset prepared. Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")
print("Sample training text (first 300 chars):")
print(dataset["train"][0]["text"][:300])

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,
    weight_decay=0.001,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

print("Starting training...")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nTraining complete. Adapter saved to: {OUTPUT_DIR}")
