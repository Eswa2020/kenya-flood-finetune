# data_prep.py
# Kenya Flood Early-Warning Assistant — Monday-style pipeline
import json
import os
import random

SYSTEM_PROMPT = """You are the Kenya Urban Flood Early-Warning and Mitigation Assistant. You help county officials and residents understand flood risk signals, know what mitigation and preparedness steps to take, and know exactly which official channel to check for real-time verified alerts (Kenya Meteorological Department bulletins, NDMA advisories, county disaster management committees). You do not predict whether or when a flood will occur at a specific place — you always direct the user to the official real-time source for that call. Your guidance is practical, calm, and never alarmist."""


def load_and_validate_data(file_path: str) -> list:
    with open(file_path, "r") as f:
        data = json.load(f)
    if len(data) < 100:
        raise ValueError("Dataset too small: please provide at least 100 records.")
    return data


def format_example(qa: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
    }


def _load_llama_tokenizer():
    try:
        from transformers import AutoTokenizer
        from dotenv import load_dotenv
        from huggingface_hub import login
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token)
        return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    except Exception as exc:
        print(f"[WARN] Could not load the LLaMA tokeniser ({type(exc).__name__}).")
        print("[WARN] Falling back to an approximate token count (chars / 4).")
        return None


def count_tokens(msgs: list, tokenizer) -> int:
    if tokenizer is not None:
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False)
        return len(tokenizer.encode(formatted))
    chars = sum(len(m.get("content", "")) for m in msgs)
    return chars // 4 + 30


def validate_dataset(examples: list) -> dict:
    tokenizer = _load_llama_tokenizer()
    errors, warnings = [], []
    seen_questions = set()
    token_counts = []

    for i, example in enumerate(examples):
        msgs = example.get("messages", [])
        if len(msgs) != 3:
            errors.append(f"Example {i}: expected 3 messages, got {len(msgs)}")
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"Example {i}: wrong role order {roles}")
            continue
        for msg in msgs:
            if not msg.get("content", "").strip():
                errors.append(f"Example {i}: empty content in role '{msg['role']}'")
        user_content = msgs[1]["content"]
        if user_content in seen_questions:
            warnings.append(f"Example {i}: duplicate question '{user_content[:60]}...'")
        seen_questions.add(user_content)
        tokens = count_tokens(msgs, tokenizer)
        token_counts.append(tokens)
        if tokens < 64:
            warnings.append(f"Example {i}: very short ({tokens} tokens)")
        if tokens > 2048:
            errors.append(f"Example {i}: too long ({tokens} tokens), exceeds training context")

    return {
        "total_examples": len(examples),
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "valid_examples": len(examples) - len(errors),
    }


def split_and_save(examples: list, output_dir: str = "data") -> None:
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    shuffled = examples[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    splits = [
        ("train", shuffled[:n_train]),
        ("val",   shuffled[n_train:n_train + n_val]),
        ("test",  shuffled[n_train + n_val:]),
    ]
    for split_name, split_data in splits:
        path = f"{output_dir}/{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(split_data)} examples to {path}")
    print(f"\nSplit summary: {len(splits[0][1])} train | {len(splits[1][1])} val | {len(splits[2][1])} test")


raw_data = load_and_validate_data("operational_data.json")
formatted_data = [format_example(record) for record in raw_data]
print(f"Formatted {len(formatted_data)} examples")

report = validate_dataset(formatted_data)
print("\n=== DATASET VALIDATION REPORT ===")
print(f"Total examples:  {report['total_examples']}")
print(f"Valid examples:  {report['valid_examples']}")
print(f"Errors:          {report['error_count']}")
print(f"Warnings:        {report['warning_count']}")
print(f"Avg token count: {report['avg_tokens']}")
print(f"Token range:     {report['min_tokens']} to {report['max_tokens']}")

for e in report["errors"]:
    print(f"  ERROR: {e}")
for w in report["warnings"]:
    print(f"  WARN:  {w}")

if report["error_count"] == 0:
    split_and_save(formatted_data)
    print("\nDataset preparation complete. Ready to upload to Nebius.")
else:
    print("\nFix the errors above before splitting. Do not train on a broken dataset.")