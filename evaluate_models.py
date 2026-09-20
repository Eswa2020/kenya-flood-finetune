import json
import gc
import torch
import pandas as pd
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from evaluator import evaluate_response
from llm_judge import llm_judge

BASE_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MERGED_PATH  = "./flood-llama-merged"
TEST_FILE    = "test.jsonl"
GROUNDEDNESS_FLOOR = 3.0


def build_pipe(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu")
    return pipeline("text-generation", model=model, tokenizer=tok), tok


def query(pipe, tok, system_prompt: str, question: str) -> str:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    out = pipe(prompt, max_new_tokens=300, do_sample=False,
               repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    return out[0]["generated_text"][len(prompt):].strip()


if __name__ == "__main__":
    with open(TEST_FILE) as f:
        test_examples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_examples)} held-out test examples")

    print("Loading base model...")
    base_pipe, base_tok = build_pipe(BASE_MODEL)
    base_responses = []
    for ex in test_examples:
        msgs = ex["messages"]
        resp = query(base_pipe, base_tok, msgs[0]["content"], msgs[1]["content"])
        base_responses.append(resp)
        print(f"  base done: {msgs[1]['content'][:50]}...")
    del base_pipe, base_tok
    gc.collect()
    print("Base model unloaded.\n")

    print("Loading fine-tuned model...")
    ft_pipe, ft_tok = build_pipe(MERGED_PATH)
    ft_responses = []
    for ex in test_examples:
        msgs = ex["messages"]
        resp = query(ft_pipe, ft_tok, msgs[0]["content"], msgs[1]["content"])
        ft_responses.append(resp)
        print(f"  ft done: {msgs[1]['content'][:50]}...")
    del ft_pipe, ft_tok
    gc.collect()
    print("Fine-tuned model unloaded.\n")

    results = []
    safety_alerts = 0
    for ex, base_resp, ft_resp in zip(test_examples, base_responses, ft_responses):
        msgs = ex["messages"]
        question, reference = msgs[1]["content"], msgs[2]["content"]

        base_auto = evaluate_response(reference, base_resp)
        ft_auto   = evaluate_response(reference, ft_resp)
        base_judge = llm_judge(question, reference, base_resp)
        ft_judge   = llm_judge(question, reference, ft_resp)

        ft_ground = ft_judge.get("groundedness", 0)
        if ft_ground < GROUNDEDNESS_FLOOR:
            safety_alerts += 1
            print(f"  [ALERT] Low groundedness ({ft_ground}/5) on: {question[:60]}...")

        results.append({
            "question":     question[:70] + "...",
            "base_response": base_resp[:150],
            "ft_response":   ft_resp[:150],
            "base_rouge_l": base_auto["rouge_l"],
            "ft_rouge_l":   ft_auto["rouge_l"],
            "rouge_delta":  round(ft_auto["rouge_l"] - base_auto["rouge_l"], 4),
            "base_judge":   base_judge.get("overall", 0),
            "ft_judge":     ft_judge.get("overall", 0),
            "judge_delta":  round(ft_judge.get("overall", 0) - base_judge.get("overall", 0), 2),
            "base_ground":  base_judge.get("groundedness", 0),
            "ft_ground":    ft_ground,
        })

    df = pd.DataFrame(results)
    df.to_csv("comparison_results.csv", index=False)
    print(f"\nSaved per-question results to comparison_results.csv")

    summary = {
        "Metric": ["ROUGE-L (avg)", "LLM Judge /5 (avg)", "Groundedness /5 (avg)"],
        "Base LLaMA": [round(df["base_rouge_l"].mean(), 3),
                       round(df["base_judge"].mean(), 2),
                       round(df["base_ground"].mean(), 2)],
        "Fine-Tuned": [round(df["ft_rouge_l"].mean(), 3),
                       round(df["ft_judge"].mean(), 2),
                       round(df["ft_ground"].mean(), 2)],
    }
    summary["Delta"] = [round(summary["Fine-Tuned"][i] - summary["Base LLaMA"][i], 3)
                        for i in range(3)]
    print(tabulate(pd.DataFrame(summary), headers="keys", tablefmt="grid", showindex=False))

    if safety_alerts > 0:
        print(f"\nCRITICAL COMPLIANCE WARNING: {safety_alerts} response(s) fell "
              f"below the groundedness floor of {GROUNDEDNESS_FLOOR}.")
    else:
        print("\nCompliance check passed: no responses below the groundedness floor.")