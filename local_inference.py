import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MERGED_PATH = "./flood-llama-merged"

SYSTEM_PROMPT = """You are the Kenya Urban Flood Early-Warning and Mitigation Assistant. You help county officials and residents understand flood risk signals, know what mitigation and preparedness steps to take, and know exactly which official channel to check for real-time verified alerts (Kenya Meteorological Department bulletins, NDMA advisories, county disaster management committees). You do not predict whether or when a flood will occur at a specific place — you always direct the user to the official real-time source for that call. Your guidance is practical, calm, and never alarmist."""

# Words signalling the model has stepped outside its scope: predicting a
# specific flood event, or making guarantees it cannot back
PROHIBITED = ["guarantee", "will definitely flood", "financial advice", "legal advice", "certain to happen"]

print("Loading merged Flood Early-Warning model...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def ask(question: str, max_new_tokens: int = 200) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    output = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = output[0]["generated_text"][len(prompt):].strip()

    if any(word in response.lower() for word in PROHIBITED):
        return ("Error: Response contains non-operational content. "
                "Please refine the operational prompt.")
    return response


if __name__ == "__main__":
    test_questions = [
        "What does a high flood risk advisory mean, and what should I do?",
        "Is it safe to drive through a flooded road if the water looks shallow?",
        "What health risks come with flooding, beyond the immediate danger of drowning?",
        "Who is responsible for coordinating evacuation during a severe flood alert?",
        "As a farmer in Uasin Gishu, what should I do to protect my land before heavy rains?",
    ]
    for question in test_questions:
        print(f"\nQ: {question}")
        print(f"A: {ask(question)}")

    r1 = ask("What does a high flood risk advisory mean, and what should I do?")
    r2 = ask("What does a high flood risk advisory mean, and what should I do?")
    assert r1 == r2, "Consistency test failed: outputs differ between runs!"
    print("\nStability test passed: identical output on repeat query.")