from local_inference import ask
from evaluator import evaluate_response

sanity_questions = [
    {
        "question": "What does a high flood risk advisory from the Kenya Meteorological Department mean, and what should I do?",
        "reference": ("A high flood risk alert from KMD means sustained heavy rainfall and rising "
                      "river levels make flooding likely in the named area within the advisory "
                      "period. Move valuables and vehicles to higher ground, avoid crossing "
                      "flooded roads or bridges, and follow instructions from your county "
                      "disaster management office."),
        "expected_keywords": ["high", "flooding", "higher ground", "county"],
    },
    {
        "question": "Is it safe to drive through a flooded road if the water looks shallow?",
        "reference": ("No. Floodwater on roads can conceal open drains, missing manhole covers, "
                      "or fast-moving currents that are not visible from a vehicle. Never attempt "
                      "to drive or walk through a flooded road."),
        "expected_keywords": ["no", "drains", "current"],
    },
    {
        "question": "What health risks come with flooding, beyond the immediate danger of drowning?",
        "reference": ("Flooding creates stagnant water that raises the risk of vector-borne "
                      "diseases such as malaria and waterborne diseases such as cholera and "
                      "typhoid. Boil or treat drinking water after a flood event."),
        "expected_keywords": ["malaria", "cholera", "water"],
    },
]

if __name__ == "__main__":
    print("=== FLOOD ASSISTANT MERGE VERIFICATION ===")
    all_passed = True
    formatting_warnings = 0

    for sq in sanity_questions:
        response = ask(sq["question"], max_new_tokens=200)
        scores = evaluate_response(sq["reference"], response)
        missing = [kw for kw in sq["expected_keywords"]
                   if kw.lower() not in response.lower()]

        is_formatted = "\n" in response or ". " in response
        if not is_formatted:
            formatting_warnings += 1
            print(f"  [WARN] Formatting requirement not met for: {sq['question'][:50]}")

        status = "PASS" if not missing and is_formatted else "WARN"
        print(f"\n  Q: {sq['question']}")
        print(f"  Status: {status}")
        print(f"  ROUGE-L: {scores['rouge_l']} | Token F1: {scores['token_f1']}")
        if missing:
            print(f"  Missing expected keywords: {missing}")
            all_passed = False

    print(f"\nFormatting warnings: {formatting_warnings}")
    print("Verification:",
          "ALL PASSED, proceed to full evaluation" if all_passed and formatting_warnings == 0
          else "REVIEW REQUIRED before full evaluation")