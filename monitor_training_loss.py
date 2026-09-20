# monitor_training_loss.py
import json
import pandas as pd

ADAPTER_DIR = "flood-llama-adapter/checkpoint-27"
CLINICAL_THRESHOLD = 0.3  # Max acceptable overfitting gap

with open(f"{ADAPTER_DIR}/trainer_state.json") as f:
    state = json.load(f)

history = state["log_history"]

train_steps = [h for h in history if "loss" in h and "eval_loss" not in h]
eval_steps  = [h for h in history if "eval_loss" in h]

if not train_steps or not eval_steps:
    raise SystemExit(
        "No training or evaluation rows found in trainer_state.json. "
        "Check that eval_strategy was set to 'steps' in TrainingArguments."
    )

train_df = pd.DataFrame(train_steps)[["step", "loss", "epoch"]].rename(
    columns={"loss": "train_loss"})
eval_df = pd.DataFrame(eval_steps)[["step", "eval_loss", "epoch"]]

print("=== TRAINING LOSS PROGRESSION ===")
print(train_df.to_string(index=False))
print("\n=== VALIDATION LOSS PROGRESSION ===")
print(eval_df.to_string(index=False))

final_train = train_df["train_loss"].iloc[-1]
final_eval  = eval_df["eval_loss"].iloc[-1]
best_eval   = eval_df["eval_loss"].min()
best_step   = int(eval_df.loc[eval_df["eval_loss"].idxmin(), "step"])
gap         = final_eval - final_train

print("\n=== TRAINING SUMMARY ===")
print(f"Final training loss:   {final_train:.4f}")
print(f"Final validation loss: {final_eval:.4f}")
print(f"Best validation loss:  {best_eval:.4f}  (at step {best_step})")
print(f"Overfitting gap:       {gap:.4f}")

if final_eval > eval_df["eval_loss"].iloc[0]:
    print("\nDIAGNOSIS: Validation loss increased overall. The model has overfit.")
    print(f"ACTION: Load the checkpoint at step {best_step} and stop training there.")
elif gap > CLINICAL_THRESHOLD:
    print(f"\n[CAUTION] Overfitting gap {gap:.4f} exceeds the safety threshold of {CLINICAL_THRESHOLD}.")
    print("ACTION: Review before deploying; consider higher lora_dropout or fewer epochs.")
else:
    print("\nDIAGNOSIS: Training looks healthy. Loss curves fell together "
          "and stayed within the safety limit.")
    print("ACTION: Proceed to merge and evaluation.")