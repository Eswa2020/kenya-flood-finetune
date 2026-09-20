"""
plot_loss_curve.py
Generates a PNG plot of training vs validation loss from trainer_state.json,
for the capstone's Deliverable 2 (loss curve analysis).
Run locally, in the same folder as monitor_training_loss.py.
"""
import json
import matplotlib.pyplot as plt

ADAPTER_DIR = "flood-llama-adapter/checkpoint-27"

with open(f"{ADAPTER_DIR}/trainer_state.json") as f:
    state = json.load(f)

history = state["log_history"]
train_steps = [h for h in history if "loss" in h and "eval_loss" not in h]
eval_steps  = [h for h in history if "eval_loss" in h]

train_x = [h["step"] for h in train_steps]
train_y = [h["loss"] for h in train_steps]
eval_x  = [h["step"] for h in eval_steps]
eval_y  = [h["eval_loss"] for h in eval_steps]

plt.figure(figsize=(8, 5))
plt.plot(train_x, train_y, marker="o", label="Training loss", color="#d62728")
plt.plot(eval_x, eval_y, marker="o", label="Validation loss", color="#1f77b4")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Kenya Flood Early-Warning Assistant — Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("Saved plot to loss_curve.png")
plt.show()