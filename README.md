# Kenya Urban Flood Early-Warning Assistant

## Overview
A LLaMA 3.1 8B model fine-tuned with QLoRA to help county officials and residents interpret Kenya Meteorological Department (KMD) flood risk advisories, understand mitigation and preparedness steps, and know exactly which official channel (KMD, NDMA, county disaster management committees) to check for real-time verified alerts. The assistant never predicts whether or when a flood will occur at a specific location — it always defers that call to official sources. Built ahead of the forecast October–December 2026 short rains, expected to bring above-average rainfall and elevated flood risk to 46 of Kenya's 47 counties under a very strong El Niño.

## Project Structure
- `data/`: train.jsonl (144), val.jsonl (18), test.jsonl (18)
- `operational_data.json`: 180 curated Q&A pairs, sourced from KMD publications and verified 2026 flood reporting
- `data_prep.py`: formats raw Q&A into LLaMA chat format, validates, splits 80/10/10
- `fine_tune.py`: QLoRA training script (run on a Nebius GPU Devlab)
- `merge_model.py`: folds the trained adapter into the base model
- `local_inference.py`: inference pipeline with a safety filter for non-operational content
- `evaluator.py`: BLEU-4, ROUGE, Token F1 lexical metrics
- `llm_judge.py`: LLM-as-a-Judge scoring (correctness, groundedness, relevance, helpfulness)
- `verify_merge.py`: 3-question sanity gate run before the full evaluation
- `evaluate_models.py`: base vs fine-tuned evaluation on the 18 sealed test questions
- `monitor_training_loss.py` / `plot_loss_curve.py`: loss curve analysis and plot
- `comparison_results.csv`: per-question evaluation results
- `loss_curve.png`: training vs validation loss plot
- `flood-llama-adapter/`: trained LoRA adapter weights and tokenizer files
- `curation_note.md`: dataset sourcing, quality criteria, and coverage gaps
- `memo.md`: stakeholder recommendation

## How to Reproduce
1. Environment setup: `pip install transformers peft bitsandbytes trl accelerate datasets huggingface_hub python-dotenv pandas nltk rouge_score tabulate langchain-openai`
2. Data: place `operational_data.json` in the project root, then run `python data_prep.py`
3. Fine-tuning: upload `data/train.jsonl` and `data/val.jsonl` to a Nebius GPU Devlab, set `HF_TOKEN`, run `python fine_tune.py`
4. Merging: download the adapter, place it at `./flood-llama-adapter`, run `python merge_model.py`
5. Verification: run `python verify_merge.py` before the full evaluation
6. Evaluation: set `OPENAI_API_KEY`, run `python evaluate_models.py` to generate `comparison_results.csv`

## Results Summary
| Metric | Base LLaMA | Fine-Tuned | Delta |
|---|---|---|---|
| ROUGE-L (avg) | 0.152 | 0.261 | +0.109 |
| LLM Judge /5 (avg) | 4.26 | 4.47 | +0.21 |
| Groundedness /5 (avg) | 3.78 | 4.17 | +0.39 |

All 18 test responses cleared the groundedness safety floor (3.0/5). The largest single improvement was on a Kenya-specific factual question (KMD's rainfall warning threshold), where judged quality rose from 1.8 to 3.8 — the base model had no way to know this fact; the fine-tuned model, trained on it, answered it correctly.

## Disclaimer
This model provides general flood preparedness and mitigation guidance only. It does not predict whether or when flooding will occur at any specific location. Always check official Kenya Meteorological Department bulletins, NDMA advisories, and your county disaster management committee for real-time, verified alerts.
