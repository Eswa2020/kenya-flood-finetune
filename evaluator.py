# evaluator.py
import nltk, string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import Counter

for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

def compute_bleu4(reference: str, hypothesis: str) -> float:
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    smoother = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoother)

def compute_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}

def _normalise(text: str) -> list:
    text = ''.join(ch for ch in text.lower() if ch not in string.punctuation)
    return text.split()

def compute_token_f1(reference: str, hypothesis: str) -> float:
    ref_tokens, hyp_tokens = _normalise(reference), _normalise(hypothesis)
    shared = sum((Counter(hyp_tokens) & Counter(ref_tokens)).values())
    if shared == 0: return 0.0
    precision = shared / len(hyp_tokens)
    recall    = shared / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)

def evaluate_response(reference: str, hypothesis: str) -> dict:
    rouge = compute_rouge(reference, hypothesis)
    return {
        "bleu_4":   round(compute_bleu4(reference, hypothesis), 4),
        "rouge_1":  rouge['rouge1'],
        "rouge_2":  rouge['rouge2'],
        "rouge_l":  rouge['rougeL'],
        "token_f1": compute_token_f1(reference, hypothesis),
    }

def evaluate_response_simple(reference: str, candidate: str) -> dict:
    """Returns {'rouge_l': float} for one reference/candidate pair."""
    rouge = compute_rouge(reference, candidate)
    return {"rouge_l": rouge["rougeL"]}