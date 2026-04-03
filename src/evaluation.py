"""
evaluation.py
=============
Evaluation utilities for LightweightRAG.

Provides:
  - normalize_answer()     : standard SQuAD/BoolQ answer normalisation
  - exact_match()          : per-sample EM
  - token_f1()             : per-sample F1
  - compute_metrics()      : batch EM, F1, hallucination rate, refusal rate
  - mcnemar_test()         : paired significance test
  - wilson_ci()            : 95% confidence interval for a proportion
  - run_evaluation()       : end-to-end evaluation loop
"""

import re
import string
import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.stats import chi2


# ══════════════════════════════════════════════════════════════════════════
# Answer normalisation (standard SQuAD preprocessing)
# ══════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, ground_truths: List[str]) -> int:
    """Return 1 if prediction matches any ground truth after normalization."""
    pred_norm = normalize_answer(prediction)
    return int(any(pred_norm == normalize_answer(gt) for gt in ground_truths))


def token_f1(prediction: str, ground_truths: List[str]) -> float:
    """Token-level F1 between prediction and best-matching ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common    = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall    = len(common) / len(gt_tokens)   if gt_tokens   else 0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        best_f1   = max(best_f1, f1)
    return best_f1


# ══════════════════════════════════════════════════════════════════════════
# Batch metrics
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    predictions:   List[str],
    ground_truths: List[List[str]],
    abstained:     List[bool],
) -> Dict:
    """
    Compute EM, F1, hallucination rate, and refusal rate for a batch.

    Parameters
    ----------
    predictions   : list of str  — model outputs (including abstentions)
    ground_truths : list of list — each entry is a list of valid answers
    abstained     : list of bool — True if the system abstained on that query

    Returns
    -------
    dict with: em, f1, hallucination_rate, refusal_rate,
               em_per_sample (list), f1_per_sample (list),
               hallucinated_per_sample (list of 0/1)
    """
    assert len(predictions) == len(ground_truths) == len(abstained)

    em_scores, f1_scores, hall_flags = [], [], []

    for pred, gts, abst in zip(predictions, ground_truths, abstained):
        if abst:
            # Abstained: counts as mismatch (conservative), EM=0, F1=0
            em_scores.append(0)
            f1_scores.append(0.0)
            hall_flags.append(1)          # abstentions are refusals, not hallucinations
        else:
            em  = exact_match(pred, gts)
            f1  = token_f1(pred, gts)
            em_scores.append(em)
            f1_scores.append(f1)
            hall_flags.append(0 if em else 1)  # mismatch = hallucination proxy

    n = len(predictions)
    return {
        "em":                   round(100 * np.mean(em_scores), 1),
        "f1":                   round(100 * np.mean(f1_scores), 1),
        "hallucination_rate":   round(100 * np.mean(hall_flags), 1),
        "refusal_rate":         round(100 * np.mean(abstained),  1),
        "em_per_sample":        em_scores,
        "f1_per_sample":        f1_scores,
        "hallucinated_per_sample": hall_flags,
        "n":                    n,
    }


# ══════════════════════════════════════════════════════════════════════════
# Statistical tests
# ══════════════════════════════════════════════════════════════════════════

def mcnemar_test(
    correct_a: List[int],
    correct_b: List[int],
) -> Tuple[float, float]:
    """
    Continuity-corrected McNemar test for paired binary outcomes.

    Parameters
    ----------
    correct_a : list of 0/1  — 1 = correct for system A
    correct_b : list of 0/1  — 1 = correct for system B

    Returns
    -------
    (chi2_stat, p_value)
    """
    assert len(correct_a) == len(correct_b)
    b, c = 0, 0   # b = A correct, B wrong;  c = A wrong, B correct
    for a_, b_ in zip(correct_a, correct_b):
        if a_ == 1 and b_ == 0:
            b += 1
        elif a_ == 0 and b_ == 1:
            c += 1

    if b + c == 0:
        return 0.0, 1.0   # no disagreements

    # Continuity-corrected statistic
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)
    return round(chi2_stat, 2), round(p_value, 4)


def wilson_ci(
    count: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Wilson score 95% confidence interval for a proportion.

    Returns (lower%, upper%) as percentages.
    """
    from scipy.stats import norm
    z   = norm.ppf(1 - (1 - confidence) / 2)
    p   = count / n if n > 0 else 0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = max(0.0, (centre - margin) * 100)
    hi = min(100.0, (centre + margin) * 100)
    return round(lo, 1), round(hi, 1)


# ══════════════════════════════════════════════════════════════════════════
# End-to-end evaluation loop
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    pipeline,
    questions:     List[str],
    ground_truths: List[List[str]],
    mode:          str = "extractive",
    verbose:       bool = True,
) -> Dict:
    """
    Run the full LightweightRAG pipeline on a list of questions and
    return aggregated metrics.

    Parameters
    ----------
    pipeline      : LightweightRAG instance (already indexed)
    questions     : list of question strings
    ground_truths : list of list of valid answer strings
    mode          : "extractive" or "boolean"
    verbose       : print progress

    Returns
    -------
    dict with em, f1, hallucination_rate, refusal_rate, confidence_intervals,
              per_sample results, and latency stats.
    """
    predictions, abstained_flags, latencies = [], [], []

    for i, (q, gts) in enumerate(zip(questions, ground_truths)):
        result = pipeline.answer(q, mode=mode)
        predictions.append(result["answer"])
        abstained_flags.append(result["abstained"])
        latencies.append(result["latency"]["total"])

        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(questions)}")

    metrics = compute_metrics(predictions, ground_truths, abstained_flags)

    # Confidence intervals for hallucination rate
    n_hall = int(round(metrics["hallucination_rate"] / 100 * metrics["n"]))
    ci_lo, ci_hi = wilson_ci(n_hall, metrics["n"])
    metrics["hall_ci"] = (ci_lo, ci_hi)

    # Latency stats
    metrics["latency_mean"] = round(np.mean(latencies), 2)
    metrics["latency_std"]  = round(np.std(latencies),  2)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  EM:                 {metrics['em']}%")
        print(f"  F1:                 {metrics['f1']}%")
        print(f"  Hallucination Rate: {metrics['hallucination_rate']}%  "
              f"(95% CI: {ci_lo}%–{ci_hi}%)")
        print(f"  Refusal Rate:       {metrics['refusal_rate']}%")
        print(f"  Mean Latency:       {metrics['latency_mean']}s ± {metrics['latency_std']}s")
        print(f"{'='*50}\n")

    return metrics
