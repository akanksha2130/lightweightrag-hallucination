"""
data_utils.py
=============
Dataset loading and preprocessing for LightweightRAG evaluation.

Supports:
  - SQuAD v2.0  (answerable subset, validation split)
  - BoolQ       (validation split)
"""

from typing import List, Tuple, Optional
from datasets import load_dataset
import random


# ══════════════════════════════════════════════════════════════════════════
# SQuAD v2.0
# ══════════════════════════════════════════════════════════════════════════

def load_squad(
    n: int,
    split: str = "validation",
    answerable_only: bool = True,
    seed: int = 42,
    offset: int = 0,
) -> Tuple[List[str], List[str], List[List[str]], List[str]]:
    """
    Load n examples from SQuAD v2.0.

    Parameters
    ----------
    n               : number of examples to load
    split           : "validation" (default — never use "train" for eval)
    answerable_only : if True, skip unanswerable questions
    seed            : random seed for reproducibility
    offset          : skip the first `offset` examples (for non-overlapping splits)

    Returns
    -------
    (questions, contexts, ground_truths, ids)
      questions     : list of question strings
      contexts      : list of passage strings (one per question)
      ground_truths : list of list of valid answer strings
      ids           : list of example IDs
    """
    dataset = load_dataset("rajpurkar/squad_v2", split=split)

    if answerable_only:
        dataset = dataset.filter(lambda x: len(x["answers"]["text"]) > 0)

    # Deterministic shuffle then slice with offset
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[offset: offset + n]

    questions, contexts, ground_truths, ids = [], [], [], []
    for i in indices:
        ex = dataset[i]
        questions.append(ex["question"])
        contexts.append(ex["context"])
        ground_truths.append(ex["answers"]["text"])   # list of valid answers
        ids.append(ex["id"])

    print(f"Loaded {len(questions)} SQuAD v2.0 examples from {split} split.")
    return questions, contexts, ground_truths, ids


# ══════════════════════════════════════════════════════════════════════════
# BoolQ
# ══════════════════════════════════════════════════════════════════════════

def load_boolq(
    n: int,
    split: str = "validation",
    seed: int = 42,
    offset: int = 0,
) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Load n examples from BoolQ.

    Returns
    -------
    (questions, passages, ground_truths)
      ground_truths : list of ["yes"] or ["no"]
    """
    dataset = load_dataset("google/boolq", split=split)

    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[offset: offset + n]

    questions, passages, ground_truths = [], [], []
    for i in indices:
        ex = dataset[i]
        questions.append(ex["question"])
        passages.append(ex["passage"])
        gt = "yes" if ex["answer"] else "no"
        ground_truths.append([gt])

    print(f"Loaded {len(questions)} BoolQ examples from {split} split.")
    return questions, passages, ground_truths


# ══════════════════════════════════════════════════════════════════════════
# Non-overlapping split helper
# ══════════════════════════════════════════════════════════════════════════

def get_nonoverlapping_splits(
    dataset_name: str,
    sizes: List[int],
    seed: int = 42,
) -> List[int]:
    """
    Returns a list of offsets so that splits of `sizes` are non-overlapping.

    Example
    -------
    # Get calibration (50), then evaluation (100) — no overlap
    offsets = get_nonoverlapping_splits("squad", [50, 100])
    cal_q, cal_c, cal_gt, _ = load_squad(50,  offset=offsets[0])
    ev_q,  ev_c,  ev_gt, _  = load_squad(100, offset=offsets[1])
    """
    offsets = [0]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)
    return offsets
