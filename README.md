# LightweightRAG: A Training-Free Hybrid RAG Framework for Hallucination Reduction on CPU-Only Infrastructure

[![CPU Only](https://img.shields.io/badge/Hardware-CPU%20Only-green)]()
[![No Training](https://img.shields.io/badge/Training-None%20Required-blue)]()
[![Python 3.10](https://img.shields.io/badge/Python-3.10-yellow)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akanksha2130/lightweightrag-hallucination/blob/main/notebooks/LightweightRAG_Experiments.ipynb)

> **Paper:** *LightweightRAG: A Statistically Validated, Training-Free Hybrid Retrieval-Augmented Generation Framework for Hallucination Reduction on CPU-Only Infrastructure*
> **Authors:** Akanksha Singh, Dr. Chandan Kumar
> **Venue:** ICSIT 2026 (under revision)

---

## What is LightweightRAG?

LightweightRAG is a **CPU-only, training-free hybrid RAG pipeline** that reduces LLM hallucination using only off-the-shelf open-source components — no GPU, no fine-tuning, no proprietary infrastructure required.

It combines:
- **BM25** sparse retrieval (keyword matching)
- **all-MiniLM-L6-v2** dense retrieval (semantic similarity, 22.7M params, cosine similarity via numpy — no external vector index required)
- **Reciprocal Rank Fusion** (parameter-free combination)
- **Flan-T5-base** answer generation (250M params, instruction-tuned)
- **Evidence verification** (cosine-similarity threshold for conditional abstention)

### Key Results

| Dataset | Baseline Hall. | LightweightRAG Hall. | Reduction | Significance |
|---|---|---|---|---|
| SQuAD v2.0 (real generation, n=500, α=0.05) | 98.0% | 41.5% | 57.7% relative | p < 0.001 |
| TriviaQA (n=200, α=0.05) | 92.0% | 67.7% | 26.4% relative | p < 0.001 |
| SQuAD v2.0 (oracle extraction, n=300) | 98.3% | 39.7% | — | McNemar vs. BM25-only: p=0.47 |

Manual validation of the EM-based hallucination metric against human judgment on a 150-sample subset: **84.7% agreement (127/150)**.

**Mean latency: ~2.4–3.3 s/query on Colab CPU (2 vCPUs, no GPU).**

---

## Repository Structure

```
lightweightrag-hallucination/
│
├── src/                          ← Core library (import this)
│   ├── __init__.py
│   ├── pipeline.py               ← LightweightRAG class (main pipeline)
│   ├── evaluation.py             ← EM, F1, McNemar test, Wilson CI, corrected HR/RR
│   └── data_utils.py             ← SQuAD v2.0, TriviaQA, and BoolQ data loaders
│
├── notebooks/
│   └── LightweightRAG_Experiments.ipynb   ← Reproduces all paper tables
│
├── results/                      ← Saved raw outputs backing each paper table
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option 1: Open in Colab (recommended)

Click the **Open in Colab** badge above.

> ⚠️ **Important:** In Colab, go to **Runtime → Change runtime type → CPU**. This paper specifically evaluates CPU-only performance — do not select GPU.

### Option 2: Run locally

```bash
git clone https://github.com/akanksha2130/lightweightrag-hallucination.git
cd lightweightrag-hallucination
pip install -r requirements.txt
```

```python
from src.pipeline import LightweightRAG

rag = LightweightRAG(tau_extractive=0.05, verbose=True)

passages = [
    "The Colorado River flows through the Grand Canyon in Arizona. "
    "It is approximately 1,450 miles long.",
    "The Eiffel Tower was completed in 1889 in Paris, France.",
]
rag.build_index(passages)

result = rag.answer("What river flows through the Grand Canyon?")
print(result["answer"])     # → "Colorado River"
print(result["abstained"])  # → False
```

---

## Reproducing Paper Results

Open `notebooks/LightweightRAG_Experiments.ipynb` in Colab and run cells in order.

| Cells | Experiment | Paper Table |
|---|---|---|
| 3–9 | SQuAD v2.0 real generation (Configs A/B/C, n=500) | Table II |
| 10–14 | Threshold calibration | Section IV-G |
| 19–28 | Final Table II compilation + Wilson CI + McNemar | Table II |
| 29–30 | SQuAD v2.0 oracle extraction (n=300) | Table I |
| 31–33 | TriviaQA cross-domain evaluation (n=200) | Table III |
| 34, 40 | Ablation study (BM25-only / dense-only / hybrid RRF, n=100) | Table IV |
| 26 | Threshold sensitivity sweep | Table V |
| 36–38 | Qualitative case studies | Section VI-F |
| 39 | Manual HR-validation set generation (n=150) | Section VII-B |
| 43–47 | Latency benchmarking | Latency columns |

**Note:** The notebook saves intermediate results to Google Drive (`/content/drive/MyDrive/lightweightrag_results/`) so long-running cells can be resumed after a Colab disconnect. If you don't want to use Drive, redirect `SAVE_PATH`/`SAVE_*` variables to a local path instead — the pipeline itself has no Drive dependency.

---

## API Reference

### `LightweightRAG`

```python
LightweightRAG(
    embed_model    = "sentence-transformers/all-MiniLM-L6-v2",
    gen_model      = "google/flan-t5-base",
    tau_extractive = 0.05,   # verification/abstention threshold (paper default, Section IV-G)
    top_k          = 5,      # chunks retrieved per retriever
    verbose        = False,
)
```

| Method | Description |
|---|---|
| `build_index(passages)` | Chunk passages, build BM25 index and dense embedding matrix |
| `answer(question, mode)` | Full pipeline → `{answer, abstained, chunks, latency}` |

**`mode`:** `"extractive"` (SQuAD/TriviaQA-style) or `"boolean"` (BoolQ-style)

### Threshold guide (SQuAD v2.0, n=500 — Table V)

| α | Hallucination | Refusal | EM | Guidance |
|---|---|---|---|---|
| 0.05 | 41.5% | 2.6% | 57.0% | **Default** |
| 0.25 | 47.0% | 32.4% | 35.8% | Not recommended |
| 0.50 | 56.7% | 79.2% | 9.0% | Not recommended |
| 0.70 | 64.3% | 97.2% | 1.0% | Not recommended |
| 0.80 | 100.0% | 99.8% | 0.0% | No utility |

Note the non-monotonic relationship: because the cosine-similarity confidence score correlates only weakly (and at times negatively, Pearson r=−0.17) with answer correctness, raising α *increases* rather than decreases hallucination on this metric. See Section VII-A/B of the paper for discussion.

---

## Hardware Requirements

| Component | Minimum | Used in paper |
|---|---|---|
| CPU | Any x86-64 | Google Colab free tier (2 vCPUs) |
| RAM | 6 GB | 12 GB available |
| GPU | **Not required** | **Not used at any stage** |
| Storage | ~2 GB (models) | — |

---

## Citation

If you use this code or paper, please cite:

```bibtex
@inproceedings{singh2026lightweightrag,
  title     = {LightweightRAG: A Statistically Validated, Training-Free Hybrid
               Retrieval-Augmented Generation Framework for Hallucination
               Reduction on CPU-Only Infrastructure},
  author    = {Singh, Akanksha and Kumar, Chandan},
  booktitle = {ICSIT 2026},
  year      = {2026},
  note      = {Code: https://github.com/akanksha2130/lightweightrag-hallucination}
}
```

---

## License

MIT License — free to use, modify, and distribute with attribution.
