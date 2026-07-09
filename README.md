# LightweightRAG: A Statistically Validated, Training-Free Hybrid RAG Framework for Hallucination Reduction on CPU-Only Infrastructure

[![CPU Only](https://img.shields.io/badge/Hardware-CPU%20Only-green)](https://github.com/akanksha2130/lightweightrag-hallucination/blob/main)
[![No Training](https://img.shields.io/badge/Training-None%20Required-blue)](https://github.com/akanksha2130/lightweightrag-hallucination/blob/main)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-yellow)](https://github.com/akanksha2130/lightweightrag-hallucination/blob/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akanksha2130/lightweightrag-hallucination/blob/main/notebooks/LightweightRAG_Experiments.ipynb)

> **Paper:** *LightweightRAG: A Statistically Validated, Training-Free Hybrid Retrieval-Augmented Generation Framework for Hallucination Reduction on CPU-Only Infrastructure*
> **Authors:** Akanksha Singh, Dr. Chandan Kumar
> **Accepted:** IEEE ICSIT 2026 (2nd IEEE International Conference on Sustainability, Innovation & Technology), Symbiosis Institute of Technology, Nagpur

---

## What is LightweightRAG?

LightweightRAG is a **CPU-only, training-free hybrid RAG pipeline** that reduces LLM hallucination using only off-the-shelf, open-source components — no GPU, no fine-tuning, no proprietary infrastructure required.

It combines:

- **BM25** sparse retrieval (keyword matching)
- **all-MiniLM-L6-v2** dense retrieval (semantic similarity, 22.7M params)
- **Reciprocal Rank Fusion (RRF)** — parameter-free combination of the two retrieval lists
- **Flan-T5-base** answer generation (250M params, instruction-tuned)
- **Evidence consistency verification** — cosine-similarity confidence gate for conditional abstention

### Key Results

| Dataset | Configuration | EM (%) | Hallucination Rate | Reduction vs. Baseline | Significance |
|---|---|---|---|---|---|
| SQuAD v2.0 — Answerable Subset (oracle extraction, n=300) | A: Baseline | 1.7 | 98.3% | — | — |
| SQuAD v2.0 — Answerable Subset (oracle extraction, n=300) | C: LightweightRAG | 59.3 | 39.7% | — | — |
| SQuAD v2.0 — Real Generation (n=500) | A: Baseline | 2.0 | 98.0% | — | — |
| SQuAD v2.0 — Real Generation (n=500) | C: LightweightRAG | 57.0 | **41.5%** | **57.7%** | p < 0.001 |
| TriviaQA — Cross-Domain (n=200) | A: Baseline | 8.0 | 92.0% | — | — |
| TriviaQA — Cross-Domain (n=200) | C: LightweightRAG | 31.5 | **67.7%** | **26.4%** | p < 0.001 |

All results use the identical, untuned configuration (**α = 0.05**) across both datasets — no dataset-specific tuning.

**Retrieval latency:** ~3–4 s per query on Colab CPU (2 vCPU, 12 GB RAM, no GPU).

---

## Repository Structure

```
lightweightrag-hallucination/
│
├── src/                          ← Core library (import this)
│   ├── __init__.py
│   ├── pipeline.py               ← LightweightRAG class (main pipeline)
│   ├── evaluation.py             ← EM, F1, McNemar test, bootstrap CI
│   └── data_utils.py             ← SQuAD v2.0 and TriviaQA data loaders
│
├── notebooks/
│   └── LightweightRAG_Experiments.ipynb   ← Reproduces ALL paper tables
│
├── assets/
│   ├── architecture.png          ← Figure 1: Pipeline architecture
│   └── threshold_sensitivity_curve.png    ← Figure 2: Threshold trade-off
│
├── requirements.txt              ← Pinned package versions
└── README.md
```

---

## Quick Start

### Option 1: Open in Colab (Recommended — no setup needed)

Click the **Open in Colab** badge above. The notebook installs all dependencies automatically.
> ⚠️ **Important:** In Colab, go to **Runtime → Change runtime type → CPU**.
> This paper specifically evaluates CPU-only performance — do NOT select GPU.

### Option 2: Run locally

```bash
git clone https://github.com/akanksha2130/lightweightrag-hallucination.git
cd lightweightrag-hallucination
pip install -r requirements.txt
```

```python
from src.pipeline import LightweightRAG

# Initialise pipeline (downloads models on first run, ~500 MB)
rag = LightweightRAG(verbose=True)

# Build index from your documents
passages = [
    "The Colorado River flows through the Grand Canyon in Arizona. "
    "It is approximately 1,450 miles long.",
    "The Eiffel Tower was completed in 1889 in Paris, France.",
]
rag.build_index(passages)

# Ask a question
result = rag.answer("What river flows through the Grand Canyon?")
print(result["answer"])     # -> "Colorado River"
print(result["abstained"])  # -> False
print(result["latency"])    # -> {'retrieval': 0.22, 'generation': 1.94, ...}
```

---

## Reproducing Paper Results

Open `notebooks/LightweightRAG_Experiments.ipynb` in Colab and run cells in order.

Each cell is labelled with the corresponding paper table:

| Experiment | Paper Table |
|---|---|
| SQuAD v2.0, oracle extraction (n=300) | Table I |
| SQuAD v2.0, real LLM generation (n=500) | Table II |
| TriviaQA, cross-domain (n=200) | Table III |
| Ablation study (BM25-only / Dense-only / Hybrid RRF) + LangChain external baseline | Table IV |
| Threshold sensitivity (α ∈ {0.05, 0.25, 0.50, 0.70, 0.80}) | Table V |
| Comparison with prior hallucination-aware RAG systems | Table VI |
| Manual HR-metric validation (n=150 sample, 84.7% human agreement) | Section VII-B |

---

## Pipeline Architecture

![Architecture](https://github.com/akanksha2130/lightweightrag-hallucination/raw/main/assets/architecture.png)

Four-stage pipeline: hybrid retrieval (BM25 + Dense via RRF) → Flan-T5-base generation → evidence-consistency verification via cosine similarity → conditional abstention below threshold α.

---

## API Reference

### `LightweightRAG`

```python
LightweightRAG(
    embed_model = "sentence-transformers/all-MiniLM-L6-v2",
    gen_model   = "google/flan-t5-base",
    alpha       = 0.05,   # evidence-verification confidence threshold (default, grid-searched on full SQuAD v2.0 set)
    top_k       = 5,      # chunks retrieved per retriever, re-ranked by RRF
    verbose     = False,
)
```

| Method | Description |
|---|---|
| `build_index(passages)` | Chunk passages (150 words, 30-word stride), build BM25 + FAISS index |
| `answer(question)` | Full pipeline → `{answer, abstained, chunks, latency}` |

### Threshold guide (Table V, SQuAD v2.0, n=500)

| α | Hallucination Rate | Refusal Rate | EM | Guidance |
|---|---|---|---|---|
| **0.05** | **41.5%** | 2.6% | **57.0%** | **Default** |
| 0.25 | 47.0% | 32.4% | 35.8% | Not recommended |
| 0.50 | 56.7% | 79.2% | 9.0% | Not recommended |
| 0.70 | 64.3% | 97.2% | 1.0% | Not recommended |
| 0.80 | 100.0% | 99.8% | 0.0% | No utility |

**Note:** unlike a conventional confidence gate, hallucination rate here *increases* with α — our cosine-similarity signal correlates only weakly (and at times negatively, Pearson r = −0.17) with correctness, so stricter thresholds preferentially retain longer, passage-echoing (and often incorrect) answers while discarding short, correct ones. We recommend the low default (α = 0.05) rather than tuning it upward.

---

## Hardware Requirements

| Component | Minimum | Used in paper |
|---|---|---|
| CPU | Any x86-64 | Google Colab free tier (2 vCPU) |
| RAM | 6 GB | 12 GB available |
| GPU | **Not required** | **Not used** |
| Storage | 2 GB (models) | — |

---

## Citation

If you use this code or paper, please cite:

```bibtex
@inproceedings{singh2026lightweightrag,
  title     = {LightweightRAG: A Statistically Validated, Training-Free Hybrid
               Retrieval-Augmented Generation Framework for Hallucination
               Reduction on CPU-Only Infrastructure},
  author    = {Singh, Akanksha and Kumar, Chandan},
  booktitle = {2nd IEEE International Conference on Sustainability, Innovation
               \& Technology (ICSIT 2026)},
  year      = {2026},
  note      = {Code: https://github.com/akanksha2130/lightweightrag-hallucination}
}
```

---

## License

MIT License — free to use, modify, and distribute with attribution.
