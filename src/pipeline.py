"""
LightweightRAG Pipeline
=======================
Training-free hybrid RAG pipeline for hallucination reduction.
Runs entirely on CPU — no GPU required.

Components:
  - BM25 sparse retrieval       (rank-bm25)
  - Dense semantic retrieval    (all-MiniLM-L6-v2 + FAISS-CPU)
  - Reciprocal Rank Fusion      (parameter-free combination)
  - Answer generation           (Flan-T5-base, beam search)
  - Evidence verification       (cosine similarity threshold)
  - Safe abstention             (when evidence is insufficient)

Usage:
    from src.pipeline import LightweightRAG
    rag = LightweightRAG()
    rag.build_index(passages)          # list of strings
    answer = rag.answer(question)
"""

import time
import numpy as np
from typing import List, Optional, Tuple, Dict

# ── Retrieval ──────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

# ── Generation ────────────────────────────────────────────────────────────
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Constants ─────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL     = "google/flan-t5-base"
TOP_K         = 5          # chunks retrieved per retriever
RRF_K         = 60         # RRF constant (standard value)
CHUNK_WORDS   = 150        # target chunk size
CHUNK_OVERLAP = 30         # overlap words between adjacent chunks
MAX_NEW_TOKENS_QA   = 50
MAX_NEW_TOKENS_BOOL = 5
NUM_BEAMS           = 4
TAU_EXTRACTIVE      = 0.25  # evidence verification threshold for span QA
TAU_BOOLEAN         = 0.30  # evidence verification threshold for boolean QA
ABSTENTION_RESPONSE = "I don't know based on the given context."


# ══════════════════════════════════════════════════════════════════════════
class LightweightRAG:
    """
    CPU-only, training-free hybrid RAG pipeline.

    Parameters
    ----------
    embed_model : str
        HuggingFace model name for sentence embeddings.
    gen_model : str
        HuggingFace model name for answer generation.
    tau_extractive : float
        Cosine similarity threshold for extractive QA verification.
    tau_boolean : float
        Question-context similarity threshold for boolean QA verification.
    top_k : int
        Number of chunks to retrieve per retriever.
    verbose : bool
        Print timing information.
    """

    def __init__(
        self,
        embed_model: str = EMBED_MODEL,
        gen_model:   str = GEN_MODEL,
        tau_extractive: float = TAU_EXTRACTIVE,
        tau_boolean:    float = TAU_BOOLEAN,
        top_k: int  = TOP_K,
        verbose: bool = False,
    ):
        self.tau_extractive = tau_extractive
        self.tau_boolean    = tau_boolean
        self.top_k          = top_k
        self.verbose        = verbose

        self._log("Loading embedding model …")
        self.embedder = SentenceTransformer(embed_model)

        self._log("Loading generation model …")
        self.tokenizer = T5Tokenizer.from_pretrained(gen_model)
        self.generator = T5ForConditionalGeneration.from_pretrained(gen_model)
        self.generator.eval()

        # Will be populated by build_index()
        self.chunks:        List[str]  = []
        self.bm25:          Optional[BM25Okapi] = None
        self.faiss_index:   Optional[faiss.IndexFlatIP] = None
        self.chunk_embeddings: Optional[np.ndarray] = None

    # ── Index building ──────────────────────────────────────────────────
    def build_index(self, passages: List[str]) -> None:
        """
        Chunk all passages, build BM25 index and FAISS index.

        Parameters
        ----------
        passages : list of str
            Raw document strings to index.
        """
        self._log("Chunking documents …")
        self.chunks = self._chunk_passages(passages)
        self._log(f"  → {len(self.chunks)} chunks created.")

        # BM25
        self._log("Building BM25 index …")
        tokenized = [c.lower().split() for c in self.chunks]
        self.bm25  = BM25Okapi(tokenized, k1=1.5, b=0.75)

        # Dense embeddings + FAISS
        self._log("Encoding chunks with MiniLM (one-time cost) …")
        t0 = time.time()
        embeddings = self.embedder.encode(
            self.chunks, batch_size=64, show_progress_bar=self.verbose,
            convert_to_numpy=True, normalize_embeddings=True
        )
        self.chunk_embeddings = embeddings.astype("float32")
        self._log(f"  → Encoded in {time.time()-t0:.1f}s")

        dim = self.chunk_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)   # inner-product on L2-normalised = cosine
        self.faiss_index.add(self.chunk_embeddings)
        self._log("Index ready.")

    # ── Public query interface ──────────────────────────────────────────
    def answer(self, question: str, mode: str = "extractive") -> Dict:
        """
        Answer a question using the full LightweightRAG pipeline.

        Parameters
        ----------
        question : str
            Natural language question.
        mode : str
            "extractive" for span QA (SQuAD-style),
            "boolean"    for yes/no QA (BoolQ-style).

        Returns
        -------
        dict with keys:
            answer       : str  — generated answer or abstention string
            abstained    : bool — True if evidence verification failed
            chunks       : list — top-k retrieved chunk strings
            latency      : dict — per-component timing in seconds
        """
        assert self.bm25 is not None, "Call build_index() before answer()."

        t_total = time.time()
        latency = {}

        # 1. Hybrid retrieval
        t0 = time.time()
        top_chunks = self._hybrid_retrieve(question)
        latency["retrieval"] = round(time.time() - t0, 3)

        context = " ".join(top_chunks)

        # 2. Generation
        t0 = time.time()
        raw_answer = self._generate(question, context, mode)
        latency["generation"] = round(time.time() - t0, 3)

        # 3. Evidence verification
        t0 = time.time()
        if mode == "extractive":
            verified = self._verify_extractive(raw_answer, top_chunks)
        else:
            verified = self._verify_boolean(question, top_chunks)
        latency["verification"] = round(time.time() - t0, 3)

        # 4. Safe abstention
        if not verified:
            final_answer = ABSTENTION_RESPONSE
            abstained    = True
        else:
            final_answer = raw_answer
            abstained    = False

        latency["total"] = round(time.time() - t_total, 3)

        return {
            "answer":   final_answer,
            "abstained": abstained,
            "chunks":   top_chunks,
            "latency":  latency,
        }

    # ── Retrieval ───────────────────────────────────────────────────────
    def _hybrid_retrieve(self, query: str) -> List[str]:
        """BM25 + Dense retrieval fused with Reciprocal Rank Fusion."""
        # BM25 ranks
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks  = np.argsort(-bm25_scores)[:self.top_k].tolist()

        # Dense ranks
        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        _, dense_idxs = self.faiss_index.search(q_emb, self.top_k)
        dense_ranks = dense_idxs[0].tolist()

        # RRF fusion
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        top_idxs = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:self.top_k]
        return [self.chunks[i] for i in top_idxs]

    # ── Generation ──────────────────────────────────────────────────────
    def _generate(self, question: str, context: str, mode: str) -> str:
        """Run Flan-T5-base generation with the appropriate prompt template."""
        if mode == "boolean":
            prompt = (
                f"Read the passage and answer yes or no.\n"
                f"Passage: {context}\n"
                f"Question: {question}\n"
                f"Answer yes or no:"
            )
            max_new = MAX_NEW_TOKENS_BOOL
        else:
            prompt = (
                f"Read the passage and answer the question.\n"
                f"Passage: {context}\n"
                f"Question: {question}"
            )
            max_new = MAX_NEW_TOKENS_QA

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=512, truncation=True
        )
        outputs = self.generator.generate(
            **inputs,
            num_beams=NUM_BEAMS,
            max_new_tokens=max_new,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ── Evidence verification ────────────────────────────────────────────
    def _verify_extractive(self, answer: str, chunks: List[str]) -> bool:
        """Cosine similarity between answer embedding and chunk embeddings."""
        if not answer or len(answer.split()) < 2:
            return False
        ans_emb = self.embedder.encode(
            [answer], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        chunk_embs = self.embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        sims = (chunk_embs @ ans_emb.T).flatten()
        return float(sims.max()) >= self.tau_extractive

    def _verify_boolean(self, question: str, chunks: List[str]) -> bool:
        """Question-context cosine similarity for boolean QA."""
        q_emb = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        chunk_embs = self.embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        sims = (chunk_embs @ q_emb.T).flatten()
        return float(sims.max()) >= self.tau_boolean

    # ── Chunking ─────────────────────────────────────────────────────────
    def _chunk_passages(self, passages: List[str]) -> List[str]:
        """Split passages into overlapping word-level chunks."""
        chunks = []
        for passage in passages:
            words = passage.split()
            if len(words) <= CHUNK_WORDS:
                chunks.append(passage.strip())
                continue
            start = 0
            while start < len(words):
                end   = min(start + CHUNK_WORDS, len(words))
                chunk = " ".join(words[start:end])
                chunks.append(chunk)
                if end == len(words):
                    break
                start += CHUNK_WORDS - CHUNK_OVERLAP
        return [c for c in chunks if c.strip()]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[LightweightRAG] {msg}")
