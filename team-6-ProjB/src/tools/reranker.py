"""
CardioAgent Reranker  Cross-Encoder Re-Scoring
==================================================
Two-stage retrieval pipeline:
  Stage 1: Dense retrieval (ECG-FM / BioViL-T / BiomedBERT ? top-k candidates)
  Stage 2: Cross-encoder reranking (this module ? re-scored top-n)

Why rerank?
  Dense retrieval is fast but uses independent encoding (query and document
  encoded separately). Cross-encoders jointly attend over query+document,
  capturing fine-grained relevance that bi-encoders miss.

  In medical RAG this matters because two ECG reports might share keywords
  ("sinus tachycardia") but differ critically in context ("post-exercise"
  vs "at rest with chest pain"). A cross-encoder catches this.

Models:
  MedCPT Cross-Encoder   trained on PubMed query-article pairs, biomedical
                           relevance scoring. Primary choice.
  BiomedBERT CE          fallback if MedCPT unavailable.

Integration:
  Sits between VectorCaseMemory.retrieve() and the planner.
  vector_memory retrieves top-k (e.g. 20) ? reranker re-scores ? top-n (e.g. 5)

Usage:
    reranker = MedCPTReranker(device="cuda")
    reranked = reranker.rerank(query_text, candidates, top_n=5)
"""

import logging
import time
import numpy as np
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class MedCPTReranker:
    """
    MedCPT cross-encoder for biomedical relevance reranking.

    Takes a query (constructed from patient findings) and a list of
    candidate cases, jointly scores each (query, candidate) pair,
    and returns the top-n by relevance.

    The cross-encoder sees the full query+document together, so it
    can capture token-level interactions that bi-encoder retrieval misses.
    """

    def __init__(
        self,
        checkpoint_path: str = "ncbi/MedCPT-Cross-Encoder",
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading MedCPT cross-encoder from {self.checkpoint_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint_path,
        )
        self._model.to(self.device).eval()
        logger.info("MedCPT cross-encoder loaded")

    def rerank(
        self,
        query: str,
        candidates: list,
        top_n: int = 5,
        text_key: str = "report",
        diagnosis_key: str = "diagnosis",
    ) -> list:
        """
        Rerank retrieved candidates using cross-encoder scoring.

        Args:
            query: Patient-specific query string (built from phenotypes/findings)
            candidates: List of dicts (RetrievedCase.to_dict() or raw records)
            top_n: Number of results to return after reranking
            text_key: Key in candidate dict for document text
            diagnosis_key: Key in candidate dict for diagnosis text

        Returns:
            List of candidates (same format), reordered by cross-encoder score,
            with "rerank_score" field added.
        """
        if not candidates:
            return []

        if len(candidates) <= top_n:
            # Not enough candidates to justify reranking cost
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            return candidates

        t0 = time.time()

        # Build document texts from candidates
        doc_texts = []
        for c in candidates:
            parts = []
            dx = c.get(diagnosis_key, "")
            if dx:
                parts.append(f"Diagnosis: {dx}")
            report = c.get(text_key, "")
            if report:
                parts.append(report[:400])
            doc_texts.append(" ".join(parts) if parts else "No information")

        # Score all (query, document) pairs
        scores = self._score_pairs(query, doc_texts)

        # Attach scores and sort
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(f"  Reranked {len(candidates)} ? top {top_n} "
                     f"({time.time()-t0:.2f}s)")

        return reranked[:top_n]

    def _score_pairs(self, query: str, documents: List[str]) -> List[float]:
        """Score (query, document) pairs in batches."""
        import torch

        all_scores = []

        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]

            # MedCPT cross-encoder expects (query, document) pairs
            pairs = [(query, doc) for doc in batch_docs]

            inputs = self.tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Cross-encoder outputs logits; take the relevance score
                if outputs.logits.shape[-1] == 1:
                    scores = outputs.logits.squeeze(-1)
                else:
                    # Binary classifier: take positive class probability
                    scores = torch.softmax(outputs.logits, dim=-1)[:, 1]

            all_scores.extend(scores.cpu().tolist())

        return all_scores

    def build_query_from_state(self, state) -> str:
        """
        Build a reranking query from AgentStateV3.

        Combines phenotype terms, key findings, and hypothesis names
        into a focused query string for the cross-encoder.
        """
        parts = []

        # Phenotype terms (most structured)
        if hasattr(state, "phenotype_terms") and state.phenotype_terms:
            terms = [p["term"] for p in state.phenotype_terms[:10]]
            parts.append("Phenotypes: " + ", ".join(terms))

        # Top hypothesis
        if hasattr(state, "hypothesis_ranking") and state.hypothesis_ranking:
            top = state.hypothesis_ranking[0]
            parts.append(f"Suspected diagnosis: {top.diagnosis}")

        # Key ECG findings
        if state.ecg_result:
            m = state.ecg_result.get("metrics", {})
            ecg_parts = []
            if m.get("rhythm_detail"):
                ecg_parts.append(m["rhythm_detail"])
            hr = m.get("heart_rate_bpm")
            if hr:
                ecg_parts.append(f"HR {hr} bpm")
            for st in m.get("st_findings", []):
                ecg_parts.append(f"ST {st.get('type', '')} {st.get('lead', '')}")
            if ecg_parts:
                parts.append("ECG: " + ", ".join(ecg_parts))

        # Abnormal labs
        if state.lab_result:
            abnormals = state.lab_result.get("abnormal", [])
            if abnormals:
                lab_strs = [f"{a['test']} {a['status']}" for a in abnormals[:5]]
                parts.append("Labs: " + ", ".join(lab_strs))

        return ". ".join(parts) if parts else "cardiac patient"


class LLMRelevanceGate:
    """
    Optional LLM-based relevance gate.

    After cross-encoder reranking, use a lightweight LLM call to
    filter out cases that scored high on surface similarity but
    are clinically irrelevant.

    This is the "LLM-based relevance gating" from the roadmap.
    Only use when you need maximum precision (e.g., <3 RAG cases
    fed to the planner) and can afford the extra LLM call.
    """

    def __init__(
        self,
        llm_url: str = "http://localhost:8000/v1",
        model_name: str = "qwen3-vl",
        threshold: float = 0.5,
    ):
        self.llm_url = llm_url
        self.model_name = model_name
        self.threshold = threshold

    def filter(
        self,
        query_summary: str,
        candidates: list,
        max_keep: int = 3,
    ) -> list:
        """
        LLM scores each candidate's clinical relevance to the query.
        Returns only candidates above threshold, up to max_keep.
        """
        from openai import OpenAI
        client = OpenAI(base_url=self.llm_url, api_key="cardioagent")

        kept = []

        for c in candidates:
            dx = c.get("diagnosis", "N/A")
            report = c.get("report", "")[:200]

            prompt = (
                f"Current patient: {query_summary}\n\n"
                f"Retrieved case  Diagnosis: {dx}\n"
                f"Report excerpt: {report}\n\n"
                f"Is this retrieved case clinically relevant to the current "
                f"patient's presentation? Answer with a relevance score "
                f"from 0.0 to 1.0 and one sentence of reasoning.\n"
                f"Format: SCORE: 0.X REASON: ..."
            )

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.0,
                )
                text = response.choices[0].message.content

                # Parse score
                import re
                match = re.search(r'SCORE:\s*([\d.]+)', text)
                if match:
                    score = float(match.group(1))
                    c["llm_relevance"] = score
                    c["llm_reason"] = text
                    if score >= self.threshold:
                        kept.append(c)
                        if len(kept) >= max_keep:
                            break
            except Exception as e:
                logger.warning(f"LLM gate failed for case: {e}")
                # On failure, keep the case (conservative)
                kept.append(c)
                if len(kept) >= max_keep:
                    break

        logger.info(f"  LLM gate: {len(kept)}/{len(candidates)} kept")
        return kept