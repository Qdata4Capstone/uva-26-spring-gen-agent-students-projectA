"""
CardioAgent Vector Memory — Embedding-Based RAG Retrieval
============================================================
Replaces keyword-based CaseMemory with dense vector retrieval
using per-modality embedding models and ChromaDB.

Architecture:
  - Separate ChromaDB collections per modality (ecg, cxr, text)
  - Late fusion: retrieve top-k per modality, then merge + rerank
  - Fully compatible with planner_v3's CaseMemory interface

Two phases:
  1. OFFLINE INDEX BUILD — run once per dataset expansion
     python vector_memory.py build --data-dir /path/to/data

  2. ONLINE RETRIEVAL — called by planner at inference time
     memory = VectorCaseMemory(db_path="/path/to/chroma_db")
     cases = memory.retrieve(ecg_signal, cxr_path, lab_text, top_k=5)

Fusion strategies:
  - "rrf"    — Reciprocal Rank Fusion (default, no tuning needed)
  - "score"  — Weighted score averaging across modalities
  - "concat" — Concatenate embeddings, single query (requires aligned dims)
"""

import json
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedCase:
    """A single case retrieved from vector memory."""
    case_id: str
    source: str                          # "mimic_iv_ecg", "symile", etc.
    diagnosis: str = ""
    report: str = ""
    score: float = 0.0                   # Fused retrieval score
    modality_scores: Dict[str, float] = field(default_factory=dict)
    modalities_matched: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.case_id,
            "source": self.source,
            "diagnosis": self.diagnosis,
            "report": self.report,
            "score": round(self.score, 4),
            "modality_scores": {
                k: round(v, 4) for k, v in self.modality_scores.items()
            },
            "modalities_matched": self.modalities_matched,
        }


# ═══════════════════════════════════════════════════════════════════════
# Vector Case Memory
# ═══════════════════════════════════════════════════════════════════════

class VectorCaseMemory:
    """
    Multi-modal vector retrieval using ChromaDB.

    Maintains separate collections per modality:
      - ecg_embeddings   : ECG-FM vectors
      - cxr_embeddings   : BioViL-T vectors
      - text_embeddings  : BiomedBERT vectors (clinical notes, lab text)

    Retrieval uses late fusion: query each collection independently,
    then merge results via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        db_path: str,
        ecg_embedder=None,
        cxr_embedder=None,
        text_embedder=None,
        fusion_strategy: str = "rrf",
        rrf_k: int = 60,
        reranker=None,
        rerank_top_n: Optional[int] = None,
    ):
        """
        Args:
            db_path: Path to ChromaDB persistent storage
            ecg_embedder: ECGFMEmbedder instance (None = skip ECG retrieval)
            cxr_embedder: BioViLTEmbedder instance (None = skip CXR retrieval)
            text_embedder: BiomedBERTEmbedder instance (None = skip text retrieval)
            fusion_strategy: "rrf" or "score"
            rrf_k: RRF smoothing constant (default 60)
            reranker: MedCPTReranker instance (None = skip reranking)
            rerank_top_n: Final count after reranking (None = same as top_k)
        """
        self.db_path = db_path
        self.ecg_embedder = ecg_embedder
        self.cxr_embedder = cxr_embedder
        self.text_embedder = text_embedder
        self.fusion_strategy = fusion_strategy
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n

        self._client = None
        self._collections = {}
        self.enabled = False

        if os.path.exists(db_path):
            self._init_client()

    def _init_client(self):
        """Initialize ChromaDB client and load existing collections."""
        import chromadb

        self._client = chromadb.PersistentClient(path=self.db_path)

        # Load whichever collections exist
        existing = {c.name for c in self._client.list_collections()}
        for name in ["ecg_embeddings", "cxr_embeddings", "text_embeddings"]:
            if name in existing:
                self._collections[name] = self._client.get_collection(name)
                count = self._collections[name].count()
                logger.info(f"  VectorMemory: loaded {name} ({count} records)")

        self.enabled = len(self._collections) > 0
        logger.info(f"  VectorMemory: {len(self._collections)} collections, "
                     f"enabled={self.enabled}")

    # ═══════════════════════════════════════════════════════════════════
    # Online Retrieval
    # ═══════════════════════════════════════════════════════════════════

    def retrieve(
        self,
        ecg_signal: Optional[np.ndarray] = None,
        cxr_image=None,
        clinical_text: Optional[str] = None,
        top_k: int = 5,
        per_modality_k: int = 10,
        rerank_query: Optional[str] = None,
    ) -> List[RetrievedCase]:
        """
        Multi-modal retrieval with late fusion + optional reranking.

        Args:
            ecg_signal: Raw ECG array (any shape ECGFMEmbedder supports)
            cxr_image: CXR image path, PIL Image, or numpy array
            clinical_text: Clinical notes / lab text string
            top_k: Number of final results to return
            per_modality_k: Candidates per modality before fusion
            rerank_query: Rich text query for cross-encoder reranking
                          (built from phenotypes/hypotheses by planner)
            top_k: Number of final fused results to return
            per_modality_k: Candidates per modality before fusion

        Returns:
            List of RetrievedCase, sorted by fused score (descending)
        """
        if not self.enabled:
            return []

        per_modality_results = {}

        # ── ECG retrieval ──
        if (ecg_signal is not None
                and self.ecg_embedder is not None
                and "ecg_embeddings" in self._collections):
            try:
                t0 = time.time()
                ecg_vec = self.ecg_embedder.embed(ecg_signal)
                results = self._query_collection(
                    "ecg_embeddings", ecg_vec, per_modality_k,
                )
                per_modality_results["ecg"] = results
                logger.info(f"  ECG retrieval: {len(results)} results "
                            f"({time.time()-t0:.2f}s)")
            except Exception as e:
                logger.warning(f"  ECG retrieval failed: {e}")

        # ── CXR retrieval ──
        if (cxr_image is not None
                and self.cxr_embedder is not None
                and "cxr_embeddings" in self._collections):
            try:
                t0 = time.time()
                cxr_vec = self.cxr_embedder.embed(cxr_image)
                results = self._query_collection(
                    "cxr_embeddings", cxr_vec, per_modality_k,
                )
                per_modality_results["cxr"] = results
                logger.info(f"  CXR retrieval: {len(results)} results "
                            f"({time.time()-t0:.2f}s)")
            except Exception as e:
                logger.warning(f"  CXR retrieval failed: {e}")

        # ── Text retrieval ──
        if (clinical_text is not None
                and self.text_embedder is not None
                and "text_embeddings" in self._collections):
            try:
                t0 = time.time()
                text_vec = self.text_embedder.embed(clinical_text)
                results = self._query_collection(
                    "text_embeddings", text_vec, per_modality_k,
                )
                per_modality_results["text"] = results
                logger.info(f"  Text retrieval: {len(results)} results "
                            f"({time.time()-t0:.2f}s)")
            except Exception as e:
                logger.warning(f"  Text retrieval failed: {e}")

        if not per_modality_results:
            return []

        # ── Fuse results ──
        # When reranker is present, retrieve more candidates for reranking
        fusion_k = top_k * 4 if self.reranker else top_k

        if self.fusion_strategy == "rrf":
            fused = self._fuse_rrf(per_modality_results, fusion_k)
        elif self.fusion_strategy == "score":
            fused = self._fuse_score(per_modality_results, fusion_k)
        else:
            fused = self._fuse_rrf(per_modality_results, fusion_k)

        logger.info(f"  Fused retrieval: {len(fused)} cases "
                     f"from {list(per_modality_results.keys())}")

        # ── Stage 2: Cross-encoder reranking ──
        if self.reranker and fused:
            rerank_n = self.rerank_top_n or top_k

            # Prefer planner-built query (has phenotypes/hypotheses),
            # fall back to raw input-based query
            query_text = rerank_query or self._build_rerank_query(
                ecg_signal, cxr_image, clinical_text,
            )

            # Convert RetrievedCase → dict for reranker
            candidate_dicts = [c.to_dict() for c in fused]

            reranked_dicts = self.reranker.rerank(
                query=query_text,
                candidates=candidate_dicts,
                top_n=rerank_n,
            )

            # Convert back to RetrievedCase with updated scores
            fused = [
                RetrievedCase(
                    case_id=d["patient_id"],
                    source=d.get("source", "unknown"),
                    diagnosis=d.get("diagnosis", ""),
                    report=d.get("report", ""),
                    score=d.get("rerank_score", d.get("score", 0.0)),
                    modality_scores=d.get("modality_scores", {}),
                    modalities_matched=d.get("modalities_matched", []),
                    metadata={"reranked": True,
                              "dense_score": d.get("score", 0.0)},
                )
                for d in reranked_dicts
            ]
            logger.info(f"  Reranked → {len(fused)} cases")

        return fused[:top_k]

    def _build_rerank_query(self, ecg_signal, cxr_image, clinical_text) -> str:
        """Build a text query for the cross-encoder reranker."""
        parts = []
        if clinical_text and len(clinical_text.strip()) > 20:
            parts.append(clinical_text[:300])
        if ecg_signal is not None:
            parts.append("Patient has ECG data available.")
        if cxr_image is not None:
            parts.append("Patient has chest X-ray available.")
        return " ".join(parts) if parts else "cardiac patient"

    def _query_collection(
        self,
        collection_name: str,
        query_vec: np.ndarray,
        top_k: int,
    ) -> List[Tuple[str, float, dict]]:
        """
        Query a single ChromaDB collection.

        Returns:
            List of (case_id, distance, metadata) tuples
        """
        collection = self._collections[collection_name]
        results = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            include=["distances", "metadatas"],
        )

        output = []
        if results and results["ids"] and results["ids"][0]:
            for case_id, dist, meta in zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                # ChromaDB returns L2 distance; convert to similarity
                similarity = 1.0 / (1.0 + dist)
                output.append((case_id, similarity, meta or {}))

        return output

    # ═══════════════════════════════════════════════════════════════════
    # Fusion Strategies
    # ═══════════════════════════════════════════════════════════════════

    def _fuse_rrf(
        self,
        per_modality: Dict[str, List[Tuple[str, float, dict]]],
        top_k: int,
    ) -> List[RetrievedCase]:
        """
        Reciprocal Rank Fusion.

        RRF(d) = Σ_m  1 / (k + rank_m(d))

        where k is a smoothing constant (default 60) and rank_m(d) is
        the rank of document d in modality m's results.
        Robust, parameter-free, works well with heterogeneous score scales.
        """
        case_scores: Dict[str, Dict] = {}

        for modality, results in per_modality.items():
            for rank, (case_id, sim, meta) in enumerate(results):
                if case_id not in case_scores:
                    case_scores[case_id] = {
                        "rrf_score": 0.0,
                        "modality_scores": {},
                        "modalities": [],
                        "meta": meta,
                    }
                rrf_contribution = 1.0 / (self.rrf_k + rank + 1)
                case_scores[case_id]["rrf_score"] += rrf_contribution
                case_scores[case_id]["modality_scores"][modality] = sim
                case_scores[case_id]["modalities"].append(modality)

        # Sort by RRF score
        ranked = sorted(
            case_scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True,
        )

        return [
            RetrievedCase(
                case_id=case_id,
                source=info["meta"].get("source", "unknown"),
                diagnosis=info["meta"].get("diagnosis", ""),
                report=info["meta"].get("report", "")[:300],
                score=info["rrf_score"],
                modality_scores=info["modality_scores"],
                modalities_matched=info["modalities"],
                metadata=info["meta"],
            )
            for case_id, info in ranked[:top_k]
        ]

    def _fuse_score(
        self,
        per_modality: Dict[str, List[Tuple[str, float, dict]]],
        top_k: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[RetrievedCase]:
        """
        Weighted score averaging.

        Default weights: ecg=0.4, cxr=0.4, text=0.2
        Customize via weights parameter.
        """
        if weights is None:
            weights = {"ecg": 0.4, "cxr": 0.4, "text": 0.2}

        case_scores: Dict[str, Dict] = {}

        for modality, results in per_modality.items():
            w = weights.get(modality, 0.33)
            for case_id, sim, meta in results:
                if case_id not in case_scores:
                    case_scores[case_id] = {
                        "weighted_score": 0.0,
                        "modality_scores": {},
                        "modalities": [],
                        "meta": meta,
                        "n_modalities": 0,
                    }
                case_scores[case_id]["weighted_score"] += w * sim
                case_scores[case_id]["modality_scores"][modality] = sim
                case_scores[case_id]["modalities"].append(modality)
                case_scores[case_id]["n_modalities"] += 1

        ranked = sorted(
            case_scores.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True,
        )

        return [
            RetrievedCase(
                case_id=case_id,
                source=info["meta"].get("source", "unknown"),
                diagnosis=info["meta"].get("diagnosis", ""),
                report=info["meta"].get("report", "")[:300],
                score=info["weighted_score"],
                modality_scores=info["modality_scores"],
                modalities_matched=info["modalities"],
                metadata=info["meta"],
            )
            for case_id, info in ranked[:top_k]
        ]

    # ═══════════════════════════════════════════════════════════════════
    # Offline Index Building
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def build_index(
        records: List[dict],
        db_path: str,
        ecg_embedder=None,
        cxr_embedder=None,
        text_embedder=None,
        ecg_signal_loader=None,
        cxr_image_loader=None,
        text_field: str = "report",
        batch_size: int = 32,
        source_name: str = "mimic_iv_ecg",
    ):
        """
        Build ChromaDB index from a list of records.

        Args:
            records: List of dicts, each with at least "patient_id"
            db_path: Path for ChromaDB persistent storage
            ecg_embedder: ECGFMEmbedder (None = skip ECG indexing)
            cxr_embedder: BioViLTEmbedder (None = skip CXR indexing)
            text_embedder: BiomedBERTEmbedder (None = skip text indexing)
            ecg_signal_loader: Callable(record) → np.ndarray or None
            cxr_image_loader: Callable(record) → image_path or None
            text_field: Key in record dict to use for text embedding
            batch_size: Batch size for embedding computation
            source_name: Dataset name to store in metadata
        """
        import chromadb

        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)

        logger.info(f"Building index for {len(records)} records → {db_path}")

        # ── ECG embeddings ──
        if ecg_embedder and ecg_signal_loader:
            logger.info("  Indexing ECG embeddings...")
            collection = client.get_or_create_collection(
                name="ecg_embeddings",
                metadata={"hnsw:space": "l2"},
            )
            _batch_index(
                collection=collection,
                records=records,
                embedder=ecg_embedder,
                data_loader=ecg_signal_loader,
                source_name=source_name,
                batch_size=batch_size,
                modality="ecg",
            )

        # ── CXR embeddings ──
        if cxr_embedder and cxr_image_loader:
            logger.info("  Indexing CXR embeddings...")
            collection = client.get_or_create_collection(
                name="cxr_embeddings",
                metadata={"hnsw:space": "l2"},
            )
            _batch_index(
                collection=collection,
                records=records,
                embedder=cxr_embedder,
                data_loader=cxr_image_loader,
                source_name=source_name,
                batch_size=batch_size,
                modality="cxr",
            )

        # ── Text embeddings ──
        if text_embedder:
            logger.info("  Indexing text embeddings...")
            collection = client.get_or_create_collection(
                name="text_embeddings",
                metadata={"hnsw:space": "l2"},
            )

            def text_loader(record):
                text = record.get(text_field, "")
                if not text or len(text.strip()) < 20:
                    return None
                return text

            _batch_index_text(
                collection=collection,
                records=records,
                embedder=text_embedder,
                text_loader=text_loader,
                source_name=source_name,
                batch_size=batch_size,
            )

        logger.info("  Index build complete!")


def _batch_index(
    collection,
    records: List[dict],
    embedder,
    data_loader,
    source_name: str,
    batch_size: int,
    modality: str,
):
    """Batch-embed and upsert into a ChromaDB collection (ECG or CXR)."""
    total = len(records)
    indexed = 0
    skipped = 0

    for i in range(0, total, batch_size):
        batch_records = records[i:i + batch_size]
        batch_data = []
        batch_ids = []
        batch_meta = []

        for rec in batch_records:
            case_id = rec.get("patient_id", f"unknown_{i}")
            try:
                data = data_loader(rec)
                if data is None:
                    skipped += 1
                    continue
                batch_data.append(data)
                batch_ids.append(str(case_id))
                meta_dict = {
                    "source": source_name,
                    "diagnosis": str(rec.get("diagnosis", ""))[:200],
                    "report": str(rec.get("report", ""))[:300],
                    "split": rec.get("split", "train"),
                }
                # Preserve modality-specific paths so retrieved cases are
                # visualizable in the UI without rebuilding from scratch.
                for key in ("cxr_path", "waveform_path", "view",
                            "subject_id", "study_id", "dicom_id"):
                    if rec.get(key):
                        meta_dict[key] = str(rec[key])[:200]
                batch_meta.append(meta_dict)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    logger.warning(f"    Skip {case_id}: {e}")

        if not batch_data:
            continue

        try:
            embeddings = embedder.embed_batch(batch_data, batch_size=batch_size)
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                metadatas=batch_meta,
            )
            indexed += len(batch_ids)
        except Exception as e:
            logger.error(f"    Batch embed failed: {e}")

        if (i // batch_size) % 10 == 0:
            logger.info(f"    [{modality}] {indexed}/{total} indexed, "
                        f"{skipped} skipped")

    logger.info(f"    [{modality}] Done: {indexed} indexed, {skipped} skipped")


def _batch_index_text(
    collection,
    records: List[dict],
    embedder,
    text_loader,
    source_name: str,
    batch_size: int,
):
    """Batch-embed text and upsert into ChromaDB."""
    total = len(records)
    indexed = 0
    skipped = 0

    for i in range(0, total, batch_size):
        batch_records = records[i:i + batch_size]
        batch_texts = []
        batch_ids = []
        batch_meta = []

        for rec in batch_records:
            case_id = rec.get("patient_id", f"unknown_{i}")
            text = text_loader(rec)
            if text is None:
                skipped += 1
                continue
            batch_texts.append(text)
            batch_ids.append(str(case_id))
            meta_dict_t = {
                "source": source_name,
                "diagnosis": str(rec.get("diagnosis", ""))[:200],
                "report": str(rec.get("report", ""))[:300],
                "split": rec.get("split", "train"),
            }
            for key in ("cxr_path", "waveform_path", "view",
                        "subject_id", "study_id", "dicom_id"):
                if rec.get(key):
                    meta_dict_t[key] = str(rec[key])[:200]
            batch_meta.append(meta_dict_t)

        if not batch_texts:
            continue

        try:
            embeddings = embedder.embed_batch(batch_texts, batch_size=batch_size)
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                metadatas=batch_meta,
            )
            indexed += len(batch_ids)
        except Exception as e:
            logger.error(f"    Batch text embed failed: {e}")

        if (i // batch_size) % 10 == 0:
            logger.info(f"    [text] {indexed}/{total} indexed, "
                        f"{skipped} skipped")

    logger.info(f"    [text] Done: {indexed} indexed, {skipped} skipped")


# ═══════════════════════════════════════════════════════════════════════
# CLI — Offline Index Builder
# ═══════════════════════════════════════════════════════════════════════

def _resolve_ecg_path(ecg_data_dir: Path, waveform_path: str) -> Optional[str]:
    """
    Resolve a waveform_path from the JSONL index to an actual file.

    MIMIC-IV-ECG directory structure (v1.0):
        files/pNNNN/pXXXXXXXX/sZZZZZZZZZ/ZZZZZZZZZ.hea
        files/pNNNN/pXXXXXXXX/sZZZZZZZZZ/ZZZZZZZZZ.dat

    The waveform_path in the index is relative, e.g.:
        files/p1000/p10001725/s41420867/41420867

    Args:
        ecg_data_dir: Root of MIMIC-IV-ECG (contains 'files/' directory)
        waveform_path: Relative path from index JSONL
    Returns:
        Full path to the record (without extension) or None if not found
    """
    if not waveform_path:
        return None

    full = ecg_data_dir / waveform_path

    # Check if .hea exists at the resolved path
    if full.with_suffix(".hea").exists():
        return str(full)

    # Sometimes the index has the path without the group dir — try searching
    # e.g. "files/p10001725/s41420867/41420867" missing the pNNNN level
    study_id = Path(waveform_path).stem
    matches = list(ecg_data_dir.rglob(f"{study_id}.hea"))
    if matches:
        return str(matches[0].with_suffix(""))

    return None


def _resolve_cxr_path(cxr_data_dir: Path, cxr_rel_path: str) -> Optional[str]:
    """
    Resolve a CXR image path from CSV/index to an actual file.

    MIMIC-CXR-JPG directory structure:
        files/p10/p10000032/s50414267/02aa804e-bde0afdd-....jpg
        files/p11/p11000032/s53214267/...

    The cxr_path in the index is relative, e.g.:
        files/p10/p10000032/s50414267/02aa804e-bde0afdd-....jpg

    Args:
        cxr_data_dir: Root of MIMIC-CXR-JPG (contains 'files/' directory)
        cxr_rel_path: Relative path from index/CSV
    Returns:
        Full path to the image or None if not found
    """
    if not cxr_rel_path:
        return None

    full = cxr_data_dir / cxr_rel_path
    if full.exists():
        return str(full)

    # Try searching by filename
    filename = Path(cxr_rel_path).name
    matches = list(cxr_data_dir.rglob(filename))
    if matches:
        return str(matches[0])

    return None


def main():
    """
    CLI for building vector indices.

    Two commands:

      build-train — Unified build for the RAG knowledge base
                    (one pass indexes ECG + CXR + clinical text together)
                    Used at offline index construction time.

      build-symile — Optional build for Symile train split
                     (for experiments that want Symile as a RAG source)
                     NOTE: Symile TEST samples are embedded on-the-fly
                     by the planner at inference time — not stored.

    Usage:
        # Main use case: unified training index from MIMIC-IV
        python vector_memory.py build-train \
            --ecg-index /path/to/mimic_iv_ecg_index.jsonl \
            --ecg-data-dir /path/to/mimic-iv-ecg \
            --cxr-data-dir /path/to/mimic-cxr-jpg \
            --db-path /path/to/chroma_db \
            --ecg-model /path/to/ecg_fm \
            --cxr-model biovilt \
            --text-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

        # Symile train split (optional RAG source)
        python vector_memory.py build-symile \
            --data-dir /path/to/symile-mimic \
            --db-path /path/to/chroma_db \
            --ecg-model /path/to/ecg_fm \
            --cxr-model biovilt \
            --text-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

    Linking MIMIC-IV-ECG to MIMIC-CXR-JPG:
        Records are joined by subject_id. For each ECG record, we look up
        the patient's CXR studies in MIMIC-CXR-JPG. If --cxr-metadata
        is provided (mimic-cxr-2.0.0-metadata.csv.gz), we pick the CXR
        closest in time to the ECG; otherwise we use the first CXR found.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Vector Memory Index Builder")
    sub = parser.add_subparsers(dest="command")

    # ══════════════════════════════════════════════════════════════
    # build-train: unified ECG + CXR + text index from MIMIC-IV
    # ══════════════════════════════════════════════════════════════
    bt = sub.add_parser(
        "build-train",
        help="Build unified training RAG index (ECG + CXR + text in one pass)",
    )
    bt.add_argument("--ecg-index", required=True,
                    help="MIMIC-IV-ECG JSONL index (mimic_iv_ecg_index.jsonl)")
    bt.add_argument("--ecg-data-dir", default="/sfs/weka/scratch/rzv4ve/cardioagent/data/MIMIC-IV-ECG-1/",
                    help="MIMIC-IV-ECG root (contains files/pNNNN/pXXX.../sZZZ.../)")
    bt.add_argument("--cxr-data-dir", default="/sfs/weka/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg-2.0.0.physionet.org",
                    help="MIMIC-CXR-JPG root (contains files/p10/p100.../). "
                         "If omitted, CXR indexing is skipped.")
    bt.add_argument("--cxr-metadata", default="/sfs/weka/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv.gz",
                    help="mimic-cxr-2.0.0-metadata.csv[.gz] for ECG↔CXR "
                         "temporal matching. If omitted, use first CXR per subject.")
    bt.add_argument("--db-path", required=True, help="Output ChromaDB directory")
    bt.add_argument("--ecg-model", default="/sfs/weka/scratch/rzv4ve/cardioagent/tools/ECG/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt", help="ECG-FM checkpoint")
    bt.add_argument("--cxr-model", default="biovilt", help="'biovilt' for BioViL-T")
    bt.add_argument("--text-model", default="/sfs/weka/scratch/rzv4ve/cardioagent/tools/Biomedbert/blobs", help="BiomedBERT HF ID")
    bt.add_argument("--batch-size", type=int, default=256)
    bt.add_argument("--max-records", type=int, default=None)
    bt.add_argument("--device", default="cuda")

    # ══════════════════════════════════════════════════════════════
    # build-symile: Symile train split (optional)
    # ══════════════════════════════════════════════════════════════
    bsym = sub.add_parser(
        "build-symile",
        help="Build index from Symile-MIMIC train split (optional RAG source)",
    )
    bsym.add_argument("--data-dir", required=True,
                      help="Symile-MIMIC root (contains data_npy/, train.csv)")
    bsym.add_argument("--db-path", required=True)
    bsym.add_argument("--ecg-model", default=None)
    bsym.add_argument("--cxr-model", default=None)
    bsym.add_argument("--text-model", default=None)
    bsym.add_argument("--batch-size", type=int, default=32)
    bsym.add_argument("--max-records", type=int, default=None)
    bsym.add_argument("--device", default="cuda")


    # build-cxr: CXR-only collection from MIMIC-CXR-JPG metadata
    bcxr = sub.add_parser(
        "build-cxr",
        help="Build CXR collection from MIMIC-CXR-JPG metadata + CheXpert labels",
    )
    bcxr.add_argument("--cxr-data-dir", default="/sfs/weka/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg-2.0.0.physionet.org",
                    help="MIMIC-CXR-JPG root (contains files/p10/p100.../)")
    bcxr.add_argument("--cxr-metadata", default="/sfs/weka/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv.gz",
                    help="mimic-cxr-2.0.0-metadata.csv.gz")
    bcxr.add_argument("--chexpert-labels", default="/sfs/weka/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz",
                    help="mimic-cxr-2.0.0-chexpert.csv.gz")
    bcxr.add_argument("--cxr-split", default="/sfs/weka/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-split.csv.gz",
                    help="mimic-cxr-2.0.0-split.csv.gz (optional, "
                        "filter to train split if provided)")
    bcxr.add_argument("--db-path", required=True)
    bcxr.add_argument("--cxr-model", default="biovilt")
    bcxr.add_argument("--batch-size", type=int, default=64)
    bcxr.add_argument("--max-records", type=int, default=None)
    bcxr.add_argument("--device", default="cuda")
    bcxr.add_argument("--frontal-only", action="store_true",
                    help="Index only PA and AP views (skip lateral)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ══════════════════════════════════════════════════════════════
    # build-train: unified index
    # ══════════════════════════════════════════════════════════════
    if args.command == "build-train":
        import pandas as pd

        # ── Load MIMIC-IV-ECG records (train split only) ──
        records = []
        with open(args.ecg_index) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("split", "train") == "train":
                        records.append(r)
                        if args.max_records and len(records) >= args.max_records:
                            break
        logger.info(f"Loaded {len(records)} train records from ECG index")

        # ── Build subject_id → CXR path lookup ──
        cxr_lookup = {}  # subject_id -> cxr_path (str)
        if args.cxr_data_dir:
            cxr_dir = Path(args.cxr_data_dir)
 
            if args.cxr_metadata and Path(args.cxr_metadata).exists():
                logger.info(f"Loading CXR metadata from {args.cxr_metadata}...")
                meta = pd.read_csv(args.cxr_metadata)
                # Parse ECG times for temporal matching
                ecg_time_by_subject = {}
                for rec in records:
                    sid = rec.get("subject_id")
                    if sid and rec.get("ecg_time"):
                        try:
                            t = pd.to_datetime(rec["ecg_time"])
                            if sid not in ecg_time_by_subject:
                                ecg_time_by_subject[sid] = t
                        except Exception:
                            pass
 
                # Parse StudyDate+StudyTime into datetime
                # StudyTime format: "213014.53100000002" = HHMMSS.frac
                if "StudyDate" in meta.columns:
                    try:
                        time_str = meta["StudyTime"].astype(str).str.split(".").str[0].str.zfill(6)
                        date_str = meta["StudyDate"].astype(str)
                        meta["study_datetime"] = pd.to_datetime(
                            date_str + " " + time_str,
                            format="%Y%m%d %H%M%S", errors="coerce",
                        )
                    except Exception as e:
                        logger.warning(f"  StudyDate/Time parsing failed: {e}")
 
                # Ensure subject_id types match
                meta["subject_id"] = meta["subject_id"].astype(int)
 
                # Prefer frontal views (PA > AP) over lateral
                view_priority = {"PA": 0, "AP": 1, "LL": 2, "LATERAL": 3}
                if "ViewPosition" in meta.columns:
                    meta["_view_rank"] = meta["ViewPosition"].map(
                        lambda v: view_priority.get(str(v).upper(), 99)
                    )
                else:
                    meta["_view_rank"] = 99
 
                n_linked = 0
                n_skipped = 0
                unique_sids = set(r.get("subject_id") for r in records)
                unique_sids.discard(None)
 
                for sid in unique_sids:
                    try:
                        sid_int = int(sid)
                    except (ValueError, TypeError):
                        continue
 
                    sub_meta = meta[meta["subject_id"] == sid_int]
                    if sub_meta.empty:
                        n_skipped += 1
                        continue
 
                    # Sort by: frontal view first, then closest to ECG time
                    sub_meta = sub_meta.sort_values("_view_rank")
 
                    target_time = ecg_time_by_subject.get(sid)
                    if target_time is not None and "study_datetime" in sub_meta.columns:
                        valid = sub_meta.dropna(subset=["study_datetime"])
                        if not valid.empty:
                            # Among frontal views, pick closest in time
                            frontal = valid[valid["_view_rank"] <= 1]
                            if frontal.empty:
                                frontal = valid
                            frontal = frontal.assign(
                                _delta=(frontal["study_datetime"] - target_time).abs()
                            ).sort_values("_delta")
                            sub_meta = frontal
 
                    # Final safety check
                    if sub_meta.empty:
                        n_skipped += 1
                        continue
 
                    row = sub_meta.iloc[0]
                    subj_str = f"p{sid_int}"
                    group = f"p{str(sid_int)[:2]}"
                    study = f"s{int(row['study_id'])}"
                    dicom = str(row.get("dicom_id", ""))
                    if dicom:
                        rel = f"files/{group}/{subj_str}/{study}/{dicom}.jpg"
                        cxr_lookup[sid] = rel
                        n_linked += 1
 
                logger.info(f"  CXR lookup built: {n_linked} linked, "
                            f"{n_skipped} subjects without CXR "
                            f"(out of {len(unique_sids)} unique subjects)")
            else:
                # Fallback: scan filesystem for first CXR per subject
                logger.info("No CXR metadata — scanning filesystem for first CXR/subject")
                for rec in records:
                    sid = rec.get("subject_id")
                    if sid is None or sid in cxr_lookup:
                        continue
                    group = f"p{str(sid)[:2]}"
                    subj_str = f"p{sid}"
                    subj_dir = cxr_dir / "files" / group / subj_str
                    if subj_dir.is_dir():
                        jpgs = list(subj_dir.rglob("*.jpg"))
                        if jpgs:
                            rel = jpgs[0].relative_to(cxr_dir)
                            cxr_lookup[sid] = str(rel)
                logger.info(f"  CXR lookup built: {len(cxr_lookup)} subjects")

        # ── Inject CXR paths into records ──
        linked_count = 0
        for rec in records:
            sid = rec.get("subject_id")
            if sid and sid in cxr_lookup:
                rec["cxr_path"] = cxr_lookup[sid]
                linked_count += 1
        if args.cxr_data_dir:
            logger.info(f"  Linked CXR to {linked_count}/{len(records)} ECG records")

        # ── Initialize embedders ──
        ecg_emb, cxr_emb, text_emb = None, None, None

        if args.ecg_model:
            from tools.embedding_tool import ECGFMEmbedder
            ecg_emb = ECGFMEmbedder(checkpoint_path=args.ecg_model,
                                    device=args.device)

        if args.cxr_model and args.cxr_data_dir:
            from tools.embedding_tool import BioViLTEmbedder
            cxr_emb = BioViLTEmbedder(device=args.device)

        if args.text_model:
            from tools.embedding_tool import BiomedBERTEmbedder
            text_emb = BiomedBERTEmbedder(checkpoint_path=args.text_model,
                                          device=args.device)

        # ── Loaders ──
        ecg_dir = Path(args.ecg_data_dir)

        def ecg_loader(rec):
            wf = rec.get("waveform_path", "")
            resolved = _resolve_ecg_path(ecg_dir, wf)
            if resolved is None:
                return None
            try:
                import wfdb
                r = wfdb.rdrecord(resolved)
                return r.p_signal.T
            except Exception as e:
                logger.debug(f"    ECG read failed for {wf}: {e}")
                return None

        cxr_loader = None
        if cxr_emb:
            cxr_dir = Path(args.cxr_data_dir)

            def cxr_loader(rec):
                cxr_path = rec.get("cxr_path", "")
                return _resolve_cxr_path(cxr_dir, cxr_path)

        # ── Build index (all modalities in one pass) ──
        logger.info("Building unified index (ECG + CXR + text in one pass)...")
        VectorCaseMemory.build_index(
            records=records,
            db_path=args.db_path,
            ecg_embedder=ecg_emb,
            cxr_embedder=cxr_emb,
            text_embedder=text_emb,
            ecg_signal_loader=ecg_loader if ecg_emb else None,
            cxr_image_loader=cxr_loader if cxr_emb else None,
            batch_size=args.batch_size,
            source_name="mimic_iv",
        )

    # ══════════════════════════════════════════════════════════════
    # build-symile: Symile train split
    # ══════════════════════════════════════════════════════════════
    elif args.command == "build-symile":
        import pandas as pd

        data_dir = Path(args.data_dir)
        npy_dir = data_dir / "data_npy" / "train"

        logger.info(f"Loading Symile-MIMIC train data from {npy_dir}...")
        ecg_data = np.load(npy_dir / "ecg_train.npy", mmap_mode="r")
        cxr_data = np.load(npy_dir / "cxr_train.npy", mmap_mode="r")
        csv = pd.read_csv(data_dir / "train.csv")
        n_total = ecg_data.shape[0]

        if args.max_records:
            n_total = min(n_total, args.max_records)
        logger.info(f"  {n_total} samples to index")

        records = []
        for i in range(n_total):
            row = csv.iloc[i]
            lab_parts = []
            for col in csv.columns:
                if col.isdigit():
                    val = row.get(col, np.nan)
                    if pd.notna(val):
                        lab_parts.append(f"{col}: {val}")
            records.append({
                "patient_id": f"symile_train_{i}",
                "source": "symile",
                "split": "train",
                "diagnosis": "",
                "report": ", ".join(lab_parts),
                "_index": i,
            })

        ecg_emb, cxr_emb, text_emb = None, None, None

        if args.ecg_model:
            from tools.embedding_tool import ECGFMEmbedder
            ecg_emb = ECGFMEmbedder(checkpoint_path=args.ecg_model,
                                    device=args.device)

        if args.cxr_model:
            from tools.embedding_tool import BioViLTEmbedder
            cxr_emb = BioViLTEmbedder(device=args.device)

        if args.text_model:
            from tools.embedding_tool import BiomedBERTEmbedder
            text_emb = BiomedBERTEmbedder(checkpoint_path=args.text_model,
                                          device=args.device)

        def ecg_loader(rec):
            idx = rec["_index"]
            return ecg_data[idx].squeeze(0).T.astype(np.float32)

        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def cxr_loader(rec):
            from PIL import Image
            idx = rec["_index"]
            cxr = cxr_data[idx].copy()
            for c in range(3):
                cxr[c] = cxr[c] * STD[c] + MEAN[c]
            cxr = np.transpose(cxr, (1, 2, 0))
            cxr = np.clip(cxr * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(cxr)

        VectorCaseMemory.build_index(
            records=records,
            db_path=args.db_path,
            ecg_embedder=ecg_emb,
            cxr_embedder=cxr_emb,
            text_embedder=text_emb,
            ecg_signal_loader=ecg_loader if ecg_emb else None,
            cxr_image_loader=cxr_loader if cxr_emb else None,
            batch_size=args.batch_size,
            source_name="symile",
        )
    
    elif args.command == "build-cxr":
        import pandas as pd
        
        # Load metadata
        logger.info(f"Loading metadata: {args.cxr_metadata}")
        meta = pd.read_csv(args.cxr_metadata)
        logger.info(f"  {len(meta)} dicom records")

        # Filter to frontal views if requested
        if args.frontal_only:
            before = len(meta)
            meta = meta[meta["ViewPosition"].isin(["PA", "AP"])]
            logger.info(f"  Frontal filter: {len(meta)}/{before} kept")

        # Filter to train split if provided
        if args.cxr_split and Path(args.cxr_split).exists():
            split_df = pd.read_csv(args.cxr_split)
            train_dicoms = set(split_df[split_df["split"] == "train"]["dicom_id"])
            before = len(meta)
            meta = meta[meta["dicom_id"].isin(train_dicoms)]
            logger.info(f"  Train split filter: {len(meta)}/{before} kept")

        # Load CheXpert labels
        logger.info(f"Loading CheXpert labels: {args.chexpert_labels}")
        chex = pd.read_csv(args.chexpert_labels)
        chex["subject_id"] = chex["subject_id"].astype(int)
        chex["study_id"] = chex["study_id"].astype(int)

        # Build study-level lookup: (subject_id, study_id) -> label dict
        label_cols = [c for c in chex.columns
                    if c not in ("subject_id", "study_id")]
        chex_lookup = {}
        for _, row in chex.iterrows():
            key = (int(row["subject_id"]), int(row["study_id"]))
            positives = []
            negatives = []
            for col in label_cols:
                v = row[col]
                if pd.notna(v):
                    if v == 1.0:
                        positives.append(col)
                    elif v == 0.0:
                        negatives.append(col)
            chex_lookup[key] = {
                "positives": positives,
                "negatives": negatives,
            }
        logger.info(f"  CheXpert lookup: {len(chex_lookup)} studies")

        # Cap records if requested
        if args.max_records:
            meta = meta.head(args.max_records)

        # Build records list
        records = []
        n_with_labels = 0
        for _, row in meta.iterrows():
            sid = int(row["subject_id"])
            stid = int(row["study_id"])
            dicom = str(row["dicom_id"])
            view = str(row.get("ViewPosition", "?"))
            
            group = f"p{str(sid)[:2]}"
            cxr_rel = f"files/{group}/p{sid}/s{stid}/{dicom}.jpg"
            
            # Get CheXpert labels for this study
            labels = chex_lookup.get((sid, stid), {"positives": [], "negatives": []})
            if labels["positives"]:
                n_with_labels += 1
            
            # Diagnosis = comma-joined positive findings (or "no finding")
            diagnosis = ", ".join(labels["positives"]) if labels["positives"] else "No finding"
            
            # Report = positive findings + negative findings as natural language
            report_parts = []
            if labels["positives"]:
                report_parts.append(
                    f"Positive findings: {', '.join(labels['positives'])}"
                )
            if labels["negatives"]:
                report_parts.append(
                    f"Negated: {', '.join(labels['negatives'][:5])}"
                )
            report_parts.append(f"View: {view}")
            report = " | ".join(report_parts)
            
            records.append({
                "patient_id": f"mimic_cxr_{sid}_{stid}_{dicom[:8]}",
                "subject_id": sid,
                "study_id": stid,
                "dicom_id": dicom,
                "source": "mimic_cxr",
                "split": "train",
                "diagnosis": diagnosis[:200],
                "report": report[:300],
                "view": view,
                "cxr_path": cxr_rel,
            })
        
        logger.info(f"  Built {len(records)} records ({n_with_labels} with positive labels)")

        # Initialize CXR embedder
        from tools.embedding_tool import BioViLTEmbedder
        cxr_emb = BioViLTEmbedder(device=args.device)

        cxr_dir = Path(args.cxr_data_dir)
        def cxr_loader(rec):
            return _resolve_cxr_path(cxr_dir, rec["cxr_path"])

        # Build only the CXR collection
        VectorCaseMemory.build_index(
            records=records,
            db_path=args.db_path,
            ecg_embedder=None,
            cxr_embedder=cxr_emb,
            text_embedder=None,
            ecg_signal_loader=None,
            cxr_image_loader=cxr_loader,
            batch_size=args.batch_size,
            source_name="mimic_cxr",
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()