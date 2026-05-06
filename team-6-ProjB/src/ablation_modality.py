"""
Ablation D -- DeepRare Pipeline + Vector RAG (Embedding-Based Retrieval)
=========================================================================
Supports granular modality-level ablation sub-settings:

  D1: ECG retrieval only     (ECG-FM)
  D2: CXR retrieval only     (BioViL-T)
  D3: Text retrieval only    (BiomedBERT)
  D4: All modalities + RRF   (no reranker)
  D5: All modalities + RRF + MedCPT reranker

  D-rand: Random cases injected (counterfactual control)
  D-shuf: Shuffled embeddings   (counterfactual control)

Usage:
    # Single-modality ablation
    python ablation_d.py --retrieval-mode D1 --max-samples 200

    # Full fusion
    python ablation_d.py --retrieval-mode D4 --max-samples 200

    # All sub-settings sequentially
    python ablation_d.py --retrieval-mode all --max-samples 200

    # Counterfactual
    python ablation_d.py --retrieval-mode D-rand --max-samples 200
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))


# ===================================================================
# Default Paths (Rivanna)
# ===================================================================

QWEN_URL = "http://localhost:8000/v1"
LINGSHU_URL = "http://localhost:8002/v1"
VECTOR_DB_PATH = "/scratch/rzv4ve/cardioagent/vector_db_multi"
ECG_FM_CHECKPOINT = "/sfs/weka/scratch/rzv4ve/cardioagent/tools/ECG/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
BIOMEDBERT_CHECKPOINT = "/scratch/rzv4ve/cardioagent/tools/Biomedbert/blobs"
MEDCPT_CHECKPOINT = "ncbi/MedCPT-Cross-Encoder"


# ===================================================================
# Retrieval Mode Definitions
# ===================================================================

RETRIEVAL_MODES = {
    "D1": {
        "name": "D1_ecg_only",
        "description": "DeepRare + ECG-FM retrieval only",
        "ecg": True, "cxr": False, "text": False, "rerank": False,
    },
    "D2": {
        "name": "D2_cxr_only",
        "description": "DeepRare + BioViL-T retrieval only",
        "ecg": False, "cxr": True, "text": False, "rerank": False,
    },
    "D3": {
        "name": "D3_text_only",
        "description": "DeepRare + BiomedBERT retrieval only",
        "ecg": False, "cxr": False, "text": True, "rerank": False,
    },
    "D4": {
        "name": "D4_all_rrf",
        "description": "DeepRare + All modalities + RRF fusion (no rerank)",
        "ecg": True, "cxr": True, "text": True, "rerank": False,
    },
    "D5": {
        "name": "D5_all_rrf_rerank",
        "description": "DeepRare + All modalities + RRF + MedCPT rerank",
        "ecg": True, "cxr": True, "text": True, "rerank": True,
    },
    "D-rand": {
        "name": "D_rand_control",
        "description": "DeepRare + Random cases injected (counterfactual)",
        "ecg": False, "cxr": False, "text": False, "rerank": False,
        "counterfactual": "random",
    },
    "D-shuf": {
        "name": "D_shuf_control",
        "description": "DeepRare + Shuffled embedding retrieval (counterfactual)",
        "ecg": True, "cxr": True, "text": True, "rerank": False,
        "counterfactual": "shuffle",
    },
}

ALL_MODES = ["D1", "D2", "D3", "D4", "D5"]
ALL_MODES_WITH_CONTROLS = ALL_MODES + ["D-rand", "D-shuf"]


class AblationD:
    """
    DeepRare pipeline + Vector RAG with configurable modality toggles.

    The retrieval_mode parameter controls which embedding models are
    active for retrieval. The diagnostic tool pathway (ECG Tool, LingShu)
    is ALWAYS active regardless of retrieval mode.
    """

    def __init__(
        self,
        retrieval_mode: str = "D4",
        qwen_url: str = QWEN_URL,
        lingshu_url: str = LINGSHU_URL,
        vector_db_path: str = VECTOR_DB_PATH,
        ecg_fm_checkpoint: str = ECG_FM_CHECKPOINT,
        biomedbert_checkpoint: str = BIOMEDBERT_CHECKPOINT,
        medcpt_checkpoint: str = MEDCPT_CHECKPOINT,
        device: str = "cuda",
        vector_top_k: int = 5,
        fusion_strategy: str = "rrf",
    ):
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"Unknown retrieval_mode '{retrieval_mode}'. "
                f"Options: {list(RETRIEVAL_MODES.keys())}"
            )

        self.mode_config = RETRIEVAL_MODES[retrieval_mode]
        self.retrieval_mode = retrieval_mode
        self.name = self.mode_config["name"]
        self.description = self.mode_config["description"]
        self.counterfactual = self.mode_config.get("counterfactual")

        logger.info(f"AblationD initializing: {retrieval_mode} -- {self.description}")

        # -- Initialize embedders based on mode config --
        ecg_embedder = None
        cxr_embedder = None
        text_embedder = None

        if self.mode_config["ecg"]:
            try:
                from embedding_tools import ECGFMEmbedder
                ecg_embedder = ECGFMEmbedder(
                    checkpoint_path=ecg_fm_checkpoint,
                    device=device,
                    strict_loading=False,
                )
                logger.info("  ECG-FM embedder initialized")
            except Exception as e:
                logger.warning(f"  ECG-FM unavailable: {e}")

        if self.mode_config["cxr"]:
            try:
                from embedding_tools import BioViLTEmbedder
                cxr_embedder = BioViLTEmbedder(device=device)
                logger.info("  BioViL-T embedder initialized")
            except Exception as e:
                logger.warning(f"  BioViL-T unavailable: {e}")

        if self.mode_config["text"]:
            try:
                from embedding_tools import BiomedBERTEmbedder
                text_embedder = BiomedBERTEmbedder(
                    checkpoint_path=biomedbert_checkpoint,
                    device=device,
                )
                logger.info("  BiomedBERT embedder initialized")
            except Exception as e:
                logger.warning(f"  BiomedBERT unavailable: {e}")

        # -- Initialize reranker (D5 only) --
        reranker = None
        if self.mode_config["rerank"]:
            try:
                from reranker import MedCPTReranker
                reranker = MedCPTReranker(
                    checkpoint_path=medcpt_checkpoint,
                    device=device,
                )
                logger.info("  MedCPT reranker initialized")
            except Exception as e:
                logger.warning(f"  MedCPT unavailable: {e}")

        # -- Load random case pool for counterfactual D-rand --
        self._random_case_pool = None
        if self.counterfactual == "random":
            self._random_case_pool = self._load_random_pool(vector_db_path)

        # -- Initialize planner v4 --
        use_vector_db = (
            vector_db_path
            and self.counterfactual != "random"
            and any([ecg_embedder, cxr_embedder, text_embedder])
        )

        from planner_v4 import CardioAgentPlannerV4, PatientData
        self.planner = CardioAgentPlannerV4(
            qwen_api_url=qwen_url,
            lingshu_api_url=lingshu_url,
            rag_index_paths=None,
            enable_reflection=True,
            max_reflection_rounds=2,
            vector_db_path=vector_db_path if use_vector_db else None,
            ecg_embedder=ecg_embedder,
            cxr_embedder=cxr_embedder,
            text_embedder=text_embedder,
            fusion_strategy=fusion_strategy,
            vector_top_k=vector_top_k,
        )

        # Inject reranker if D5
        if reranker and self.planner.vector_memory:
            self.planner.vector_memory.reranker = reranker

        self.PatientData = PatientData
        self._vector_db_path = vector_db_path

        active = []
        if ecg_embedder: active.append("ECG-FM")
        if cxr_embedder: active.append("BioViL-T")
        if text_embedder: active.append("BiomedBERT")
        if reranker: active.append("MedCPT")
        logger.info(f"  Active retrieval: {active or ['none (counterfactual)']}")

    def _load_random_pool(self, db_path: str) -> List[dict]:
        """Load case metadata from ChromaDB for random sampling."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=db_path)
            pool = []
            for name in ["ecg_embeddings", "text_embeddings", "cxr_embeddings"]:
                try:
                    col = client.get_collection(name)
                    result = col.get(limit=1000, include=["metadatas"])
                    for cid, meta in zip(result["ids"], result["metadatas"]):
                        pool.append({
                            "patient_id": cid,
                            "source": meta.get("source", "unknown"),
                            "diagnosis": meta.get("diagnosis", ""),
                            "report": meta.get("report", ""),
                            "score": round(random.random() * 0.3 + 0.1, 4),
                            "modalities_matched": [
                                name.replace("_embeddings", "")
                            ],
                            "modality_scores": {
                                name.replace("_embeddings", ""): round(
                                    random.random() * 0.3 + 0.1, 4
                                )
                            },
                        })
                except Exception:
                    pass
            logger.info(f"  Random case pool: {len(pool)} cases loaded")
            return pool
        except Exception as e:
            logger.warning(f"  Failed to load random pool: {e}")
            return []

    def _get_random_cases(self, top_k: int = 5) -> list:
        """Return random cases from the pool (counterfactual D-rand)."""
        if not self._random_case_pool:
            return []
        return random.sample(
            self._random_case_pool,
            min(top_k, len(self._random_case_pool)),
        )

    def _extract_metadata(self, state, extra: dict = None) -> dict:
        """Extract diagnostic metadata from planner state."""
        meta = {
            "retrieval_mode": self.retrieval_mode,
            "mode_name": self.name,
            "hypothesis_ranking": [
                {"rank": h.rank, "dx": h.diagnosis,
                 "conf": round(h.confidence, 4)}
                for h in state.hypothesis_ranking[:5]
            ],
            "phenotypes": [p["term"] for p in state.phenotype_terms],
            "reflection_rounds": state.reflection_rounds,
            "n_rag_cases": len(state.rag_cases),
            "cross_modal_agreements": state.cross_modal_consistency.get(
                "n_agreements", 0),
            "cross_modal_contradictions": state.cross_modal_consistency.get(
                "n_contradictions", 0),
        }

        if state.rag_cases:
            all_modalities = set()
            meta["rag_case_details"] = []
            for c in state.rag_cases[:5]:
                if isinstance(c, dict):
                    meta["rag_case_details"].append({
                        "id": c.get("patient_id", "?"),
                        "dx": c.get("diagnosis", "")[:60],
                        "score": c.get("score", 0),
                        "modalities": c.get("modalities_matched", []),
                    })
                    all_modalities.update(c.get("modalities_matched", []))
            meta["retrieval_modalities"] = sorted(all_modalities)

        if extra:
            meta.update(extra)
        return meta

    def predict_symile(self, data, idx: int, cache_dir: str) -> dict:
        """Predict on Symile-MIMIC sample."""
        from eval_v5 import PATHOLOGIES, JSON_OUTPUT_INSTRUCTION, extract_predictions

        ecg = data.get_ecg(idx)
        ecg_path = os.path.join(cache_dir, f"ecg_eval_{idx}.npy")
        np.save(ecg_path, ecg)
        cxr_path, cxr_source = data.get_cxr_best_path(idx, cache_dir)
        lab_dict = data.get_lab_dict(idx)
        lab_text = data.get_lab_text(idx)

        # -- Counterfactual: random cases --
        if self.counterfactual == "random":
            state = self.planner.run(self.PatientData(
                patient_id=f"eval_{idx}",
                ecg_path=ecg_path,
                image_path=cxr_path,
                image_type="cxr",
                lab_results=lab_dict if lab_dict else None,
                lab_text=lab_text if not lab_dict else None,
                clinical_notes=JSON_OUTPUT_INSTRUCTION,
            ))
            state.rag_cases = self._get_random_cases(5)

        # -- Counterfactual: shuffled embeddings --
        elif self.counterfactual == "shuffle":
            shuf_idx = (idx + 137) % data.n_samples
            shuf_ecg = data.get_ecg(shuf_idx)
            shuf_ecg_path = os.path.join(cache_dir, f"ecg_shuf_{idx}.npy")
            np.save(shuf_ecg_path, shuf_ecg)

            state = self.planner.run(self.PatientData(
                patient_id=f"eval_{idx}",
                ecg_path=shuf_ecg_path,
                image_path=cxr_path,
                image_type="cxr",
                lab_results=lab_dict if lab_dict else None,
                lab_text=lab_text if not lab_dict else None,
                clinical_notes=JSON_OUTPUT_INSTRUCTION,
            ))

        # -- Normal retrieval (D1-D5) --
        else:
            state = self.planner.run(self.PatientData(
                patient_id=f"eval_{idx}",
                ecg_path=ecg_path,
                image_path=cxr_path,
                image_type="cxr",
                lab_results=lab_dict if lab_dict else None,
                lab_text=lab_text if not lab_dict else None,
                clinical_notes=JSON_OUTPUT_INSTRUCTION,
            ))

        preds = extract_predictions(state.final_report, PATHOLOGIES)
        result = {
            "predictions": preds,
            "raw_text": state.final_report,
            "cxr_source": cxr_source,
            "elapsed_s": state.elapsed_s,
        }
        result.update(self._extract_metadata(
            state, {"counterfactual": self.counterfactual}
        ))
        return result

    def predict_mimic4(self, test_data, idx: int, cache_dir: str) -> dict:
        """Predict on MIMIC-IV-ECG sample."""
        from eval_v5 import (
            PATHOLOGIES, ECG_DIAGNOSES, JSON_OUTPUT_INSTRUCTION,
            extract_predictions,
        )

        ecg_path = test_data.get_ecg_path(idx)
        record = test_data.get_record(idx)

        state = self.planner.run(self.PatientData(
            patient_id=record.get("patient_id", f"mimic4_{idx}"),
            ecg_path=ecg_path,
            clinical_notes=(
                f"Machine ECG report: {record.get('machine_report', '')}\n"
                + JSON_OUTPUT_INSTRUCTION
            ),
        ))

        result = {
            "predictions": extract_predictions(state.final_report, PATHOLOGIES),
            "ecg_predictions": extract_predictions(
                state.final_report, ECG_DIAGNOSES
            ),
            "raw_text": state.final_report,
            "elapsed_s": state.elapsed_s,
        }
        result.update(self._extract_metadata(state))
        return result


# ===================================================================
# Standalone Runner
# ===================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Ablation D -- Vector RAG with modality sub-settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Retrieval modes:
  D1       ECG-FM retrieval only
  D2       BioViL-T (CXR) retrieval only
  D3       BiomedBERT (text) retrieval only
  D4       All modalities + RRF fusion
  D5       All modalities + RRF + MedCPT reranker
  D-rand   Random cases injected (counterfactual control)
  D-shuf   Shuffled embeddings (counterfactual control)
  all      Run D1, D2, D3, D4, D5 sequentially
  full     Run all including counterfactuals
        """,
    )
    parser.add_argument(
        "--retrieval-mode", default="D4",
        choices=list(RETRIEVAL_MODES.keys()) + ["all", "full"],
    )
    parser.add_argument("--data-source", default="symile",
                        choices=["symile", "mimic4ecg", "both"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--data-dir-symile",
                        default="/scratch/rzv4ve/cardioagent/data/symile-mimic")
    parser.add_argument("--data-dir-mimic4",
                        default="/scratch/rzv4ve/cardioagent/data/MIMIC-IV-ECG-1")
    parser.add_argument("--vector-db-path", default=VECTOR_DB_PATH)
    parser.add_argument("--qwen-url", default=QWEN_URL)
    parser.add_argument("--lingshu-url", default=LINGSHU_URL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir",
                        default="/scratch/rzv4ve/cardioagent/ablation_results_v5_cxr_rag")
    args = parser.parse_args()

    from eval_v5 import SymileTestData, MIMIC4ECGTestData, run_ablation

    os.makedirs(args.output_dir, exist_ok=True)

    if args.retrieval_mode == "all":
        modes = ALL_MODES
    elif args.retrieval_mode == "full":
        modes = ALL_MODES_WITH_CONTROLS
    else:
        modes = [args.retrieval_mode]

    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Running Ablation {mode}")
        logger.info(f"{'='*60}\n")

        runner = AblationD(
            retrieval_mode=mode,
            qwen_url=args.qwen_url,
            lingshu_url=args.lingshu_url,
            vector_db_path=args.vector_db_path,
            device=args.device,
        )

        if args.data_source in ["symile", "both"]:
            sdata = SymileTestData(args.data_dir_symile)
            run_ablation(
                runner, sdata, args.max_samples,
                args.output_dir, f"{runner.name}_symile", "symile",
            )

        if args.data_source in ["mimic4ecg", "both"]:
            db_dir = "/scratch/rzv4ve/cardioagent/results_db"
            idx_path = os.path.join(db_dir, "mimic_iv_ecg_index.jsonl")
            if os.path.exists(idx_path):
                mdata = MIMIC4ECGTestData(
                    args.data_dir_mimic4, idx_path, args.max_samples,
                )
                run_ablation(
                    runner, mdata, args.max_samples,
                    args.output_dir, f"{runner.name}_mimic4ecg", "mimic4ecg",
                )

        logger.info(f"  Ablation {mode} complete.\n")