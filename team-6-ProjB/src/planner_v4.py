"""
CardioAgent Planner v4 — Dual-Pathway Retrieval
==================================================
Extends Planner v3 with embedding-based vector retrieval running
IN PARALLEL with the diagnostic tool pathway.

Changes from v3:
  - VectorCaseMemory replaces/supplements keyword CaseMemory
  - Embedding models (ECG-FM, BioViL-T, BiomedBERT) run alongside
    diagnostic tools (ECGFounder, LingShu, ECGTool)
  - Late fusion via Reciprocal Rank Fusion (RRF)
  - New ablation setting D: DeepRare + Vector RAG

The diagnostic pathway is UNCHANGED from v3 — this only adds
the parallel retrieval pathway and modifies Phase 4 (RAG retrieval).

Usage:
    # Ablation D: DeepRare + Vector RAG
    planner = CardioAgentPlannerV4(
        qwen_api_url=url,
        lingshu_api_url=url,
        vector_db_path="/path/to/chroma_db",
        ecg_embedder=ecg_fm_embedder,
        cxr_embedder=biovil_t_embedder,
        text_embedder=biomed_bert_embedder,
        enable_reflection=True,
    )

    # Still supports keyword RAG (ablation C) via rag_index_paths
    # Still supports no RAG (ablation B) via omitting both
"""

import time
import logging
import numpy as np
from typing import Optional, List

logger = logging.getLogger(__name__)

# Import everything from planner_v3 (the document 2 code)
# On Rivanna this file is planner_v2.py
from planner_v3 import (
    CardioAgentPlannerV3,
    PatientData,
    AgentStateV3,
    ThinkingStep,
    PhenotypeExtractor,
    HypothesisGenerator,
    ReflectionEngine,
    CaseMemory,
    EvidenceLink,
)


class CardioAgentPlannerV4(CardioAgentPlannerV3):
    """
    Extends v3 with a parallel vector retrieval pathway.

    Inherits ALL diagnostic tool execution, phenotype extraction,
    hypothesis generation, reflection, and synthesis from v3.

    Only overrides Phase 4 (RAG retrieval) to add vector-based retrieval.
    """

    def __init__(
        self,
        # -- v3 params (passed through) --
        qwen_api_url: str = "http://localhost:8000/v1",
        qwen_model_name: str = "qwen3-vl",
        lingshu_api_url: str = "http://localhost:8001/v1",
        on_step_update=None,
        enable_reflection: bool = True,
        max_reflection_rounds: int = 2,
        rag_index_paths: List[str] = None,
        rag_top_k: int = 5,
        # -- v4 new params: vector retrieval --
        vector_db_path: Optional[str] = None,
        ecg_embedder=None,
        cxr_embedder=None,
        text_embedder=None,
        fusion_strategy: str = "rrf",
        vector_top_k: int = 5,
        vector_per_modality_k: int = 10,
    ):
        # Initialize v3 (diagnostic tools, phenotype, hypothesis, reflection)
        super().__init__(
            qwen_api_url=qwen_api_url,
            qwen_model_name=qwen_model_name,
            lingshu_api_url=lingshu_api_url,
            on_step_update=on_step_update,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            rag_index_paths=rag_index_paths,
            rag_top_k=rag_top_k,
        )

        # -- Vector retrieval pathway --
        self.vector_memory = None
        self.vector_top_k = vector_top_k
        self.vector_per_modality_k = vector_per_modality_k

        # Store embedders for on-the-fly embedding at inference time
        self._ecg_embedder = ecg_embedder
        self._cxr_embedder = cxr_embedder
        self._text_embedder = text_embedder

        if vector_db_path:
            from vec_memory import VectorCaseMemory
            self.vector_memory = VectorCaseMemory(
                db_path=vector_db_path,
                ecg_embedder=ecg_embedder,
                cxr_embedder=cxr_embedder,
                text_embedder=text_embedder,
                fusion_strategy=fusion_strategy,
            )
            logger.info(f"  PlannerV4: vector memory enabled "
                        f"(strategy={fusion_strategy})")

    def run(self, data: Optional[PatientData] = None, **kwargs) -> AgentStateV3:
        """
        Full pipeline — same as v3 but Phase 4 uses vector retrieval.

        Pipeline:
          1. Tool execution (ECG, CXR, Labs, Notes)     — UNCHANGED
          2. Phenotype extraction                         — UNCHANGED
          3. Hypothesis generation                        — UNCHANGED
          4. RAG retrieval (VECTOR-BASED)                 — NEW
          5. Reflection loop                              — UNCHANGED
          6. Final synthesis                              — UNCHANGED
        """
        if data is None:
            data = self._build_patient_data(**kwargs)

        state = AgentStateV3(patient_id=data.patient_id, started_at=time.time())

        # -- Phase 1: Detect + Execute modality tools --
        available = self._detect_modalities(data)
        state.modalities_available = available

        step = ThinkingStep(
            step_name="planning", status="success",
            detail=self._build_plan_v4(data, available),
        )
        self._notify(step, state)

        if not available:
            state.final_report = "No data available for analysis."
            state.finished_at = time.time()
            return state

        if "ecg" in available:
            self._run_ecg(data, state)
        if "image" in available:
            self._run_image(data, state)
        if "lab" in available:
            self._run_labs(data, state)
        if "notes" in available:
            self._run_notes(data, state)

        # -- Phase 2: Phenotype extraction --
        t0 = time.time()
        phenotypes = self.phenotype_extractor.extract(state)
        state.phenotype_terms = phenotypes

        step = ThinkingStep(
            step_name="phenotype_extraction", status="success",
            detail=f"Extracted {len(phenotypes)} phenotype terms: "
                   f"{', '.join(p['term'] for p in phenotypes[:5])}"
                   f"{'...' if len(phenotypes) > 5 else ''}",
            duration_s=round(time.time() - t0, 2),
            output_data={"phenotypes": phenotypes},
        )
        self._notify(step, state)

        # -- Phase 3: Hypothesis generation --
        t0 = time.time()
        hypotheses = self.hypothesis_generator.generate(phenotypes, state)
        state.hypothesis_ranking = hypotheses

        step = ThinkingStep(
            step_name="hypothesis_generation", status="success",
            detail=f"{len(hypotheses)} hypotheses: "
                   + ", ".join(f"{h.diagnosis} ({h.confidence:.2f})"
                               for h in hypotheses[:3]),
            duration_s=round(time.time() - t0, 2),
        )
        self._notify(step, state)

        # --------------------------------------------------------------
        # Phase 4: RAG retrieval — VECTOR-BASED (v4 override)
        # --------------------------------------------------------------
        rag_cases = self._run_vector_retrieval(data, state)

        # Fallback to keyword CaseMemory if vector retrieval unavailable
        if not rag_cases and self.case_memory.enabled:
            rag_cases = self._run_keyword_retrieval(state, phenotypes)

        state.rag_cases = [
            c.to_dict() if hasattr(c, "to_dict") else c
            for c in rag_cases
        ]

        # -- Phase 5: Reflection loop --
        if self.enable_reflection and hypotheses:
            t0 = time.time()
            # Convert RetrievedCase ? dict for reflection engine compatibility
            rag_dicts = state.rag_cases
            hypotheses, consistency, n_rounds, ref_log = \
                self.reflection_engine.run(hypotheses, phenotypes, rag_dicts)

            state.hypothesis_ranking = hypotheses
            state.cross_modal_consistency = consistency
            state.reflection_rounds = n_rounds
            state.reflection_log = ref_log

            step = ThinkingStep(
                step_name="reflection", status="success",
                detail=f"{n_rounds} rounds, "
                       f"{consistency.get('n_agreements', 0)} agreements, "
                       f"{consistency.get('n_contradictions', 0)} contradictions. "
                       f"Top: {hypotheses[0].diagnosis} ({hypotheses[0].confidence:.2f})"
                       if hypotheses else "No hypotheses",
                duration_s=round(time.time() - t0, 2),
            )
            self._notify(step, state)
        else:
            step = ThinkingStep(
                step_name="reflection", status="skipped",
                detail="Reflection disabled" if not self.enable_reflection
                       else "No hypotheses generated",
            )
            self._notify(step, state)

        # -- Phase 6: Final synthesis --
        self._run_synthesis(data, state)
        state.finished_at = time.time()
        return state

    # -------------------------------------------------------------------
    # Vector Retrieval (new in v4)
    # -------------------------------------------------------------------

    def _run_vector_retrieval(self, data: PatientData, state: AgentStateV3) -> list:
        """
        Run embedding-based vector retrieval across available modalities.
        Returns list of RetrievedCase objects.
        """
        if not self.vector_memory or not self.vector_memory.enabled:
            step = ThinkingStep(
                step_name="vector_retrieval", status="skipped",
                detail="Vector memory not enabled",
            )
            self._notify(step, state)
            return []

        t0 = time.time()
        step = ThinkingStep(
            step_name="vector_retrieval", status="running",
            detail="Querying vector memory (ECG-FM + BioViL-T + BiomedBERT)...",
        )
        self._notify(step, state)

        try:
            # Prepare inputs for each modality
            ecg_signal = None
            if data.ecg_path:
                try:
                    ecg_signal = np.load(data.ecg_path)
                except Exception:
                    try:
                        import wfdb
                        from pathlib import Path
                        rec = wfdb.rdrecord(str(Path(data.ecg_path).with_suffix("")))
                        ecg_signal = rec.p_signal.T
                    except Exception:
                        pass

            cxr_image = data.image_path if data.image_type == "cxr" else None

            clinical_text = None
            if state.lab_result:
                clinical_text = state.lab_result.get("text_summary", "")
            if state.clinical_notes:
                clinical_text = (clinical_text or "") + " " + state.clinical_notes

            # Query vector memory
            cases = self.vector_memory.retrieve(
                ecg_signal=ecg_signal,
                cxr_image=cxr_image,
                clinical_text=clinical_text if clinical_text and len(clinical_text.strip()) > 20 else None,
                top_k=self.vector_top_k,
                per_modality_k=self.vector_per_modality_k,
            )

            # Log modality breakdown
            modalities_queried = set()
            for c in cases:
                modalities_queried.update(c.modalities_matched)

            step = ThinkingStep(
                step_name="vector_retrieval", status="success",
                detail=f"Retrieved {len(cases)} cases via {sorted(modalities_queried)} "
                       f"(top: {cases[0].diagnosis[:40]}... score={cases[0].score:.3f})"
                       if cases else "No cases retrieved",
                duration_s=round(time.time() - t0, 2),
                output_data={
                    "n_cases": len(cases),
                    "modalities_queried": sorted(modalities_queried),
                    "top_scores": [round(c.score, 4) for c in cases[:3]],
                },
            )
            self._notify(step, state)
            return cases

        except Exception as e:
            step = ThinkingStep(
                step_name="vector_retrieval", status="error",
                detail=f"Vector retrieval failed: {e}",
                duration_s=round(time.time() - t0, 2),
            )
            self._notify(step, state)
            logger.error(f"Vector retrieval error: {e}", exc_info=True)
            return []

    def _run_keyword_retrieval(self, state: AgentStateV3, phenotypes: list) -> list:
        """Fallback: keyword CaseMemory retrieval (same as v3 Phase 4)."""
        t0 = time.time()
        cases = self.case_memory.retrieve(
            phenotypes, state, top_k=self.rag_top_k,
        )
        step = ThinkingStep(
            step_name="keyword_retrieval", status="success",
            detail=f"Keyword fallback: {len(cases)} cases retrieved",
            duration_s=round(time.time() - t0, 2),
        )
        self._notify(step, state)
        return cases

    # -------------------------------------------------------------------
    # Plan description (updated for v4)
    # -------------------------------------------------------------------

    def _build_plan_v4(self, data: PatientData, available: list) -> str:
        all_mods = {"ecg": "ECG", "image": "Medical Image",
                    "lab": "Lab Results", "notes": "Clinical Notes"}
        present = [all_mods[m] for m in available]
        missing = [v for k, v in all_mods.items() if k not in available]

        plan = f"Patient {data.patient_id}: "
        plan += f"{len(available)} modalities ({', '.join(present)})"
        if missing:
            plan += f", missing ({', '.join(missing)})"
        plan += ".\nPipeline: Tools ? Phenotypes ? Hypotheses"

        if self.vector_memory and self.vector_memory.enabled:
            plan += " ? Vector RAG (RRF fusion)"
        elif self.case_memory.enabled:
            plan += " ? Keyword RAG"

        if self.enable_reflection:
            plan += " ? Reflection"
        plan += " ? Synthesis"
        return plan