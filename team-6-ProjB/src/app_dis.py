"""
CardioAgent Streamlit App
============================
Transparent clinical decision-support interface for CardioAgent.

Features:
  1. Live DST workflow visualization -- each phase streams with status
  2. RAG transparency -- shows retrieved cases, per-modality scores, rerank
  3. Diagnostic reasoning panel -- phenotypes, hypotheses, reflection rounds
  4. Conversational refinement -- follow-up dialogue with Qwen about the case
  5. File upload for patient data (ECG .hea/.npy, CXR .jpg/.png, lab JSON/CSV)

Run:
    streamlit run app.py --server.port 8501

The app assumes the agent stack is running:
  - Qwen3-VL on port 8000
  - LingShu on port 8001
  - ChromaDB vector_db built at configured path
"""

import os
import sys
import time
import json
import base64
import logging
import tempfile
import threading
from pathlib import Path
from queue import Queue, Empty
from dataclasses import asdict
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np


# --- Path setup: import from parent codebase -------------------------
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "tools"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===================================================================
# CXR Region Map (for LingShu textual location -> bounding box)
# ===================================================================

CXR_REGIONS = {
    "left upper lobe":           (0.55, 0.05, 0.95, 0.40),
    "left lower lobe":           (0.55, 0.40, 0.95, 0.85),
    "right upper lobe":          (0.05, 0.05, 0.45, 0.40),
    "right lower lobe":          (0.05, 0.40, 0.45, 0.85),
    "left lung":                 (0.55, 0.05, 0.95, 0.85),
    "right lung":                (0.05, 0.05, 0.45, 0.85),
    "cardiac silhouette":        (0.30, 0.35, 0.70, 0.78),
    "heart":                     (0.30, 0.35, 0.70, 0.78),
    "mediastinum":               (0.40, 0.05, 0.60, 0.65),
    "left hilum":                (0.55, 0.30, 0.75, 0.55),
    "right hilum":               (0.25, 0.30, 0.45, 0.55),
    "costophrenic angle left":   (0.70, 0.75, 0.95, 0.95),
    "costophrenic angle right":  (0.05, 0.75, 0.30, 0.95),
    "diaphragm":                 (0.10, 0.72, 0.90, 0.90),
    "trachea":                   (0.45, 0.05, 0.55, 0.30),
    "aortic arch":               (0.40, 0.20, 0.60, 0.40),
}


# 
# Configuration
# 

DEFAULT_CONFIG = {
    "qwen_url": "http://localhost:8000/v1",
    "lingshu_url": "http://localhost:8002/v1",
    "vector_db_path": "/scratch/rzv4ve/cardioagent/vector_db_multi",
    "ecg_fm_path": "/scratch/rzv4ve/cardioagent/tools/ECG/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt",
    "upload_dir": "/scratch/rzv4ve/cardioagent/cardioagent_uploads",
}

# 
# Streamlit Page Setup
# 

st.set_page_config(
    page_title="CardioAgent -- DST Workflow",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS -- clinical, editorial, readable
st.markdown("""
<style>
    :root {
        --ca-primary: #1a3a5c;
        --ca-accent: #c44536;
        --ca-success: #2d6a4f;
        --ca-warning: #b45309;
        --ca-bg-soft: #f4f1ea;
        --ca-border: rgba(26, 58, 92, 0.15);
    }
    .main .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
    }
    h1, h2, h3 {
        font-family: 'Georgia', 'Charter', serif !important;
        letter-spacing: -0.01em;
    }
    .phase-card {
        background: var(--ca-bg-soft);
        border-left: 3px solid var(--ca-primary);
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 2px;
        font-family: 'SF Mono', 'Menlo', monospace;
        font-size: 0.85rem;
    }
    .phase-success { border-left-color: var(--ca-success); }
    .phase-running { border-left-color: var(--ca-warning); }
    .phase-error   { border-left-color: var(--ca-accent); }
    .phase-skipped { border-left-color: #888; opacity: 0.6; }
    .evidence-tag {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px 3px 2px 0;
        border-radius: 3px;
        background: rgba(26, 58, 92, 0.1);
        color: var(--ca-primary);
        font-size: 0.78rem;
        font-family: 'SF Mono', monospace;
    }
    .evidence-contradicts {
        background: rgba(196, 69, 54, 0.1);
        color: var(--ca-accent);
    }
    .hypothesis-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0.6rem;
        border-bottom: 1px solid var(--ca-border);
        font-size: 0.9rem;
    }
    .rag-case {
        background: #fafaf7;
        border: 1px solid var(--ca-border);
        border-radius: 4px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .modality-pill {
        display: inline-block;
        padding: 1px 6px;
        margin-right: 4px;
        background: var(--ca-primary);
        color: white;
        font-size: 0.7rem;
        border-radius: 2px;
        font-family: 'SF Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# 
# Session State Initialization
# 

def init_session():
    defaults = {
        "config": DEFAULT_CONFIG.copy(),
        "planner": None,
        "agent_state": None,
        "thinking_steps": [],
        "chat_history": [],
        "uploaded_files": {},
        "running": False,
        "step_queue": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# 
# Helper: Build chat context from agent state
# 

def _build_chat_context(state, chat_history: List[dict]) -> List[dict]:
    """Build OpenAI-format messages with agent state as system context."""
    # Summarize the diagnostic state for context
    summary_parts = [
        f"Patient ID: {state.patient_id}",
        f"Modalities analyzed: {', '.join(state.modalities_used)}",
        "",
        f"## Extracted Phenotypes ({len(state.phenotype_terms)})",
    ]
    for p in state.phenotype_terms[:10]:
        summary_parts.append(f"- {p['term']} [{p['source']}] ({p['hpo']})")

    summary_parts.append(f"\n## Top Hypotheses")
    for h in state.hypothesis_ranking[:5]:
        summary_parts.append(
            f"{h.rank}. {h.diagnosis} "
            f"(confidence: {h.confidence:.2f}, "
            f"support: {h.support_count}, contradict: {h.contradict_count})"
        )
        if h.reasoning:
            summary_parts.append(f"   Reasoning: {h.reasoning}")

    cc = state.cross_modal_consistency
    if cc.get("agreements"):
        summary_parts.append(f"\n## Cross-Modal Agreements")
        for a in cc["agreements"][:5]:
            summary_parts.append(
                f"- {a['phenotype_a']} + {a['phenotype_b']}: "
                f"{a['explanation']}"
            )

    if state.rag_cases:
        summary_parts.append(f"\n## Retrieved Similar Cases ({len(state.rag_cases)})")
        for i, case in enumerate(state.rag_cases[:5]):
            if isinstance(case, dict):
                summary_parts.append(
                    f"{i+1}. [{case.get('source', '?')}] "
                    f"Dx: {case.get('diagnosis', 'N/A')[:80]} "
                    f"(score: {case.get('score', 0):.3f})"
                )

    summary_parts.append(f"\n## Final Report")
    summary_parts.append(state.final_report[:2000])

    system_prompt = (
        "You are CardioAgent, a multi-modal cardiac diagnostic AI in a "
        "clinical decision-support interface. A clinician is reviewing "
        "your analysis and wants to discuss it. Be concise, clinically "
        "accurate, and open to refinement. If the clinician challenges "
        "a finding or suggests an alternative diagnosis, reason carefully "
        "about whether to update your assessment.\n\n"
        "## Current Diagnostic State\n"
        + "\n".join(summary_parts)
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return messages


def _call_qwen_chat(qwen_url: str, messages: List[dict]) -> str:
    """Call Qwen3-VL for conversational response."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=qwen_url, api_key="cardioagent")
        response = client.chat.completions.create(
            model="qwen3-vl",
            messages=messages,
            max_tokens=1500,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"*Error calling Qwen: {e}*"
        
# 
# Visualization Helpers
# 

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]


def load_ecg_signal(ecg_path):
    """
    Load ECG signal from .npy or WFDB format.
    Returns (signal, fs, leads) or (None, None, None) on failure.
    signal shape: (n_leads, n_samples)
    """
    if ecg_path is None:
        return None, None, None
    try:
        p = Path(str(ecg_path))
        # .npy format
        if p.suffix == ".npy" or (not p.suffix and p.with_suffix(".npy").exists()):
            npy_path = str(p) if p.suffix == ".npy" else str(p.with_suffix(".npy"))
            arr = np.load(npy_path)
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if arr.ndim == 2 and arr.shape[0] > arr.shape[1] and arr.shape[1] <= 12:
                arr = arr.T
            n_leads = min(arr.shape[0], 12)
            return arr[:n_leads].astype(np.float32), 500, STANDARD_LEADS[:n_leads]
        # WFDB format
        try:
            import wfdb
            base = str(p.with_suffix("")) if p.suffix else str(p)
            record = wfdb.rdrecord(base)
            signal = record.p_signal.T
            return signal.astype(np.float32), int(record.fs), \
                [s.strip() for s in record.sig_name]
        except Exception:
            pass
        return None, None, None
    except Exception as e:
        logger.warning(f"ECG load failed for {ecg_path}: {e}")
        return None, None, None


def plot_ecg_waveform(signal, fs, leads, title="ECG Waveform"):
    """Plot 12-lead ECG using matplotlib. Returns a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_leads = signal.shape[0]
    duration = signal.shape[1] / fs
    t = np.linspace(0, duration, signal.shape[1])

    fig, axes = plt.subplots(n_leads, 1, figsize=(12, max(n_leads * 1.1, 5)),
                             sharex=True)
    if n_leads == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=12, fontweight="medium", y=0.98)

    for i, ax in enumerate(axes):
        lead_name = leads[i] if i < len(leads) else f"Lead {i+1}"
        sig = signal[i] - np.mean(signal[i])
        ax.plot(t, sig, color="#1a3a5c", linewidth=0.5, alpha=0.9)
        ax.set_ylabel(lead_name, fontsize=8, rotation=0, labelpad=28,
                      fontweight="medium")
        ax.set_xlim(0, duration)
        # ECG-style grid
        ax.set_facecolor("#fefefe")
        ax.grid(True, which="major", color="#ffcccc", linewidth=0.3)
        ax.grid(True, which="minor", color="#ffe6e6", linewidth=0.15)
        ax.minorticks_on()
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def display_ecg(ecg_path, title="Patient ECG", container=None):
    """Load and display ECG waveform in a Streamlit container."""
    target = container or st
    signal, fs, leads = load_ecg_signal(ecg_path)
    if signal is None:
        target.caption("_(ECG data not available)_")
        return
    n_leads, n_samples = signal.shape
    target.caption(f"{n_leads} leads | {fs} Hz | {n_samples/fs:.1f}s")
    fig = plot_ecg_waveform(signal, fs, leads, title=title)
    target.pyplot(fig, use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(fig)

def display_ecg_sync_annotated(ecg_path, ecg_findings, title, container):
    """ECG with annotations -- works for both patient and retrieved cases."""
    signal, fs, leads = load_ecg_signal(ecg_path)
    if signal is None:
        container.caption("_(ECG not available)_")
        return
    container.caption(f"{signal.shape[0]} leads | {fs} Hz | {signal.shape[1]/fs:.1f}s")
    fig = plot_ecg_waveform_with_annotations(
        signal, fs, leads, ecg_findings=ecg_findings, title=title,
    )
    container.pyplot(fig, use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(fig)


def display_cxr(cxr_path, title="Chest X-Ray", container=None):
    """Load and display CXR image."""
    target = container or st
    if cxr_path is None or not Path(str(cxr_path)).exists():
        target.caption("_(CXR image not available)_")
        return
    from PIL import Image
    img = Image.open(str(cxr_path))
    target.image(img, caption=title, use_container_width=True)


def overlay_region_boxes(cxr_path, lingshu_findings):
    """Draw bounding boxes for findings with location keywords."""
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = Image.open(cxr_path)
    w, h = img.size

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray" if img.mode == "L" else None)

    for f in lingshu_findings:
        finding_text = f.get("finding", "").lower()
        # Find matching region
        for region_name, (x1, y1, x2, y2) in CXR_REGIONS.items():
            if region_name in finding_text:
                px1, py1 = x1 * w, y1 * h
                px2, py2 = x2 * w, y2 * h
                rect = Rectangle((px1, py1), px2-px1, py2-py1,
                                 linewidth=2, edgecolor="#c44536",
                                 facecolor="none")
                ax.add_patch(rect)
                ax.text(px1, py1-10, f.get("finding", "")[:40],
                        fontsize=9, color="#c44536", fontweight="medium",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="#c44536"))
                break

    ax.axis("off")
    return fig
    
def plot_ecg_waveform_with_annotations(signal, fs, leads, ecg_findings=None, title="ECG"):
    """Same as plot_ecg_waveform but draws colored boxes on abnormal regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n_leads = signal.shape[0]
    duration = signal.shape[1] / fs
    t = np.linspace(0, duration, signal.shape[1])

    fig, axes = plt.subplots(n_leads, 1, figsize=(12, max(n_leads * 1.1, 5)),
                             sharex=True)
    if n_leads == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight="medium", y=0.98)

    # Build lead-name -> axis lookup
    lead_to_ax = {leads[i]: axes[i] for i in range(min(len(leads), len(axes)))}

    # Severity color map
    severity_colors = {
        "severe": "#c44536", "moderate": "#b45309",
        "mild": "#2d6a4f", "info": "#1a3a5c",
    }

    for i, ax in enumerate(axes):
        lead_name = leads[i] if i < len(leads) else f"Lead {i+1}"
        sig = signal[i] - np.mean(signal[i])
        ax.plot(t, sig, color="#1a3a5c", linewidth=0.5, alpha=0.9)
        ax.set_ylabel(lead_name, fontsize=8, rotation=0, labelpad=28,
                      fontweight="medium")
        ax.set_xlim(0, duration)
        ax.set_facecolor("#fefefe")
        ax.grid(True, which="major", color="#ffcccc", linewidth=0.3)
        ax.grid(True, which="minor", color="#ffe6e6", linewidth=0.15)
        ax.minorticks_on()
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Overlay annotations on relevant leads
    if ecg_findings:
        for f in ecg_findings:
            target_lead = f.get("lead", "")
            if target_lead not in lead_to_ax:
                continue
            ax = lead_to_ax[target_lead]
            severity = f.get("severity", "info")
            color = severity_colors.get(severity, "#888")

            # Time range from sample indices
            onset = f.get("onset_sample", 0)
            duration_samples = f.get("duration_samples", 200)
            t_start = onset / fs
            t_end = (onset + duration_samples) / fs

            # Y range from current axis limits
            ymin, ymax = ax.get_ylim()

            # Draw highlight rectangle
            rect = Rectangle(
                (t_start, ymin), t_end - t_start, ymax - ymin,
                linewidth=1.5, edgecolor=color, facecolor=color,
                alpha=0.15, zorder=0,
            )
            ax.add_patch(rect)

            # Add annotation label above
            label = f.get("type", f.get("finding", ""))[:20]
            ax.text(t_start, ymax * 0.85, label,
                    fontsize=7, color=color, fontweight="medium",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor=color, alpha=0.9))

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def overlay_biovilt_heatmap(cxr_path, biovilt_embedder, query_text=None):
    """
    Generate a saliency heatmap from BioViL-T's image-text similarity.
    If query_text is provided (e.g., 'cardiomegaly'), shows where in the image
    the model thinks that finding is located.
    """
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    img = Image.open(cxr_path).convert("L")
    transform = biovilt_embedder.image_inference.transform
    tensor = transform(img).unsqueeze(0).to(biovilt_embedder.device)

    with torch.no_grad():
        # Get spatial features (before global pooling)
        output = biovilt_embedder.image_inference.model(tensor)

        # BioViL-T outputs both global and patch-level features
        if hasattr(output, "patch_embeddings"):
            patches = output.patch_embeddings  # (1, H*W, C)
        elif hasattr(output, "img_embedding") and len(output.img_embedding.shape) == 4:
            patches = output.img_embedding  # (1, C, H, W)
        else:
            return None  # No spatial features available

        # If query text provided, compute similarity map
        if query_text:
            text_emb = biovilt_embedder.embed_text(query_text)
            text_emb = torch.from_numpy(text_emb).to(biovilt_embedder.device)

            # Reshape patches to (H, W, C)
            if patches.dim() == 4:  # (1, C, H, W)
                _, c, h, w = patches.shape
                patches_flat = patches.reshape(1, c, h*w).permute(0, 2, 1)
            else:  # (1, N, C)
                patches_flat = patches
                n = patches_flat.shape[1]
                h = w = int(np.sqrt(n))

            # Cosine similarity per patch
            patches_norm = patches_flat / patches_flat.norm(dim=-1, keepdim=True)
            text_norm = text_emb / text_emb.norm()
            sim = (patches_norm @ text_norm).squeeze().cpu().numpy()
            heatmap = sim.reshape(h, w)
        else:
            # Use feature norm as generic saliency
            if patches.dim() == 4:
                heatmap = patches.norm(dim=1).squeeze().cpu().numpy()
            else:
                norms = patches.norm(dim=-1).squeeze().cpu().numpy()
                h = w = int(np.sqrt(len(norms)))
                heatmap = norms.reshape(h, w)

    # Render overlay
    img_resized = img.resize((448, 448))
    img_arr = np.array(img_resized)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_arr, cmap="gray")
    # Upsample heatmap to image size
    from scipy.ndimage import zoom
    h_factor = img_arr.shape[0] / heatmap.shape[0]
    w_factor = img_arr.shape[1] / heatmap.shape[1]
    heatmap_up = zoom(heatmap, (h_factor, w_factor), order=1)
    ax.imshow(heatmap_up, cmap="jet", alpha=0.4)
    ax.axis("off")
    title = f"Attention: '{query_text}'" if query_text else "Saliency"
    ax.set_title(title, fontsize=11)
    return fig
    
def resolve_retrieved_ecg_path(case_dict):
    """Try to find ECG waveform file for a retrieved RAG case."""
    # Check metadata first
    meta = case_dict.get("metadata", {})
    wf_path = meta.get("waveform_path", "")

    if not wf_path:
        # Parse from case_id: "mimic4ecg_SUBJ_STUDY"
        case_id = case_dict.get("patient_id", "")
        parts = case_id.split("_")
        if len(parts) >= 3:
            subj = parts[1]
            study = parts[2]
            group = f"p{subj[:4]}"
            wf_path = f"files/{group}/p{subj}/s{study}/{study}"

    if not wf_path:
        return None

    # Search MIMIC-IV-ECG directories
    ecg_dirs = [
        "/scratch/rzv4ve/cardioagent/data/mimic-iv-ecg",
        "/sfs/weka/scratch/rzv4ve/cardioagent/data/MIMIC-IV-ECG-1",
    ]
    for d in ecg_dirs:
        full = Path(d) / wf_path
        if full.with_suffix(".hea").exists():
            return str(full)
    return None

def resolve_retrieved_cxr_path(case_dict):
    """Resolve CXR image path from metadata or by parsing case_id."""
    meta = case_dict.get("metadata", {})
    cxr_dirs = [
        "/sfs/weka/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg-2.0.0.physionet.org",
        "/scratch/rzv4ve/cardioagent/data/mimic-cxr-jpg",
    ]

    # Try metadata cxr_path first (works after rebuild with patched _batch_index)
    cxr_rel = meta.get("cxr_path", "")
    if cxr_rel:
        for d in cxr_dirs:
            full = Path(d) / cxr_rel
            if full.exists():
                return str(full)

    # Fallback: parse case_id like "mimic_cxr_<sid>_<study>_<dicom8>"
    case_id = case_dict.get("patient_id", "")
    if case_id.startswith("mimic_cxr_"):
        parts = case_id.split("_")
        if len(parts) >= 5:
            sid = parts[2]
            study = parts[3]
            dicom_prefix = parts[4]
            group = f"p{sid[:2]}"
            for d in cxr_dirs:
                study_dir = Path(d) / "files" / group / f"p{sid}" / f"s{study}"
                if study_dir.is_dir():
                    matches = list(study_dir.glob(f"{dicom_prefix}*.jpg"))
                    if matches:
                        return str(matches[0])
    return None

# 
# Planner Loading (cached)
# 

@st.cache_resource
def load_planner(qwen_url: str, lingshu_url: str, vector_db_path: str,
                 ecg_fm_path: str):
    """Load CardioAgentPlannerV4 with embedders. Cached across reruns."""
    try:
        from tools.embedding_tool import ECGFMEmbedder, BioViLTEmbedder, BiomedBERTEmbedder
        from planner_v4 import CardioAgentPlannerV4
        import numpy as np

        ecg_emb = ECGFMEmbedder(
            checkpoint_path=ecg_fm_path,
            device="cuda",
            strict_loading=False,
        )
        cxr_emb = BioViLTEmbedder(device="cuda")
        text_emb = BiomedBERTEmbedder(device="cuda")

        # Pre-warm models to avoid lazy loading on first retrieval
        logger.info("Pre-warming embedders...")
        try:
            _ = ecg_emb.embed(np.random.randn(12, 5000).astype(np.float32))
            logger.info("  ECG-FM warmed")
        except Exception as e:
            logger.warning(f"  ECG-FM warmup failed: {e}")

        try:
            from PIL import Image
            _ = cxr_emb.embed(Image.new('L', (480, 480)))
            logger.info("  BioViL-T warmed")
        except Exception as e:
            logger.warning(f"  BioViL-T warmup failed: {e}")

        try:
            _ = text_emb.embed("warmup")
            logger.info("  BiomedBERT warmed")
        except Exception as e:
            logger.warning(f"  BiomedBERT warmup failed: {e}")

        planner = CardioAgentPlannerV4(
            qwen_api_url=qwen_url,
            lingshu_api_url=lingshu_url,
            vector_db_path=vector_db_path,
            ecg_embedder=ecg_emb,
            cxr_embedder=cxr_emb,
            text_embedder=text_emb,
            enable_reflection=True,
            max_reflection_rounds=2,
            vector_top_k=5,
        )
        return planner, None
    except Exception as e:
        return None, str(e)
        
# 
# File Upload Handling
# 

def save_upload(uploaded_file, subdir: str) -> str:
    """Save uploaded file to temp dir and return path."""
    upload_root = Path(st.session_state.config["upload_dir"]) / subdir
    upload_root.mkdir(parents=True, exist_ok=True)
    dest = upload_root / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


# 
# Agent Execution with Step Streaming
# 

def run_agent_streaming(planner, patient_data, step_queue: Queue):
    """Run planner in background thread, stream steps via queue."""
    def on_step(step, state):
        # Serialize step for cross-thread transfer
        step_queue.put({
            "type": "step",
            "step_name": step.step_name,
            "status": step.status,
            "detail": step.detail,
            "duration_s": step.duration_s,
            "output_data": step.output_data,
        })

    planner.on_step = on_step

    try:
        result_state = planner.run(patient_data)
        step_queue.put({"type": "done", "state": result_state})
    except Exception as e:
        import traceback
        step_queue.put({"type": "error", "error": str(e),
                        "traceback": traceback.format_exc()})


# 
# Sidebar -- Configuration + Patient Input
# 

with st.sidebar:
    st.markdown("##  Configuration")

    with st.expander("Service endpoints", expanded=False):
        st.session_state.config["qwen_url"] = st.text_input(
            "Qwen3-VL URL", value=st.session_state.config["qwen_url"],
        )
        st.session_state.config["lingshu_url"] = st.text_input(
            "LingShu URL", value=st.session_state.config["lingshu_url"],
        )
        st.session_state.config["vector_db_path"] = st.text_input(
            "Vector DB path", value=st.session_state.config["vector_db_path"],
        )

    st.markdown("---")
    st.markdown("##  Patient Data")
    st.caption("Upload any subset of modalities. Missing data is OK.")

    patient_id = st.text_input("Patient ID", value="demo_patient_01")

    # ECG upload
    ecg_file = st.file_uploader(
        "ECG (.hea+.dat, .npy, .edf)",
        type=["hea", "dat", "npy", "edf"],
        accept_multiple_files=True,
        help="Upload .hea AND .dat for WFDB, or a single .npy array",
    )

    # CXR upload
    cxr_file = st.file_uploader(
        "Chest X-Ray (.jpg, .png, .dcm)",
        type=["jpg", "jpeg", "png", "dcm"],
    )

    # Lab upload
    lab_file = st.file_uploader(
        "Lab Results (.json, .csv, .txt)",
        type=["json", "csv", "txt"],
    )

    # Clinical notes
    clinical_notes = st.text_area(
        "Clinical Notes",
        placeholder="e.g., 68yo M with acute chest pain, SOB, diaphoresis...",
        height=100,
    )

    st.markdown("---")
    run_button = st.button(
        " Run Diagnostic Workflow",
        type="primary",
        disabled=st.session_state.running,
        use_container_width=True,
    )

    if st.session_state.running:
        st.caption(" Analysis in progress...")


# 
# Main Layout
# 

st.markdown("# CardioAgent")
st.caption(
    "Multi-modal cardiac diagnostic agent  DeepRare-inspired pipeline "
    "with transparent DST workflow and vector RAG retrieval"
)

# --- Handle new run --------------------------------------------------
if run_button:
    # Validate + save inputs
    from planner_v4 import PatientData

    ecg_path = None
    if ecg_file:
        # Handle multi-file WFDB upload (.hea + .dat)
        if isinstance(ecg_file, list):
            saved_paths = [save_upload(f, "ecg") for f in ecg_file]
            # Find .hea and use it (wfdb will auto-find .dat)
            for p in saved_paths:
                if p.endswith(".hea"):
                    ecg_path = p.replace(".hea", "")
                    break
            if not ecg_path and saved_paths:
                ecg_path = saved_paths[0]
        else:
            ecg_path = save_upload(ecg_file, "ecg")

    cxr_path = save_upload(cxr_file, "cxr") if cxr_file else None

    lab_results = None
    lab_text = None
    if lab_file:
        lab_path = save_upload(lab_file, "lab")
        if lab_path.endswith(".json"):
            with open(lab_path) as f:
                lab_results = json.load(f)
        elif lab_path.endswith(".csv"):
            df = pd.read_csv(lab_path)
            lab_results = df.to_dict() if len(df) > 0 else None
        else:
            with open(lab_path) as f:
                lab_text = f.read()

    patient_data = PatientData(
        patient_id=patient_id,
        ecg_path=ecg_path,
        image_path=cxr_path,
        image_type="cxr" if cxr_path else None,
        lab_results=lab_results,
        lab_text=lab_text,
        clinical_notes=clinical_notes if clinical_notes.strip() else None,
    )

    # Load planner if not cached
    #lanner, err = load_planner(st.session_state.config)
    
    
    cfg = st.session_state.config
    planner, err = load_planner(qwen_url=cfg["qwen_url"],
          lingshu_url=cfg["lingshu_url"],
          vector_db_path=cfg["vector_db_path"],
          ecg_fm_path=cfg["ecg_fm_path"],
    )
    
    if err:
        st.error(f"Failed to initialize CardioAgent: {err}")
        st.stop()
        
    # Reset state and start background execution
    st.session_state.planner = planner
    st.session_state.thinking_steps = []
    st.session_state.agent_state = None
    st.session_state.running = True
    st.session_state.chat_history = []
    st.session_state.uploaded_ecg_path = ecg_path
    st.session_state.uploaded_cxr_path = cxr_path
    st.session_state.patient_data = patient_data

    q = Queue()
    st.session_state.step_queue = q
    thread = threading.Thread(
        target=run_agent_streaming,
        args=(planner, patient_data, q),
        daemon=True,
    )
    thread.start()
    st.rerun()


# --- Drain step queue (live update during execution) ----------------
if st.session_state.running and st.session_state.step_queue:
    q = st.session_state.step_queue
    got_update = False
    try:
        while True:
            msg = q.get_nowait()
            got_update = True
            if msg["type"] == "step":
                st.session_state.thinking_steps.append(msg)
            elif msg["type"] == "done":
                st.session_state.agent_state = msg["state"]
                st.session_state.running = False
                break
            elif msg["type"] == "error":
                st.error(f"Agent execution failed: {msg['error']}")
                with st.expander("Traceback"):
                    st.code(msg["traceback"])
                st.session_state.running = False
                break
    except Empty:
        pass

    if st.session_state.running:
        time.sleep(0.3)
        st.rerun()


# 
# Tab Layout
# 

tab_data, tab_workflow, tab_reasoning, tab_rag, tab_report, tab_chat = st.tabs([
    "Patient Data",
    "DST Workflow",
    "Reasoning",
    "RAG Retrieval",
    "Final Report",
    "Discuss",
])



# -----------------------------------------------------------------------
# Tab 0: Patient Data (uploaded ECG waveform + CXR image)
# -----------------------------------------------------------------------
with tab_data:
    st.markdown("### Uploaded Patient Data")
    
    ecg_path = st.session_state.get("uploaded_ecg_path")
    cxr_path = st.session_state.get("uploaded_cxr_path")
    state = st.session_state.agent_state
    
    if state and state.ecg_result:
        with st.expander("DEBUG: raw ECG result", expanded=False):
            st.json(state.ecg_result)
        
    if ecg_path is None and cxr_path is None:
        st.info("Upload ECG and/or CXR files in the sidebar, then run the workflow.")
    else:
        col_ecg, col_cxr = st.columns([3, 2])

        with col_ecg:
            st.markdown("#### ECG Waveform")
            if ecg_path:
                ecg_findings = []
                if state and state.ecg_result:
                    metrics = state.ecg_result.get("metrics", {})
                    # Combine all findings with location info
                    ecg_findings.extend(metrics.get("st_findings", []))
                    ecg_findings.extend(metrics.get("rhythm_findings", []))
                    # ECG tool findings already include lead and severity
                    for f in state.ecg_result.get("findings", []):
                        if "lead" in f and "onset_sample" in f:
                            ecg_findings.append(f)
                st.write("DEBUG ECG findings:", ecg_findings)
                
                if ecg_path:
                    signal, fs, leads = load_ecg_signal(ecg_path)
                    if signal is not None:
                        fig = plot_ecg_waveform_with_annotations(
                            signal, fs, leads, ecg_findings=ecg_findings,
                            title="Uploaded ECG (annotated)",
                        )
                        st.pyplot(fig, use_container_width=True)

                # Show ECG tool analysis results if available
                if state and state.ecg_result:
                    metrics = state.ecg_result.get("metrics", {})
                    with st.expander("ECG Tool Analysis", expanded=True):
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Heart Rate",
                                  f"{metrics.get('heart_rate_bpm', 'N/A')} bpm")
                        m2.metric("Rhythm",
                                  metrics.get("rhythm_detail", "N/A"))
                        m3.metric("QTc",
                                  f"{metrics.get('qtc_bazett_ms', 'N/A')} ms")

                        findings = state.ecg_result.get("findings", [])
                        if findings:
                            st.markdown("**Findings:**")
                            for f in findings:
                                severity = f.get("severity", "")
                                sev_color = {"severe": "#c44536",
                                             "moderate": "#b45309",
                                             "mild": "#2d6a4f"}.get(severity, "#555")
                                st.markdown(
                                    f"- {f.get('finding', '')} "
                                    f"(<span style='color:{sev_color}'>"
                                    f"{severity}</span>)",
                                    unsafe_allow_html=True,
                                )

                        st_findings = metrics.get("st_findings", [])
                        if st_findings:
                            st.markdown("**ST Segment:**")
                            for stf in st_findings:
                                st.markdown(
                                    f"- {stf.get('type', '')} in "
                                    f"{stf.get('lead', '')}: "
                                    f"{stf.get('st_deviation_mv', 0):.2f} mV"
                                )
            else:
                st.caption("_(No ECG uploaded)_")

        with col_cxr:
            st.markdown("#### Chest X-Ray")
            if cxr_path:
                # Get LingShu findings for region overlay
                lingshu_findings = []
                if state and state.image_result:
                    lingshu_findings = state.image_result.get("findings", [])

                if lingshu_findings:
                    fig = overlay_region_boxes(cxr_path, lingshu_findings)
                    st.pyplot(fig, use_container_width=True)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                else:
                    display_cxr(cxr_path, title="Uploaded CXR", container=st)

                # Show LingShu analysis if available
                if state and state.image_result:
                    with st.expander("LingShu CXR Analysis", expanded=True):
                        analysis = state.image_result.get(
                            "lingshu_analysis", "")
                        if analysis:
                            st.markdown(analysis[:1500])
                        img_findings = state.image_result.get("findings", [])
                        if img_findings:
                            st.markdown("**Structured findings:**")
                            for f in img_findings:
                                st.markdown(
                                    f"- {f.get('finding', '')[:150]}"
                                )
            else:
                st.caption("_(No CXR uploaded)_")

        # Lab results summary
        if state and state.lab_result:
            with st.expander("Lab Results", expanded=False):
                summary = state.lab_result.get("text_summary", "")
                if summary:
                    st.text(summary)
                abnormals = state.lab_result.get("abnormal", [])
                if abnormals:
                    df = pd.DataFrame(abnormals)
                    st.dataframe(df, hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------
# Tab 1: DST Workflow (live streaming steps)
# -----------------------------------------------------------------------
with tab_workflow:
    st.markdown("### Pipeline Execution")
    st.caption(
        "Each phase runs sequentially. Watch tools fire, phenotypes get "
        "extracted, hypotheses form, RAG retrieve, reflection converge, "
        "and Qwen synthesize the final report."
    )

    if not st.session_state.thinking_steps and not st.session_state.running:
        st.info("Upload patient data and click **Run Diagnostic Workflow** to start.")
    else:
        # Phase grouping for visual hierarchy
        phase_groups = {
            "Phase 1 -- Tool execution": [
                "planning", "ecg_analysis", "image_analysis",
                "lab_analysis", "clinical_notes",
            ],
            "Phase 2 -- Phenotype extraction": ["phenotype_extraction"],
            "Phase 3 -- Hypothesis generation": ["hypothesis_generation"],
            "Phase 4 -- RAG retrieval": [
                "vector_retrieval", "keyword_retrieval", "rag_retrieval",
            ],
            "Phase 5 -- Reflection loop": ["reflection"],
            "Phase 6 -- Synthesis": ["planner_synthesis"],
        }

        for phase_label, step_names in phase_groups.items():
            matching = [s for s in st.session_state.thinking_steps
                        if s["step_name"] in step_names]
            if not matching:
                continue

            st.markdown(f"**{phase_label}**")
            for step in matching:
                css_class = f"phase-{step['status']}"
                duration = f"{step['duration_s']}s" if step['duration_s'] else ""
                status_icon = {
                    "success": "",
                    "running": "",
                    "error": "",
                    "skipped": "--",
                }.get(step['status'], "")

                st.markdown(
                    f'<div class="phase-card {css_class}">'
                    f'<strong>{status_icon} {step["step_name"]}</strong> '
                    f'<span style="color:#888;float:right">{duration}</span>'
                    f'<br>{step["detail"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if st.session_state.running:
            st.info(" Pipeline running... steps will appear as they complete.")


# -----------------------------------------------------------------------
# Tab 2: Reasoning (phenotypes, hypotheses, reflection)
# -----------------------------------------------------------------------
with tab_reasoning:
    state = st.session_state.agent_state

    if state is None:
        st.info("No reasoning data yet. Run the workflow first.")
    else:
        col_left, col_right = st.columns([1, 1])

        # -- Phenotypes --
        with col_left:
            st.markdown("### Extracted Phenotypes")
            st.caption("HPO-aligned terms identified across modalities")
            if state.phenotype_terms:
                for p in state.phenotype_terms:
                    modality = p.get("source", "?")
                    st.markdown(
                        f'<span class="modality-pill">{modality.upper()}</span>'
                        f'**{p["term"]}** '
                        f'<span style="color:#888;font-size:0.8rem">'
                        f'({p["hpo"]})</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("-- no phenotypes extracted --")

        # -- Hypotheses --
        with col_right:
            st.markdown("### Diagnostic Hypotheses")
            st.caption(
                f"Ranked by evidence after "
                f"{state.reflection_rounds} reflection round(s)"
            )
            if state.hypothesis_ranking:
                for h in state.hypothesis_ranking[:5]:
                    conf_pct = int(h.confidence * 100)
                    bar_color = "#2d6a4f" if conf_pct > 70 else \
                                "#b45309" if conf_pct > 40 else "#c44536"

                    st.markdown(
                        f"**{h.rank}. {h.diagnosis}**"
                    )
                    st.markdown(
                        f'<div style="background:#eee;height:6px;border-radius:3px;'
                        f'margin:4px 0 8px;">'
                        f'<div style="width:{conf_pct}%;height:100%;'
                        f'background:{bar_color};border-radius:3px"></div></div>'
                        f'<div style="font-size:0.8rem;color:#555">'
                        f'Confidence: {h.confidence:.2f}  '
                        f'{h.support_count} supporting  '
                        f'{h.contradict_count} contradicting  '
                        f'modalities: {", ".join(h.modalities_involved)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if h.reasoning:
                        st.caption(f"_{h.reasoning}_")
                    st.markdown("---")
            else:
                st.caption("-- no hypotheses generated --")

        # -- Cross-modal consistency --
        st.markdown("### Cross-Modal Consistency")
        cc = state.cross_modal_consistency

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"** Agreements** ({cc.get('n_agreements', 0)})")
            for a in cc.get("agreements", []):
                st.markdown(
                    f'<span class="evidence-tag">{a["phenotype_a"]} + '
                    f'{a["phenotype_b"]}</span><br>'
                    f'<small>{a["explanation"]}</small>',
                    unsafe_allow_html=True,
                )
        with c2:
            st.markdown(f"** Contradictions** ({cc.get('n_contradictions', 0)})")
            for c in cc.get("contradictions", []):
                st.markdown(
                    f'<span class="evidence-tag evidence-contradicts">'
                    f'{c["phenotype_a"]} vs {c["phenotype_b"]}</span><br>'
                    f'<small>{c["explanation"]}</small>',
                    unsafe_allow_html=True,
                )

        # -- Reflection log --
        if state.reflection_log:
            with st.expander(
                f"Reflection log ({state.reflection_rounds} rounds)",
                expanded=False,
            ):
                for round_info in state.reflection_log:
                    st.markdown(f"**Round {round_info['round']}** "
                                f"-- rank changed: {round_info.get('rank_changed')}")
                    ranking = round_info.get("ranking", [])
                    if ranking:
                        df = pd.DataFrame(ranking)
                        st.dataframe(df, hide_index=True,
                                     use_container_width=True)


# -----------------------------------------------------------------------
# Tab 3: RAG Retrieval (what cases retrieved, why, scores)
# -----------------------------------------------------------------------
with tab_rag:
    state = st.session_state.agent_state

    if state is None or not state.rag_cases:
        st.info(
            "No RAG cases retrieved. This could mean:\n"
            "- Vector DB is empty\n"
            "- No modalities matched\n"
            "- Retrieval disabled for this ablation"
        )
    else:
        n_cases = len(state.rag_cases)
        st.markdown(f"### {n_cases} Similar Cases Retrieved")
        st.caption(
            "Two-stage pipeline: dense retrieval (ECG-FM + BioViL-T + BiomedBERT) "
            "-> RRF fusion -> MedCPT cross-encoder reranking"
        )

        # --- Patient vs Top Retrieved Case (side by side comparison) ---
        top_case = state.rag_cases[0] if state.rag_cases else None
        patient_ecg = st.session_state.get("uploaded_ecg_path")
        patient_cxr = st.session_state.get("uploaded_cxr_path")

        has_patient_data = patient_ecg or patient_cxr
        has_retrieved_data = False
        if top_case and isinstance(top_case, dict):
            has_retrieved_data = (
                resolve_retrieved_ecg_path(top_case) is not None or
                resolve_retrieved_cxr_path(top_case) is not None
            )

        if has_patient_data and has_retrieved_data:
            with st.expander("Patient vs Top Retrieved Case (side by side)",
                             expanded=True):
                top_dx = top_case.get("diagnosis", "N/A")[:50]
                top_score = top_case.get("score", 0)
                st.markdown(
                    f"**Top match:** {top_dx} (score: {top_score:.3f})"
                )

                patient_ecg_findings = []
                if state.ecg_result:
                    metrics = state.ecg_result.get("metrics", {})
                    patient_ecg_findings.extend(metrics.get("st_findings", []))
                    patient_ecg_findings.extend(metrics.get("rhythm_findings", []))
                    for f in state.ecg_result.get("findings", []):
                        if "lead" in f and "onset_sample" in f:
                            patient_ecg_findings.append(f)
                # Assign severity if missing
                for f in patient_ecg_findings:
                    if not f.get("severity"):
                        st_dev = abs(f.get("st_deviation_mv", 0))
                        if st_dev > 0.2:
                            f["severity"] = "severe"
                        elif st_dev > 0.1:
                            f["severity"] = "moderate"
                        else:
                            f["severity"] = "mild"

                if patient_ecg:
                    st.markdown("##### ECG Comparison")
                    cmp_left, cmp_right = st.columns(2)
                    with cmp_left:
                        st.markdown("**Patient**")
                        display_ecg_sync_annotated(
                            patient_ecg,
                            ecg_findings=patient_ecg_findings,
                            title="Patient ECG",
                            container=st,
                        )
                    with cmp_right:
                        st.markdown("**Top Retrieved Case**")
                        ret_path = resolve_retrieved_ecg_path(top_case)
                        if ret_path:
                            # KEY: pass the SAME findings to retrieved case
                            # This visually shows "the model matched on these"
                            display_ecg_sync_annotated(
                                ret_path,
                                ecg_findings=patient_ecg_findings,
                                title=f"Retrieved: {top_dx}",
                                container=st,
                            )
                        else:
                            st.caption("_(ECG not available)_")

                if patient_cxr:
                    st.markdown("##### CXR Comparison")
                    cmp_left2, cmp_right2 = st.columns(2)
                    with cmp_left2:
                        st.markdown("**Patient**")
                        display_cxr(patient_cxr, title="Patient CXR",
                                    container=st)
                    with cmp_right2:
                        st.markdown("**Top Retrieved Case**")
                        ret_cxr = resolve_retrieved_cxr_path(top_case)
                        if ret_cxr:
                            display_cxr(ret_cxr,
                                        title=f"Retrieved: {top_dx}",
                                        container=st)
                        else:
                            st.caption("_(CXR not available)_")

        # Summary stats
        all_modalities = set()
        for c in state.rag_cases:
            if isinstance(c, dict):
                all_modalities.update(c.get("modalities_matched", []))

        col1, col2, col3 = st.columns(3)
        col1.metric("Cases retrieved", n_cases)
        col2.metric("Modalities queried", len(all_modalities))
        col3.metric("Top score",
                    f"{state.rag_cases[0].get('score', 0):.3f}"
                    if state.rag_cases else "--")

        st.markdown("---")

        # Each retrieved case
        for i, case in enumerate(state.rag_cases):
            if not isinstance(case, dict):
                continue

            case_id = case.get("patient_id", "?")
            diagnosis = case.get("diagnosis", "N/A")
            report = case.get("report", "")
            score = case.get("score", 0)
            modalities = case.get("modalities_matched", [])
            mod_scores = case.get("modality_scores", {})
            source = case.get("source", "unknown")

            with st.container():
                st.markdown(f"#### Case {i+1}: `{case_id}`")

                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.markdown(f"**Diagnosis:** {diagnosis}")
                    if report:
                        st.markdown(f"**Report:** _{report[:400]}_")
                    st.markdown(
                        f'<small>Source: <code>{source}</code></small>',
                        unsafe_allow_html=True,
                    )

                with col_r:
                    st.metric("Fused score", f"{score:.4f}")
                    st.caption("Matched via:")
                    for mod in modalities:
                        mod_score = mod_scores.get(mod, 0)
                        st.markdown(
                            f'<span class="modality-pill">{mod.upper()}</span> '
                            f'<code>{mod_score:.3f}</code>',
                            unsafe_allow_html=True,
                        )

                # --- Visualize retrieved case ECG + CXR ---
                with st.expander(
                    f"View Case {i+1} waveforms and images",
                    expanded=False,
                ):
                    vis_ecg, vis_cxr = st.columns([3, 2])

                    with vis_ecg:
                        ret_ecg_path = resolve_retrieved_ecg_path(case)
                        if ret_ecg_path:
                            signal, fs, leads = load_ecg_signal(ret_ecg_path)
                            if signal is not None:
                                st.caption(
                                    f"Retrieved ECG: {signal.shape[0]} leads, "
                                    f"{fs} Hz, {signal.shape[1]/fs:.1f}s"
                                )
                                fig = plot_ecg_waveform(
                                    signal, fs, leads,
                                    title=f"Case {i+1}: {diagnosis[:40]}",
                                )
                                st.pyplot(fig, use_container_width=True)
                                import matplotlib.pyplot as plt
                                plt.close(fig)
                            else:
                                st.caption("_(ECG waveform not loadable)_")
                        else:
                            st.caption("_(ECG path not resolved)_")

                    with vis_cxr:
                        ret_cxr_path = resolve_retrieved_cxr_path(case)
                        if ret_cxr_path:
                            display_cxr(
                                ret_cxr_path,
                                title=f"Case {i+1} CXR",
                                container=st,
                            )
                        else:
                            st.caption("_(CXR image not available)_")

                st.markdown("---")

        # Why these cases? (LLM-generated rationale)
        with st.expander(" Why were these cases retrieved?", expanded=False):
            st.markdown(
                "The retrieval pipeline scores cases by **physiological "
                "similarity** across modalities, not just keyword overlap:\n\n"
                "1. **ECG-FM** encodes the 12-lead waveform into a dense "
                "vector; similar electrophysiology clusters together\n"
                "2. **BioViL-T** encodes the CXR into a radiology-aligned "
                "vector space\n"
                "3. **BiomedBERT** encodes clinical text\n"
                "4. **RRF fusion** merges per-modality rankings "
                "(parameter-free, robust to score scales)\n"
                "5. **MedCPT cross-encoder** reranks using joint query+case "
                "attention -- catches fine-grained relevance that bi-encoders miss"
            )


# -----------------------------------------------------------------------
# Tab 4: Final Report
# -----------------------------------------------------------------------
with tab_report:
    state = st.session_state.agent_state

    if state is None:
        st.info("No report yet. Run the workflow first.")
    else:
        st.markdown("### Final Synthesized Report")
        st.caption(
            f"Patient {state.patient_id}  "
            f"{len(state.modalities_used)} modalities used  "
            f"{state.elapsed_s}s total"
        )

        # Render the markdown report
        report = state.final_report or "_(no report generated)_"
        st.markdown(report)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                " Download report (Markdown)",
                data=report,
                file_name=f"cardioagent_{state.patient_id}.md",
                mime="text/markdown",
            )
        with col2:
            # Export full state as JSON for clinical archiving
            state_dict = {
                "patient_id": state.patient_id,
                "modalities_used": state.modalities_used,
                "phenotypes": state.phenotype_terms,
                "hypotheses": [
                    {"rank": h.rank, "diagnosis": h.diagnosis,
                     "confidence": h.confidence}
                    for h in state.hypothesis_ranking[:5]
                ],
                "reflection_rounds": state.reflection_rounds,
                "rag_cases": state.rag_cases,
                "cross_modal_consistency": state.cross_modal_consistency,
                "final_report": state.final_report,
                "elapsed_s": state.elapsed_s,
            }
            st.download_button(
                " Export full state (JSON)",
                data=json.dumps(state_dict, indent=2, default=str),
                file_name=f"cardioagent_state_{state.patient_id}.json",
                mime="application/json",
            )


# -----------------------------------------------------------------------
# Tab 5: Discuss (conversational refinement with Qwen)
# -----------------------------------------------------------------------
with tab_chat:
    state = st.session_state.agent_state

    if state is None:
        st.info(
            "Run the workflow first, then use this tab to:\n"
            "- Ask why the agent made specific decisions\n"
            "- Request deeper explanation of any finding\n"
            "- Suggest alternative diagnoses for reconsideration\n"
            "- Ask about the retrieved cases in detail"
        )
    else:
        st.markdown("### Conversational Refinement")
        st.caption(
            "Discuss the case with Qwen. The full diagnostic state is in "
            "context -- ask about any finding, hypothesis, or retrieved case."
        )

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_msg = st.chat_input("Ask about the diagnosis, request changes, or challenge a finding...")

        if user_msg:
            # Add to history immediately
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_msg,
            })

            # Build context prompt with full agent state
            context = _build_chat_context(state, st.session_state.chat_history)

            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("Qwen is thinking..."):
                    response = _call_qwen_chat(
                        st.session_state.config["qwen_url"],
                        context,
                    )
                    st.markdown(response)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
            })
            st.rerun()


