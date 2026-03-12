"""
CardioAgent — Interactive Diagnostic Agent GUI
================================================
Streamlit interface for multi-modal cardiac analysis.

Features:
  - Upload raw ECG (WFDB/EDF/CSV/NPY) and DICOM MRI files
  - Watch the agent think step-by-step in real time
  - View generated waveforms and montages
  - Read integrated diagnostic report

Run: streamlit run app.py --server.port 8501
"""

import streamlit as st
import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from planner import CardioAgentPlanner, AgentState, ThinkingStep

# ── Page Config ──
st.set_page_config(
    page_title="CardioAgent",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* Dark clinical theme */
    .stApp {
        background-color: #0a0e17;
    }
    
    /* Header */
    .agent-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
    }
    .agent-header h1 {
        color: #4ecdc4;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .agent-header p {
        color: #6b7b8d;
        font-size: 14px;
        margin: 6px 0 0 0;
    }
    
    /* Thinking step cards */
    .think-step {
        background: #111827;
        border-left: 3px solid #374151;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .think-step.running {
        border-left-color: #f59e0b;
        background: #1a1a0a;
    }
    .think-step.success {
        border-left-color: #10b981;
    }
    .think-step.error {
        border-left-color: #ef4444;
        background: #1a0a0a;
    }
    .think-step.skipped {
        border-left-color: #6b7280;
        opacity: 0.6;
    }
    
    .step-name {
        font-size: 13px;
        font-weight: 600;
        color: #e5e7eb;
    }
    .step-detail {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 4px;
        line-height: 1.5;
    }
    .step-time {
        font-size: 11px;
        color: #6b7280;
        float: right;
    }
    
    /* Status icons */
    .status-icon {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-icon.running { background: #f59e0b; animation: pulse 1s infinite; }
    .status-icon.success { background: #10b981; }
    .status-icon.error { background: #ef4444; }
    .status-icon.skipped { background: #6b7280; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    /* Substep */
    .substep {
        font-size: 11px;
        color: #6b7280;
        padding: 2px 0 2px 20px;
        border-left: 1px dashed #374151;
        margin-left: 8px;
    }
    .substep.success { color: #6ee7b7; }
    .substep.error { color: #fca5a5; }
    
    /* Report card */
    .report-card {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 24px;
        margin-top: 16px;
    }
    
    /* Metric card */
    .metric-card {
        background: #1f2937;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #4ecdc4;
    }
    .metric-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* File upload area */
    .upload-zone {
        border: 2px dashed #1e3a5f;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False


# ── Helper Functions ──

def render_thinking_step(step: ThinkingStep):
    """Render a single thinking step as styled HTML"""
    icon_map = {
        "running": "⏳", "success": "✅", "error": "❌", "skipped": "⏭️"
    }
    name_map = {
        "planning": "🧠 Planning",
        "ecg_analysis": "💓 ECG Analysis (ECGFounder)",
        "mri_analysis": "🫀 MRI Analysis (LingShu-8B)",
        "planner_synthesis": "🔬 Integrated Synthesis (Qwen3-VL)",
    }

    icon = icon_map.get(step.status, "⏳")
    display_name = name_map.get(step.step_name, step.step_name)
    time_str = f"{step.duration_s}s" if step.duration_s > 0 else ""

    st.markdown(f"""
    <div class="think-step {step.status}">
        <span class="status-icon {step.status}"></span>
        <span class="step-name">{icon} {display_name}</span>
        <span class="step-time">{time_str}</span>
        <div class="step-detail">{step.detail}</div>
    </div>
    """, unsafe_allow_html=True)

    # Show substeps if available
    if step.substeps:
        for sub in step.substeps:
            sub_status = sub.get("status", "")
            sub_icon = "✓" if sub_status == "success" else "✗" if sub_status == "error" else "→"
            st.markdown(
                f'<div class="substep {sub_status}">{sub_icon} {sub.get("step", "")}: {sub.get("detail", "")}</div>',
                unsafe_allow_html=True
            )


def save_uploaded_files(ecg_files, dicom_files) -> tuple:
    """Save uploaded files to temp directory, return paths"""
    tmp_dir = tempfile.mkdtemp(prefix="cardioagent_")

    ecg_path = None
    dicom_path = None

    if ecg_files:
        ecg_dir = os.path.join(tmp_dir, "ecg")
        os.makedirs(ecg_dir, exist_ok=True)
        for f in ecg_files:
            fpath = os.path.join(ecg_dir, f.name)
            with open(fpath, "wb") as out:
                out.write(f.getbuffer())
            # Use .hea file as the ECG path (for WFDB)
            if f.name.endswith(".hea"):
                ecg_path = fpath
            elif ecg_path is None:
                ecg_path = fpath

    if dicom_files:
        dicom_dir = os.path.join(tmp_dir, "dicom")
        os.makedirs(dicom_dir, exist_ok=True)
        for f in dicom_files:
            fpath = os.path.join(dicom_dir, f.name)
            with open(fpath, "wb") as out:
                out.write(f.getbuffer())
        # If multiple DICOMs, use directory; if single, use file
        if len(dicom_files) > 1:
            dicom_path = dicom_dir
        else:
            dicom_path = os.path.join(dicom_dir, dicom_files[0].name)

    return ecg_path, dicom_path, tmp_dir


# ── Header ──
st.markdown("""
<div class="agent-header">
    <h1>🫀 CardioAgent</h1>
    <p>Multi-Modal Cardiac Diagnostic Agent — ECGFounder + LingShu-8B + Qwen3-VL</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Input ──
with st.sidebar:
    st.markdown("### 📁 Patient Data Input")

    patient_id = st.text_input("Patient ID", value="Patient_001")

    st.markdown("---")
    st.markdown("**ECG Data**")
    st.caption("Upload .hea + .dat (WFDB), .edf, .csv, or .npy")
    ecg_files = st.file_uploader(
        "ECG files",
        type=["hea", "dat", "edf", "csv", "npy", "xml"],
        accept_multiple_files=True,
        key="ecg_upload",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Cardiac MRI / CT**")
    st.caption("Upload .dcm files (single or multi-slice series)")
    dicom_files = st.file_uploader(
        "DICOM files",
        type=["dcm"],
        accept_multiple_files=True,
        key="dicom_upload",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Clinical Notes**")
    clinical_notes = st.text_area(
        "Clinical context",
        placeholder="e.g., 55yo male, acute chest pain, troponin elevated...",
        height=100,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Server Configuration**")
    qwen_url = st.text_input("Qwen3-VL URL", value="http://localhost:8000/v1")
    lingshu_url = st.text_input("LingShu-8B URL", value="http://localhost:8001/v1")

    st.markdown("---")
    run_button = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running or (not ecg_files and not dicom_files),
    )

# ── Main Content ──
col_thinking, col_results = st.columns([2, 3])

with col_thinking:
    st.markdown("### 🧠 Agent Thinking Process")
    thinking_container = st.container()

with col_results:
    st.markdown("### 📊 Results")
    results_container = st.container()


# ── Run Analysis ──
if run_button and (ecg_files or dicom_files):
    st.session_state.is_running = True

    # Save uploaded files
    ecg_path, dicom_path, tmp_dir = save_uploaded_files(ecg_files, dicom_files)

    # Initialize planner
    planner = CardioAgentPlanner(
        qwen_api_url=qwen_url,
        qwen_model_name="Qwen3-VL-30B",
        lingshu_api_url=lingshu_url,
    )

    # Run with real-time updates
    with thinking_container:
        st.markdown(f"**Patient:** {patient_id} | **Started:** {time.strftime('%H:%M:%S')}")

        # Progress bar
        progress = st.progress(0, text="Initializing...")

        # Step placeholders for real-time updates
        step_placeholders = {}
        for step_name in ["planning", "ecg_analysis", "mri_analysis", "planner_synthesis"]:
            step_placeholders[step_name] = st.empty()

        # Callback for real-time step updates
        step_count = [0]
        total_steps = 4

        def on_step_update(step: ThinkingStep, state: AgentState):
            step_count[0] += 1
            progress.progress(
                min(step_count[0] / total_steps, 1.0),
                text=f"Step {step_count[0]}/{total_steps}: {step.step_name}"
            )
            with step_placeholders.get(step.step_name, st.empty()):
                render_thinking_step(step)

        planner.on_step = on_step_update

        # Execute pipeline
        state = planner.run(
            patient_id=patient_id,
            ecg_path=ecg_path,
            dicom_input=dicom_path,
            clinical_notes=clinical_notes,
        )

        st.session_state.agent_state = state
        progress.progress(1.0, text=f"Complete in {state.elapsed_s}s")

    # Display results
    with results_container:
        # ── Metrics Row ──
        if state.ecg_result and state.ecg_result.get("metrics"):
            m = state.ecg_result["metrics"]
            met_cols = st.columns(4)
            with met_cols[0]:
                st.metric("Heart Rate", f"{m.get('heart_rate_bpm', '—')} bpm")
            with met_cols[1]:
                st.metric("Rhythm", m.get("rhythm_detail", "—"))
            with met_cols[2]:
                st.metric("RR Interval", f"{m.get('rr_mean_ms', '—')} ms")
            with met_cols[3]:
                st.metric("Beats Detected", m.get("n_beats", "—"))

        st.markdown("---")

        # ── Images ──
        img_cols = st.columns(2)

        with img_cols[0]:
            if state.images.get("ecg_waveform") and os.path.exists(state.images["ecg_waveform"]):
                st.markdown("**12-Lead ECG Waveform**")
                st.image(state.images["ecg_waveform"], use_container_width=True)
            else:
                st.info("No ECG waveform generated")

        with img_cols[1]:
            if state.images.get("mri_montage") and os.path.exists(state.images["mri_montage"]):
                st.markdown("**Cardiac MRI Montage**")
                st.image(state.images["mri_montage"], use_container_width=True)
            else:
                st.info("No MRI montage generated")

        st.markdown("---")

        # ── Findings Summary ──
        st.markdown("**📋 Tool Findings**")

        find_col1, find_col2 = st.columns(2)

        with find_col1:
            st.markdown("**ECG Findings**")
            ecg_findings = state.ecg_result.get("findings", [])
            if ecg_findings:
                for f in ecg_findings:
                    sev = f.get("severity", "unknown")
                    color = {"severe": "🔴", "moderate": "🟡", "mild": "🟢", "normal": "⚪"}.get(sev, "⚫")
                    st.markdown(f"{color} {f.get('finding', 'N/A')}")
            else:
                st.caption("No ECG findings")

        with find_col2:
            st.markdown("**MRI Findings**")
            mri_findings = state.mri_result.get("findings", [])
            if mri_findings:
                for f in mri_findings:
                    sev = f.get("severity", "unknown")
                    color = {"severe": "🔴", "moderate": "🟡", "mild": "🟢", "normal": "⚪"}.get(sev, "⚫")
                    st.markdown(f"{color} {f.get('finding', 'N/A')[:150]}")
            else:
                st.caption("No MRI findings")

        st.markdown("---")

        # ── Integrated Report (Qwen3-VL) ──
        st.markdown("### 📝 Integrated Diagnostic Report")
        st.markdown(f"""<div class="report-card">{state.final_report}</div>""",
                    unsafe_allow_html=True)

        # ── Raw Data Expander ──
        with st.expander("🔍 Raw Tool Outputs (Debug)"):
            tab1, tab2, tab3 = st.tabs(["ECG Raw", "MRI Raw", "Thinking Steps"])
            with tab1:
                st.json(state.ecg_result if state.ecg_result else {"status": "no data"})
            with tab2:
                # Truncate long lingshu analysis for display
                mri_display = dict(state.mri_result) if state.mri_result else {"status": "no data"}
                if "lingshu_analysis" in mri_display and len(str(mri_display["lingshu_analysis"])) > 2000:
                    mri_display["lingshu_analysis"] = str(mri_display["lingshu_analysis"])[:2000] + "..."
                st.json(mri_display)
            with tab3:
                for step in state.thinking_steps:
                    st.json({
                        "step": step.step_name,
                        "status": step.status,
                        "detail": step.detail,
                        "duration_s": step.duration_s,
                        "substeps": step.substeps,
                    })

    # Cleanup
    st.session_state.is_running = False

    # Cleanup temp files after a delay
    # (Don't clean immediately — images are still being displayed)


# ── Empty State ──
elif not st.session_state.agent_state:
    with thinking_container:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; color: #6b7280;">
            <p style="font-size: 40px; margin: 0;">🫀</p>
            <p style="font-size: 16px; margin: 10px 0 4px 0; color: #9ca3af;">Upload ECG or DICOM files to begin</p>
            <p style="font-size: 12px;">
                The agent will analyze data through ECGFounder → LingShu-8B → Qwen3-VL<br>
                and show each reasoning step in real time.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with results_container:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; color: #6b7280;">
            <p style="font-size: 14px;">Results will appear here after analysis.</p>
            <p style="font-size: 12px; color: #4b5563;">
                Supported inputs:<br>
                <b>ECG:</b> .hea+.dat (WFDB/PTB-XL) · .edf · .csv · .npy · .xml<br>
                <b>MRI/CT:</b> .dcm (single file or multi-slice series)
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Previous Results ──
elif st.session_state.agent_state and not st.session_state.is_running:
    state = st.session_state.agent_state

    with thinking_container:
        st.markdown(f"**Last run:** {state.patient_id} ({state.elapsed_s}s)")
        for step in state.thinking_steps:
            render_thinking_step(step)

    with results_container:
        st.markdown("*Showing results from previous run. Upload new files and click Run to analyze again.*")
        if state.final_report:
            st.markdown(state.final_report)