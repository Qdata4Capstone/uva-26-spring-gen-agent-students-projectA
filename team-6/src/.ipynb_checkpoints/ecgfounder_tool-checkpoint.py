"""
ECGFounder MCP Tool
====================
Wraps ECGFounder as an MCP-compatible tool.
Handles: raw ECG (.hea/.edf/.csv/.npy/.xml) → preprocess → classify → structured output

Since ECGFounder is ~90M params, it runs on CPU within this process.
No separate GPU server needed.
"""

import json
import os
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ECGFounderTool:
    """
    Complete ECG analysis pipeline:
    1. Read any ECG format
    2. Preprocess to (12, 5000) normalized numpy
    3. (Optional) Run ECGFounder classification
    4. Run NeuroKit2 signal analysis
    5. Generate 12-lead waveform image
    6. Return structured findings
    """

    STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                      "V1", "V2", "V3", "V4", "V5", "V6"]
    REQUIRED_FS = 500
    REQUIRED_LEN = 5000  # 10s × 500Hz

    def __init__(self, checkpoint_path: Optional[str] = None, output_dir: str = "/tmp/cardioagent"):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._model = None

    # ── Public API (MCP interface) ──

    def analyze(self, ecg_path: str, patient_id: str = "unknown") -> dict:
        """
        Full analysis pipeline. Returns dict with:
        - findings: list of detected abnormalities
        - metrics: HR, QRS, rhythm, etc.
        - waveform_image: path to 12-lead PNG
        - raw_predictions: ECGFounder 150-class probabilities (if model loaded)
        """
        result = {"patient_id": patient_id, "ecg_path": ecg_path, "steps": []}

        # Step 1: Read ECG
        try:
            signal, fs, leads = self._read_ecg(ecg_path)
            result["steps"].append({
                "step": "read_ecg",
                "status": "success",
                "detail": f"Read {signal.shape[1]} samples, {signal.shape[0]} leads, {fs}Hz"
            })
        except Exception as e:
            result["steps"].append({"step": "read_ecg", "status": "error", "detail": str(e)})
            result["error"] = f"Failed to read ECG: {e}"
            return result

        # Step 2: Preprocess
        try:
            processed = self._preprocess(signal, fs, leads)
            result["steps"].append({
                "step": "preprocess",
                "status": "success",
                "detail": f"Resampled to 500Hz, normalized, shape: {processed.shape}"
            })
        except Exception as e:
            result["steps"].append({"step": "preprocess", "status": "error", "detail": str(e)})
            processed = None

        # Step 3: Signal analysis (NeuroKit2)
        try:
            metrics = self._analyze_signal(signal, fs, leads)
            result["metrics"] = metrics
            result["steps"].append({
                "step": "signal_analysis",
                "status": "success",
                "detail": f"HR: {metrics.get('heart_rate_bpm')} bpm, Rhythm: {metrics.get('rhythm')}"
            })
        except Exception as e:
            result["metrics"] = {"error": str(e)}
            result["steps"].append({"step": "signal_analysis", "status": "error", "detail": str(e)})

        # Step 4: ECGFounder classification (if model available)
        if processed is not None and self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                predictions = self._run_ecgfounder(processed)
                result["raw_predictions"] = predictions
                result["steps"].append({
                    "step": "ecgfounder_classify",
                    "status": "success",
                    "detail": f"{predictions.get('n_abnormalities', 0)} abnormalities detected"
                })
            except Exception as e:
                result["steps"].append({"step": "ecgfounder_classify", "status": "error", "detail": str(e)})
        else:
            result["steps"].append({
                "step": "ecgfounder_classify",
                "status": "skipped",
                "detail": "ECGFounder checkpoint not loaded — using signal analysis only"
            })

        # Step 5: Generate waveform image
        try:
            img_path = self._plot_waveform(signal, fs, leads, patient_id, result.get("metrics", {}))
            result["waveform_image"] = img_path
            result["steps"].append({
                "step": "plot_waveform",
                "status": "success",
                "detail": f"Saved to {img_path}"
            })
        except Exception as e:
            result["steps"].append({"step": "plot_waveform", "status": "error", "detail": str(e)})

        # Step 6: Compile findings
        result["findings"] = self._compile_findings(result)

        return result

    # ── Reading ──

    def _read_ecg(self, path: str):
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix in (".hea", ".dat") or p.with_suffix(".hea").exists():
            return self._read_wfdb(path)
        elif suffix == ".edf":
            return self._read_edf(path)
        elif suffix == ".csv":
            return self._read_csv(path)
        elif suffix == ".npy":
            return self._read_npy(path)
        elif suffix == ".xml":
            return self._read_xml(path)
        else:
            # Try WFDB first, then CSV
            try:
                return self._read_wfdb(path)
            except Exception:
                return self._read_csv(path)

    def _read_wfdb(self, path):
        import wfdb
        rec = wfdb.rdrecord(str(Path(path).with_suffix("")))
        return rec.p_signal.T, rec.fs, rec.sig_name

    def _read_edf(self, path):
        import pyedflib
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        sigs = np.array([f.readSignal(i) for i in range(n)])
        leads = [f.getLabel(i).strip() for i in range(n)]
        fs = int(f.getSampleFrequency(0))
        f.close()
        return sigs, fs, leads

    def _read_csv(self, path):
        import pandas as pd
        df = pd.read_csv(path)
        return df.values.T, 500, list(df.columns)

    def _read_npy(self, path):
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
        leads = self.STANDARD_LEADS[:arr.shape[0]]
        return arr, 500, leads

    def _read_xml(self, path):
        import xml.etree.ElementTree as ET
        import base64
        tree = ET.parse(path)
        root = tree.getroot()
        lead_data = {}
        for lead_el in root.iter("LeadData"):
            name = lead_el.findtext("LeadID", "")
            b64 = lead_el.findtext("WaveFormData", "")
            if b64 and name:
                raw = base64.b64decode(b64)
                lead_data[name] = np.frombuffer(raw, dtype=np.int16).astype(float)
        leads = list(lead_data.keys())
        max_len = max(len(v) for v in lead_data.values())
        signal = np.zeros((len(leads), max_len))
        for i, name in enumerate(leads):
            signal[i, :len(lead_data[name])] = lead_data[name]
        return signal, 500, leads

    # ── Preprocessing ──

    def _preprocess(self, signal, fs, leads):
        from scipy.signal import resample

        # Resample to 500Hz
        if fs != self.REQUIRED_FS:
            new_len = int(signal.shape[1] * self.REQUIRED_FS / fs)
            signal = np.array([resample(signal[i], new_len) for i in range(signal.shape[0])])

        # Reorder to standard 12-lead
        name_map = {n.strip().upper().replace("LEAD ", ""): i for i, n in enumerate(leads)}
        result = np.zeros((12, signal.shape[1]))
        for i, std_name in enumerate(self.STANDARD_LEADS):
            key = std_name.upper()
            if key in name_map:
                result[i] = signal[name_map[key]]

        # Crop/pad to 5000 samples
        if result.shape[1] >= self.REQUIRED_LEN:
            result = result[:, :self.REQUIRED_LEN]
        else:
            padded = np.zeros((12, self.REQUIRED_LEN))
            padded[:, :result.shape[1]] = result
            result = padded

        # Z-score normalize per lead
        for i in range(12):
            std = np.std(result[i])
            if std > 1e-6:
                result[i] = (result[i] - np.mean(result[i])) / std

        return result

    # ── Signal Analysis ──

    def _analyze_signal(self, signal, fs, leads):
        import neurokit2 as nk

        # Use Lead II for primary analysis
        lead_ii_idx = next(
            (i for i, n in enumerate(leads) if n.strip().upper() in ("II", "LEAD II")),
            min(1, signal.shape[0] - 1)
        )
        lead_ii = signal[lead_ii_idx]

        cleaned = nk.ecg_clean(lead_ii, sampling_rate=fs)
        _, info = nk.ecg_process(cleaned, sampling_rate=fs)

        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) < 2:
            return {"heart_rate_bpm": None, "rhythm": "undetermined", "n_beats": len(r_peaks)}

        rr = np.diff(r_peaks) / fs * 1000  # ms
        hr = 60000 / np.mean(rr)

        metrics = {
            "heart_rate_bpm": round(float(hr), 1),
            "rr_mean_ms": round(float(np.mean(rr)), 1),
            "rr_std_ms": round(float(np.std(rr)), 1),
            "n_beats": len(r_peaks),
            "duration_s": round(signal.shape[1] / fs, 1),
            "rhythm": "regular" if np.std(rr) < 120 else "irregular",
        }

        # Classify rhythm
        if hr > 100:
            metrics["rhythm_detail"] = "Sinus tachycardia" if metrics["rhythm"] == "regular" else "Tachyarrhythmia"
        elif hr < 60:
            metrics["rhythm_detail"] = "Sinus bradycardia" if metrics["rhythm"] == "regular" else "Bradyarrhythmia"
        else:
            metrics["rhythm_detail"] = "Normal sinus rhythm" if metrics["rhythm"] == "regular" else "Irregular rhythm"

        # Per-lead ST analysis
        st_findings = []
        for i, lead_name in enumerate(leads[:12]):
            try:
                lead_sig = signal[i]
                c = nk.ecg_clean(lead_sig, sampling_rate=fs)
                _, rpks = nk.ecg_peaks(c, sampling_rate=fs)
                peaks = rpks.get("ECG_R_Peaks", [])
                if len(peaks) < 3:
                    continue

                st_devs = []
                for r in peaks[1:-1]:
                    st_pt = r + int(0.08 * fs)
                    bl_pt = r - int(0.04 * fs)
                    if 0 < bl_pt and st_pt < len(c):
                        st_devs.append(c[st_pt] - c[bl_pt])

                if st_devs:
                    mean_st = float(np.mean(st_devs))
                    if abs(mean_st) > 0.1:
                        st_findings.append({
                            "lead": lead_name.strip(),
                            "st_deviation_mv": round(mean_st, 3),
                            "type": "elevation" if mean_st > 0 else "depression",
                        })
            except Exception:
                continue

        metrics["st_findings"] = st_findings
        return metrics

    # ── ECGFounder Inference ──

    def _run_ecgfounder(self, processed_signal):
        import torch

        if self._model is None:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            # Adapt based on ECGFounder's actual model class
            # from ecgfounder.model import ECGFounderModel
            # self._model = ECGFounderModel(num_classes=150)
            # self._model.load_state_dict(checkpoint)
            # self._model.eval()
            raise NotImplementedError("Load ECGFounder model from checkpoint here")

        tensor = torch.from_numpy(processed_signal).float().unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.sigmoid(logits).numpy()[0]

        top_indices = np.argsort(probs)[::-1][:10]
        return {
            "top_findings": [
                {"class_idx": int(i), "probability": round(float(probs[i]), 3)}
                for i in top_indices if probs[i] > 0.3
            ],
            "n_abnormalities": int(np.sum(probs > 0.5)),
        }

    # ── Visualization ──

    def _plot_waveform(self, signal, fs, leads, patient_id, metrics):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_leads = min(12, signal.shape[0])
        duration = min(10, signal.shape[1] / fs)
        n_samp = int(duration * fs)
        t = np.arange(n_samp) / fs

        # Find abnormal leads from ST analysis
        abnormal_leads = set()
        for st in metrics.get("st_findings", []):
            abnormal_leads.add(st["lead"])

        fig, axes = plt.subplots(n_leads, 1, figsize=(14, n_leads * 1.1), facecolor="#0d1117")

        for i in range(n_leads):
            ax = axes[i]
            ax.set_facecolor("#0d1117")
            lead_name = leads[i].strip() if i < len(leads) else f"Lead {i+1}"

            is_abnormal = lead_name in abnormal_leads
            color = "#ff6b6b" if is_abnormal else "#4ecdc4"
            lw = 1.2 if is_abnormal else 0.7

            ax.plot(t, signal[i, :n_samp], color=color, linewidth=lw)
            ax.set_ylabel(lead_name, color="#e6edf3", fontsize=8, rotation=0, labelpad=30)
            ax.set_xlim(0, duration)
            ax.tick_params(colors="#484f58", labelsize=6)
            ax.grid(True, alpha=0.15, color="#f85149", linewidth=0.3)

            for spine in ax.spines.values():
                spine.set_color("#21262d")

            if is_abnormal:
                st_info = next((s for s in metrics.get("st_findings", []) if s["lead"] == lead_name), None)
                if st_info:
                    label = f"ST {st_info['type']} {st_info['st_deviation_mv']:.2f}mV"
                    ax.text(0.98, 0.85, label, transform=ax.transAxes,
                            fontsize=6, ha="right", color="#ff6b6b",
                            fontweight="bold", fontstyle="italic")

        axes[-1].set_xlabel("Time (s)", color="#8b949e", fontsize=8)

        hr = metrics.get("heart_rate_bpm", "?")
        rhythm = metrics.get("rhythm_detail", "?")
        fig.suptitle(
            f"12-Lead ECG — {patient_id}  |  HR: {hr} bpm  |  {rhythm}",
            color="#58a6ff", fontsize=10, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        path = os.path.join(self.output_dir, f"ecg_{patient_id}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        return path

    # ── Compile Findings ──

    def _compile_findings(self, result):
        findings = []
        metrics = result.get("metrics", {})

        # Rhythm findings
        rhythm = metrics.get("rhythm_detail", "")
        if "tachycardia" in rhythm.lower():
            findings.append({"finding": rhythm, "severity": "moderate", "confidence": 0.85})
        elif "bradycardia" in rhythm.lower():
            findings.append({"finding": rhythm, "severity": "mild", "confidence": 0.80})
        elif "irregular" in rhythm.lower():
            findings.append({"finding": rhythm, "severity": "moderate", "confidence": 0.75})
        else:
            findings.append({"finding": rhythm, "severity": "normal", "confidence": 0.90})

        # ST findings
        for st in metrics.get("st_findings", []):
            sev = "severe" if abs(st["st_deviation_mv"]) > 0.2 else "moderate"
            findings.append({
                "finding": f"ST {st['type']} in {st['lead']} ({st['st_deviation_mv']:.2f} mV)",
                "severity": sev,
                "confidence": min(0.9, 0.5 + abs(st["st_deviation_mv"])),
                "lead": st["lead"],
            })

        return findings


# ── MCP Server ──

def create_mcp_server():
    """Create MCP server instance for ECGFounder tool"""
    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("ecgfounder")
    tool = ECGFounderTool(output_dir="/tmp/cardioagent")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="analyze_ecg",
                description=(
                    "Analyze ECG recording. Accepts raw formats: WFDB (.hea), EDF, CSV, numpy, XML. "
                    "Returns: heart rate, rhythm, ST analysis, waveform image, structured findings."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ecg_path": {"type": "string", "description": "Path to ECG file"},
                        "patient_id": {"type": "string", "default": "unknown"},
                    },
                    "required": ["ecg_path"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "analyze_ecg":
            result = tool.analyze(arguments["ecg_path"], arguments.get("patient_id", "unknown"))
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server