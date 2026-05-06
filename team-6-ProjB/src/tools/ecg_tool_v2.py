"""
ECG Tool v2 — Robust ECG Analysis
===================================
Fixes:
  1. Handles .dat files (auto-finds matching .hea)
  2. Robust signal analysis (no more division by zero)
  3. Works with MIMIC-IV, PTB-XL, Symile-MIMIC formats
  
pip install wfdb scipy matplotlib numpy
(neurokit2 is optional — we use scipy as fallback)
"""

import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Optional
from scipy.signal import find_peaks, resample, butter, filtfilt

logger = logging.getLogger(__name__)

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGTool:
    def __init__(self, output_dir: str = "/tmp/cardioagent"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze(self, ecg_path: str, patient_id: str = "unknown") -> dict:
        result = {"patient_id": patient_id, "ecg_path": ecg_path, "steps": []}

        # Step 1: Fix path — if .dat given, find .hea
        ecg_path = self._fix_path(ecg_path)
        
        # Step 2: Read ECG
        try:
            signal, fs, leads = self._read_ecg(ecg_path)
            result["steps"].append({
                "step": "read_ecg", "status": "success",
                "detail": f"Read {signal.shape[1]} samples, {signal.shape[0]} leads, {fs}Hz"
            })
        except Exception as e:
            result["steps"].append({"step": "read_ecg", "status": "error", "detail": str(e)})
            result["error"] = str(e)
            return result

        # Step 3: Signal analysis (robust, no neurokit2 dependency)
        try:
            metrics = self._analyze_signal(signal, fs, leads)
            result["metrics"] = metrics
            result["steps"].append({
                "step": "signal_analysis", "status": "success",
                "detail": f"HR: {metrics.get('heart_rate_bpm', '?')} bpm, "
                          f"Rhythm: {metrics.get('rhythm_detail', '?')}, "
                          f"{len(metrics.get('st_findings', []))} ST abnormalities"
            })
        except Exception as e:
            logger.warning(f"Signal analysis failed: {e}")
            result["metrics"] = self._basic_metrics(signal, fs)
            result["steps"].append({
                "step": "signal_analysis", "status": "success",
                "detail": f"Basic analysis only: {result['metrics'].get('heart_rate_bpm', '?')} bpm"
            })

        # Step 4: Plot waveform
        try:
            img_path = self._plot_waveform(signal, fs, leads, patient_id, result.get("metrics", {}))
            result["waveform_image"] = img_path
            result["steps"].append({
                "step": "plot_waveform", "status": "success",
                "detail": f"Saved to {img_path}"
            })
        except Exception as e:
            result["steps"].append({"step": "plot_waveform", "status": "error", "detail": str(e)})

        # Step 5: Compile findings
        result["findings"] = self._compile_findings(result.get("metrics", {}))
        return result

    # ── Path Fixing ──

    def _fix_path(self, path: str) -> str:
        """
        Fix ECG file path for different formats.
        WFDB needs path without extension. Other formats pass through as-is.
        """
        p = Path(path)

        # Non-WFDB formats: return as-is
        if p.suffix in (".npy", ".csv", ".edf", ".xml"):
            return str(p)

        # .dat given → strip to base for WFDB
        if p.suffix == ".dat":
            return str(p.with_suffix(""))

        # .hea given → strip to base for WFDB
        if p.suffix == ".hea":
            return str(p.with_suffix(""))

        # No extension → check if .hea exists (WFDB)
        if not p.suffix:
            if p.with_suffix(".hea").exists():
                return str(p)
            # Maybe it's actually .npy without extension?
            if p.with_suffix(".npy").exists():
                return str(p.with_suffix(".npy"))

        # Directory → find first .hea
        if p.is_dir():
            hea_files = list(p.glob("*.hea"))
            if hea_files:
                return str(hea_files[0].with_suffix(""))

        return str(p)

    # ── Reading ──

    def _read_ecg(self, path: str):
        """
        Read ECG from any format. Checks extension FIRST to avoid
        unnecessary WFDB attempts on non-WFDB files.
        """
        p = Path(path)

        # ── Route by extension first ──

        # .npy — numpy array (from Symile-MIMIC, preprocessed data)
        if p.suffix == ".npy":
            return self._read_npy(str(p))

        # .csv — columnar data
        if p.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(str(p))
            return df.values.T, 500, list(df.columns)

        # .edf — European Data Format
        if p.suffix == ".edf":
            return self._read_edf(str(p))

        # .hea / .dat / no extension — WFDB format
        try:
            import wfdb
            record = wfdb.rdrecord(str(Path(path).with_suffix("")))
            signal = record.p_signal  # (samples, leads)

            # Handle NaN values (common in MIMIC-IV)
            if np.any(np.isnan(signal)):
                signal = np.nan_to_num(signal, nan=0.0)

            return signal.T, record.fs, record.sig_name
        except Exception as e:
            logger.debug(f"WFDB read failed: {e}")

        # Last resort: try as numpy regardless of extension
        try:
            return self._read_npy(str(p))
        except Exception:
            pass

        raise ValueError(f"Cannot read ECG from {path}")

    def _read_npy(self, path: str):
        """
        Read ECG from .npy file.
        Handles shapes from various sources:
          (12, 5000)       — standard 12-lead
          (5000, 12)       — transposed
          (1, 5000, 12)    — Symile-MIMIC (squeezed batch dim)
          (N,)             — single lead
        """
        arr = np.load(path)

        # Handle NaN
        if np.any(np.isnan(arr)):
            arr = np.nan_to_num(arr, nan=0.0)

        # Squeeze leading batch dims: (1, 5000, 12) → (5000, 12)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        # Single lead
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Ensure shape is (leads, samples) — leads should be the smaller dim
        if arr.ndim == 2:
            if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 12:
                arr = arr.T  # (5000, 12) → (12, 5000)
            elif arr.shape[0] > 12 and arr.shape[1] > 12:
                # Ambiguous — assume (samples, leads) if shape[1] is closer to 12
                if abs(arr.shape[1] - 12) < abs(arr.shape[0] - 12):
                    arr = arr.T

        n_leads = arr.shape[0]
        return arr.astype(np.float32), 500, STANDARD_LEADS[:n_leads]

    def _read_edf(self, path: str):
        """Read EDF format ECG"""
        import pyedflib
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        sigs = np.array([f.readSignal(i) for i in range(n)])
        leads = [f.getLabel(i).strip() for i in range(n)]
        fs = int(f.getSampleFrequency(0))
        f.close()
        return sigs, fs, leads

    # ── Robust Signal Analysis ──

    def _analyze_signal(self, signal, fs, leads):
        """
        Robust ECG analysis using scipy only.
        No neurokit2 dependency — avoids division by zero issues.
        """
        # Use Lead II for primary analysis (index 1 if available)
        lead_ii_idx = self._find_lead_index(leads, "II")
        lead_ii = signal[lead_ii_idx]

        # Bandpass filter (0.5-40 Hz)
        lead_ii_filtered = self._bandpass_filter(lead_ii, fs, low=0.5, high=40.0)

        # R-peak detection
        r_peaks = self._detect_r_peaks(lead_ii_filtered, fs)

        # Compute metrics
        metrics = {}

        if len(r_peaks) >= 2:
            rr_intervals = np.diff(r_peaks) / fs * 1000  # ms

            # Remove physiologically impossible RR intervals
            valid_rr = rr_intervals[(rr_intervals > 200) & (rr_intervals < 3000)]

            if len(valid_rr) > 0:
                mean_rr = float(np.mean(valid_rr))
                std_rr = float(np.std(valid_rr))
                hr = 60000.0 / mean_rr

                metrics["heart_rate_bpm"] = round(hr, 1)
                metrics["rr_mean_ms"] = round(mean_rr, 1)
                metrics["rr_std_ms"] = round(std_rr, 1)
                metrics["n_beats"] = len(r_peaks)
                metrics["rhythm"] = "regular" if std_rr < 120 else "irregular"

                # Rhythm classification
                if hr > 100:
                    metrics["rhythm_detail"] = "Sinus tachycardia" if metrics["rhythm"] == "regular" else "Tachyarrhythmia"
                elif hr < 60:
                    metrics["rhythm_detail"] = "Sinus bradycardia" if metrics["rhythm"] == "regular" else "Bradyarrhythmia"
                else:
                    metrics["rhythm_detail"] = "Normal sinus rhythm" if metrics["rhythm"] == "regular" else "Irregular rhythm"
            else:
                metrics.update(self._basic_metrics(signal, fs))
        else:
            metrics.update(self._basic_metrics(signal, fs))

        metrics["duration_s"] = round(signal.shape[1] / fs, 1)
        metrics["sampling_rate"] = fs
        metrics["n_leads"] = signal.shape[0]

        # Per-lead ST analysis
        metrics["st_findings"] = self._analyze_st_segments(signal, fs, leads, r_peaks)

        return metrics

    def _basic_metrics(self, signal, fs) -> dict:
        """Fallback metrics when R-peak detection fails"""
        return {
            "heart_rate_bpm": None,
            "rr_mean_ms": None,
            "rhythm": "undetermined",
            "rhythm_detail": "Unable to determine (R-peak detection failed)",
            "n_beats": 0,
            "duration_s": round(signal.shape[1] / fs, 1),
            "st_findings": [],
        }

    def _find_lead_index(self, leads, target="II") -> int:
        """Find index of a specific lead name"""
        for i, name in enumerate(leads):
            if name.strip().upper().replace("LEAD ", "") == target.upper():
                return i
        # Fallback: lead II is usually index 1
        return min(1, len(leads) - 1)

    def _bandpass_filter(self, signal, fs, low=0.5, high=40.0) -> np.ndarray:
        """Apply bandpass filter to remove baseline wander and noise"""
        try:
            nyquist = fs / 2
            low_norm = max(low / nyquist, 0.001)
            high_norm = min(high / nyquist, 0.999)
            b, a = butter(3, [low_norm, high_norm], btype='band')
            return filtfilt(b, a, signal)
        except Exception:
            return signal

    def _detect_r_peaks(self, signal, fs) -> np.ndarray:
        """
        Robust R-peak detection using scipy.
        Much more reliable than neurokit2 on noisy signals.
        """
        # Normalize
        sig = signal - np.mean(signal)
        std = np.std(sig)
        if std < 1e-10:
            return np.array([])
        sig = sig / std

        # Minimum distance between R-peaks: 200ms (300 bpm max)
        min_distance = int(0.2 * fs)

        # Height threshold: at least 0.3 std above mean
        # Try multiple thresholds if first attempt finds too few peaks
        for height_thresh in [0.5, 0.3, 0.2, 0.1]:
            peaks, properties = find_peaks(
                sig,
                height=height_thresh,
                distance=min_distance,
                prominence=0.1,
            )
            # Expect at least 30 bpm → at least signal_duration/2 peaks
            expected_min = max(2, int(len(signal) / fs / 2))
            if len(peaks) >= expected_min:
                return peaks

        # Last resort: just find the tallest peaks
        peaks, _ = find_peaks(sig, distance=min_distance)
        if len(peaks) > 0:
            return peaks

        return np.array([])

    def _analyze_st_segments(self, signal, fs, leads, r_peaks) -> list:
        """Analyze ST segment deviation per lead"""
        st_findings = []

        if len(r_peaks) < 3:
            return st_findings

        n_leads = min(12, signal.shape[0])
        for i in range(n_leads):
            lead_name = leads[i].strip() if i < len(leads) else f"Lead_{i}"
            lead_sig = signal[i]

            try:
                # Filter this lead
                filtered = self._bandpass_filter(lead_sig, fs, 0.5, 40)

                st_devs = []
                for r in r_peaks[1:-1]:
                    # J-point: ~60-80ms after R-peak
                    j_point = r + int(0.08 * fs)
                    # Baseline: ~40ms before R-peak (PR segment)
                    baseline_point = r - int(0.04 * fs)

                    if baseline_point > 0 and j_point < len(filtered):
                        st_dev = filtered[j_point] - filtered[baseline_point]
                        st_devs.append(st_dev)

                if len(st_devs) >= 2:
                    mean_st = float(np.median(st_devs))  # median is more robust
                    if abs(mean_st) > 0.1:
                        st_findings.append({
                            "lead": lead_name,
                            "st_deviation_mv": round(mean_st, 3),
                            "type": "elevation" if mean_st > 0 else "depression",
                        })
            except Exception:
                continue

        return st_findings

    # ── Visualization ──

    def _plot_waveform(self, signal, fs, leads, patient_id, metrics):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_leads = min(12, signal.shape[0])
        duration = min(10, signal.shape[1] / fs)
        n_samp = int(duration * fs)
        t = np.arange(n_samp) / fs

        abnormal_leads = {s["lead"] for s in metrics.get("st_findings", [])}

        fig, axes = plt.subplots(n_leads, 1, figsize=(14, n_leads * 1.1), facecolor="#0d1117")

        for i in range(n_leads):
            ax = axes[i] if n_leads > 1 else axes
            ax.set_facecolor("#0d1117")
            lead_name = leads[i].strip() if i < len(leads) else f"Lead {i+1}"

            is_abnormal = lead_name in abnormal_leads
            color = "#ff6b6b" if is_abnormal else "#4ecdc4"
            lw = 1.2 if is_abnormal else 0.7

            sig_plot = signal[i, :n_samp]
            ax.plot(t, sig_plot, color=color, linewidth=lw)
            ax.set_ylabel(lead_name, color="#e6edf3", fontsize=8, rotation=0, labelpad=30)
            ax.set_xlim(0, duration)
            ax.tick_params(colors="#484f58", labelsize=6)
            ax.grid(True, alpha=0.15, color="#f85149", linewidth=0.3)
            for spine in ax.spines.values():
                spine.set_color("#21262d")

            if is_abnormal:
                st_info = next((s for s in metrics.get("st_findings", []) if s["lead"] == lead_name), None)
                if st_info:
                    ax.text(0.98, 0.85, f"ST {st_info['type']} {st_info['st_deviation_mv']:.2f}mV",
                            transform=ax.transAxes, fontsize=6, ha="right", color="#ff6b6b", fontweight="bold")

        last_ax = axes[-1] if n_leads > 1 else axes
        last_ax.set_xlabel("Time (s)", color="#8b949e", fontsize=8)

        hr = metrics.get("heart_rate_bpm", "?")
        rhythm = metrics.get("rhythm_detail", "?")
        fig.suptitle(f"12-Lead ECG — {patient_id}  |  HR: {hr} bpm  |  {rhythm}",
                     color="#58a6ff", fontsize=10, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        path = os.path.join(self.output_dir, f"ecg_{patient_id}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        return path

    # ── Compile Findings ──

    def _compile_findings(self, metrics) -> list:
        findings = []
        rhythm = metrics.get("rhythm_detail", "")

        if "tachycardia" in rhythm.lower():
            findings.append({"finding": rhythm, "severity": "moderate", "confidence": 0.85, "modality": "ecg"})
        elif "bradycardia" in rhythm.lower():
            findings.append({"finding": rhythm, "severity": "mild", "confidence": 0.80, "modality": "ecg"})
        elif "irregular" in rhythm.lower():
            findings.append({"finding": rhythm, "severity": "moderate", "confidence": 0.75, "modality": "ecg"})
        elif rhythm and "unable" not in rhythm.lower():
            findings.append({"finding": rhythm, "severity": "normal", "confidence": 0.90, "modality": "ecg"})

        for st in metrics.get("st_findings", []):
            sev = "severe" if abs(st["st_deviation_mv"]) > 0.2 else "moderate"
            findings.append({
                "finding": f"ST {st['type']} in {st['lead']} ({st['st_deviation_mv']:.2f} mV)",
                "severity": sev,
                "confidence": min(0.9, 0.5 + abs(st["st_deviation_mv"])),
                "lead": st["lead"],
                "modality": "ecg",
            })

        if not findings:
            findings.append({"finding": "ECG analysis inconclusive", "severity": "unknown", "modality": "ecg"})

        return findings