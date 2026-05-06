#!/usr/bin/env python3
"""GPU Cluster Monitor - Multi-cluster SLURM GPU node dashboard."""

import base64
import json
import sqlite3
import subprocess
import re
import os
import secrets
import shlex
import sys
import time
import threading
import urllib.request
from datetime import datetime
from flask import Flask, jsonify, Response, request
from openai import OpenAI

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "history.db")
AGENT_LOG_PATH = os.path.join(LOG_DIR, "agent_actions.log")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config():
    """Load config.json from SCRIPT_DIR. Returns empty dict if missing/invalid."""
    cfg_path = os.path.join(SCRIPT_DIR, "config.json")
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: config.json parse error: {e}", file=sys.stderr)
        return {}

CONFIG = load_config()

# ---------------------------------------------------------------------------
# Cluster definitions
# ---------------------------------------------------------------------------

def _clusters_from_config(cfg):
    """Build CLUSTERS dict from config.json, falling back to safe defaults."""
    clusters_cfg = cfg.get("clusters", {})
    result = {}
    cs_cfg = clusters_cfg.get("cs", {})
    result["cs"] = {
        "name": cs_cfg.get("name", "CS"),
        "ssh": cs_cfg.get("ssh"),
        "account": cs_cfg.get("account"),
    }
    if "hpc" in clusters_cfg:
        hpc_cfg = clusters_cfg["hpc"]
        result["hpc"] = {
            "name": hpc_cfg.get("name", "HPC"),
            "ssh": hpc_cfg.get("ssh"),
            "account": hpc_cfg.get("account"),
        }
    return result

CLUSTERS = _clusters_from_config(CONFIG)

# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

SEPARATOR = "###SECTION###"


def run_cmd(cmd, ssh_prefix=None):
    """Run a shell command (optionally via SSH) and return stdout lines."""
    if ssh_prefix:
        escaped = cmd.replace("'", "'\\''")
        cmd = f"{ssh_prefix} '{escaped}'"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception as e:
        print(f"Error running command: {cmd} -> {e}", file=sys.stderr)
        return []


def run_batched(commands, ssh_prefix=None):
    """Run multiple commands in one shell (one SSH connection) separated by markers."""
    joined = f"\necho '{SEPARATOR}'\n".join(commands)
    lines = run_cmd(joined, ssh_prefix)
    sections = []
    current = []
    for line in lines:
        if line.strip() == SEPARATOR:
            sections.append(current)
            current = []
        else:
            current.append(line)
    sections.append(current)
    return sections


def expand_nodelist(nodelist_str):
    """Expand SLURM nodelist like 'node[01-03,05],other' into individual names.
    Pure Python implementation - no subprocess calls."""
    results = []
    # Split on commas that are NOT inside brackets
    depth = 0
    current = ""
    for ch in nodelist_str + ",":
        if ch == "[":
            depth += 1
            current += ch
        elif ch == "]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            if current:
                results.extend(_expand_single(current))
            current = ""
        else:
            current += ch
    return results


def _expand_single(token):
    """Expand a single token like 'node[01-03,05]' or just 'node01'."""
    m = re.match(r"^(.+?)\[(.+)\](.*)$", token)
    if not m:
        return [token]
    prefix, ranges, suffix = m.group(1), m.group(2), m.group(3)
    nodes = []
    for part in ranges.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            width = len(lo)
            for i in range(int(lo), int(hi) + 1):
                nodes.append(f"{prefix}{str(i).zfill(width)}{suffix}")
        else:
            nodes.append(f"{prefix}{part}{suffix}")
    return nodes


def normalize_gpu_name(raw):
    """Normalize raw GRES GPU name to a clean, consistent display name."""
    s = raw.lower().strip()
    # --- A100 special handling ---
    if "a100" in s:
        if "40g" in s:
            return "A100 40GB"
        if "80g" in s:
            return "A100 80GB"
        return "A100"  # will be refined later using AvailableFeatures
    # --- Strip common prefixes ---
    for prefix in ("nvidia_geforce_", "nvidia_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # --- Strip PCIe / memory suffixes (e.g. -pcie-12gb) ---
    s = re.sub(r"-pcie-\d+gb", "", s)
    # --- Specific model clean-ups ---
    name_map = {
        "h100_nvl": "H100 NVL", "h100": "H100",
        "h200": "H200",
        "rtx_5090": "RTX 5090", "rtx_5080": "RTX 5080",
        "a6000": "A6000",
        "a40": "A40",
        "a16": "A16",
        "rtx_a4500": "RTX A4500",
        "rtx_4000_ada_generation": "RTX 4000 Ada",
        "rtx_a4000": "RTX A4000",
        "quadro_rtx_6000": "Quadro RTX 6000",
        "quadro_rtx_4000": "Quadro RTX 4000",
        "v100": "V100",
        "rtx_3090": "RTX 3090", "rtx3090": "RTX 3090",
        "rtx_2080_ti": "RTX 2080 Ti",
        "rtx2080": "RTX 2080", "rtx_2080": "RTX 2080",
        "titan_xp": "Titan Xp", "titan_x": "Titan X",
        "gtx_1080_ti": "GTX 1080 Ti",
        "gtx_1080": "GTX 1080",
        "tesla_p100": "Tesla P100",
    }
    if s in name_map:
        return name_map[s]
    # --- Fallback: replace _ with space, uppercase ---
    return s.replace("_", " ").upper()


def parse_gres(gres_str):
    """Parse GRES string like 'gpu:nvidia_a100:4(S:0-1)'.
    Returns (display_name, gpu_count, slurm_gres_name).
    slurm_gres_name is the raw token used in --gres=gpu:<slurm_gres_name>:<count>.
    """
    if not gres_str or gres_str == "(null)":
        return ("N/A", 0, "")
    m = re.match(r"gpu:([^:(]+)", gres_str)
    if m:
        raw = m.group(1)                          # e.g. "nvidia_a40", "a100_80gb"
        count_m = re.search(r":(\d+)", gres_str)
        count = int(count_m.group(1)) if count_m else 1
        return (normalize_gpu_name(raw), count, raw)
    m = re.match(r"gpu:(\d+)", gres_str)
    if m:
        return ("GPU", int(m.group(1)), "")
    return ("Unknown", 0, "")


def parse_sinfo_lines(lines):
    """Parse sinfo output into a dict of GPU nodes."""
    nodes = {}
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) < 8:
            continue
        nodename, partition, gres, cpus_aiot, mem_mb, state, free_mem, cpu_load = parts
        gpu_type, gpu_count, gpu_type_raw = parse_gres(gres)
        if gpu_count == 0:
            continue
        if nodename in nodes and nodes[nodename]["gpu_count"] >= gpu_count:
            continue
        state_raw = state.rstrip("*~#$-")
        # For compound states like MIXED+RESERVED, use the last (more specific) part
        state_clean = state_raw.split("+")[-1] if "+" in state_raw else state_raw
        try:
            mem_gb = round(int(mem_mb) / 1024, 1)
        except ValueError:
            mem_gb = 0
        try:
            free_mem_gb = round(int(free_mem) / 1024, 1)
            if free_mem_gb > mem_gb:
                free_mem_gb = mem_gb
        except ValueError:
            free_mem_gb = 0
        try:
            cpu_load_val = float(cpu_load)
        except ValueError:
            cpu_load_val = 0.0
        cpu_parts = cpus_aiot.split("/")
        if len(cpu_parts) == 4:
            cpus_alloc = int(cpu_parts[0])
            cpus_total = int(cpu_parts[3])
            cpus_free = cpus_total - cpus_alloc
        else:
            cpus_total = int(cpus_aiot) if cpus_aiot.isdigit() else 0
            cpus_free = cpus_total

        nodes[nodename] = {
            "node": nodename,
            "partition": partition,
            "gpu_type": gpu_type,         # normalized display name, e.g. "A40"
            "gpu_type_raw": gpu_type_raw, # SLURM GRES token, e.g. "nvidia_a40"
            "gpu_count": gpu_count,
            "gpus_free": gpu_count,
            "cpus": cpus_total,
            "cpus_free": cpus_free,
            "mem_gb": mem_gb,
            "free_mem_gb": free_mem_gb,
            "cpu_load": cpu_load_val,
            "state": state_clean,
        }
    return nodes


def parse_squeue_lines(lines, gpu_node_names):
    """Parse squeue output into a dict of jobs by node.
    Only expand nodelists that overlap with known GPU nodes."""
    jobs_by_node = {}
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) < 8:
            continue
        job_id, partition, name, user, state, time_used, num_nodes, nodelist = [
            p.strip() for p in parts
        ]
        if state != "R" or not nodelist:
            continue
        job_info = {
            "job_id": job_id,
            "partition": partition,
            "name": name,
            "user": user,
            "state": state,
            "time": time_used,
            "nodelist": nodelist,
        }
        # Inline nodelist expansion (already expanded in batch)
        # We store raw nodelist for now; will be expanded below
        if nodelist not in jobs_by_node:
            jobs_by_node[nodelist] = []
        jobs_by_node[nodelist].append(job_info)
    return jobs_by_node


def parse_scontrol_lines(lines):
    """Parse scontrol show node output for GPU alloc, memory alloc, features, and state."""
    gpu_alloc = {}
    mem_alloc = {}  # AllocMem in MB
    features = {}   # AvailableFeatures per node
    node_state = {}  # Full compound state (e.g. MIXED+RESERVED)
    current_node = None
    for line in lines:
        m = re.match(r"NodeName=(\S+)", line.strip())
        if m:
            current_node = m.group(1)
        if current_node and "AllocMem=" in line:
            mm = re.search(r"AllocMem=(\d+)", line)
            if mm:
                mem_alloc[current_node] = int(mm.group(1))
        if current_node and re.search(r"\bState=", line):
            sm = re.search(r"State=(\S+)", line)
            if sm:
                raw = sm.group(1).rstrip("*~#$-")
                # For compound states like MIXED+RESERVED: show RESERVED
                # For MIXED+PLANNED or IDLE+NOT_RESPONDING: show MIXED or IDLE
                # Rule: ignore decorative suffixes, keep important ones
                _ignore_suffixes = {"PLANNED", "NOT_RESPONDING", "COMPLETING",
                                    "POWERING_UP", "POWERING_DOWN", "REBOOT_ISSUED"}
                parts = raw.split("+")
                # Pick the last non-ignorable part, fallback to first
                picked = parts[0]
                for p in parts:
                    if p not in _ignore_suffixes:
                        picked = p
                node_state[current_node] = picked
        if current_node and "AvailableFeatures=" in line:
            fm = re.search(r"AvailableFeatures=(\S+)", line)
            if fm:
                features[current_node] = fm.group(1).split(",")
        if current_node and "AllocTRES=" in line:
            gm = re.search(r"gres/gpu=(\d+)", line)
            gpu_alloc[current_node] = int(gm.group(1)) if gm else 0
    return gpu_alloc, mem_alloc, features, node_state


def get_cluster_data(cluster_id):
    """Collect all data for a given cluster in one batched call."""
    cluster = CLUSTERS.get(cluster_id)
    if not cluster:
        return {"error": "unknown cluster"}
    ssh = cluster["ssh"]

    # Step 1: batch sinfo + squeue + scontrol(GPU nodes only) in one call
    cmd_sinfo = 'sinfo -N -o "%N|%P|%G|%C|%m|%T|%e|%O" --noheader'
    cmd_squeue = 'squeue -t R -o "%.18i|%P|%.40j|%.10u|%.2t|%.12M|%.4D|%N" --noheader'
    # Only query scontrol for GPU nodes (extract from sinfo inline)
    cmd_scontrol = (
        'GPU_NODES=$(sinfo -N -o "%N|%G" --noheader '
        "| awk -F'|' '$2 ~ /gpu/' | cut -d'|' -f1 | sort -u | tr '\\n' ',') && "
        'scontrol show node ${GPU_NODES%,}'
    )

    sections = run_batched([cmd_sinfo, cmd_squeue, cmd_scontrol], ssh)
    if len(sections) < 3:
        # Pad missing sections
        sections += [[] for _ in range(3 - len(sections))]

    sinfo_lines, squeue_lines, scontrol_lines = sections[0], sections[1], sections[2]

    # Parse everything
    nodes = parse_sinfo_lines(sinfo_lines)
    gpu_alloc, mem_alloc, node_features, node_state = parse_scontrol_lines(scontrol_lines)

    # Build jobs_by_node: expand nodelists using scontrol hostname (local parse)
    # We parse squeue raw nodelists and map to individual nodes
    jobs_by_node = {}
    for line in squeue_lines:
        parts = line.strip().split("|")
        if len(parts) < 8:
            continue
        job_id, partition, name, user, state, time_used, num_nodes, nodelist = [
            p.strip() for p in parts
        ]
        if state != "R" or not nodelist:
            continue
        job_info = {
            "job_id": job_id,
            "partition": partition,
            "name": name,
            "user": user,
            "state": state,
            "time": time_used,
            "nodelist": nodelist,
        }
        expanded = expand_nodelist(nodelist)
        for node in expanded:
            if node in nodes:
                if node not in jobs_by_node:
                    jobs_by_node[node] = []
                jobs_by_node[node].append(job_info)

    # Merge
    node_list = []
    for name, info in sorted(nodes.items()):
        info["jobs"] = jobs_by_node.get(name, [])
        gpus_used = gpu_alloc.get(name, 0)
        info["gpus_free"] = info["gpu_count"] - gpus_used
        # Use SLURM AllocMem for accurate free memory (sinfo %e is unreliable)
        if name in mem_alloc:
            alloc_gb = round(mem_alloc[name] / 1024, 1)
            info["free_mem_gb"] = round(info["mem_gb"] - alloc_gb, 1)
            if info["free_mem_gb"] < 0:
                info["free_mem_gb"] = 0
        # Refine A100 name using AvailableFeatures (a100_40gb / a100_80gb)
        feats = node_features.get(name, [])
        if "a100" in info["gpu_type"].lower():
            if "a100_40gb" in feats:
                info["gpu_type"] = "A100 40GB"
            elif "a100_80gb" in feats:
                info["gpu_type"] = "A100 80GB"
        # Use scontrol State for accurate compound states (e.g. MIXED+RESERVED -> RESERVED)
        if name in node_state:
            info["state"] = node_state[name]
        node_list.append(info)

    total_gpus = sum(n["gpu_count"] for n in node_list)
    free_gpus = sum(n["gpus_free"] for n in node_list)
    total_cpus = sum(n["cpus"] for n in node_list)
    free_cpus = sum(n["cpus_free"] for n in node_list)

    return {
        "cluster": cluster_id,
        "cluster_name": cluster["name"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nodes": node_list,
        "total_gpu_nodes": len(node_list),
        "total_gpus": total_gpus,
        "free_gpus": free_gpus,
        "total_cpus": total_cpus,
        "free_cpus": free_cpus,
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/api/data/<cluster_id>")
def api_data(cluster_id):
    return jsonify(get_cluster_data(cluster_id))


@app.route("/api/validate_config")
def api_validate_config():
    """Check that config paths and environment are usable before launch."""
    errors = []
    result = {}

    script_dir = CONFIG.get("script_dir", SCRIPT_DIR)
    result["script_dir_exists"] = os.path.isdir(script_dir)
    if not result["script_dir_exists"]:
        errors.append(f"script_dir not found: {script_dir}")

    monitor_py = os.path.join(script_dir, "monitor.py")
    result["monitor_py_exists"] = os.path.isfile(monitor_py)
    if not result["monitor_py_exists"]:
        errors.append(f"monitor.py not found in script_dir: {monitor_py}")

    conda_env = CONFIG.get("conda_env", "")
    if conda_env:
        python_bin = os.path.join(conda_env, "bin", "python")
        result["conda_python_exists"] = os.path.isfile(python_bin)
        if not result["conda_python_exists"]:
            errors.append(f"conda python not found: {python_bin}")
        else:
            try:
                r = subprocess.run(
                    [python_bin, "-c", "import flask, openai"],
                    capture_output=True, text=True, timeout=10,
                )
                result["required_packages_ok"] = r.returncode == 0
                if r.returncode != 0:
                    errors.append(f"missing packages in conda env: {r.stderr.strip()}")
            except Exception as e:
                result["required_packages_ok"] = False
                errors.append(f"package check failed: {e}")
    else:
        result["conda_python_exists"] = None
        result["required_packages_ok"] = None

    log_dir = os.path.join(script_dir, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        result["log_dir_ok"] = True
    except OSError as e:
        result["log_dir_ok"] = False
        errors.append(f"cannot create log dir: {e}")

    ssh_results = {}
    for cid, cluster in CLUSTERS.items():
        ssh = cluster.get("ssh")
        if ssh:
            try:
                r = subprocess.run(
                    f"{ssh} 'echo ok'", shell=True,
                    capture_output=True, text=True, timeout=15,
                )
                ok = r.returncode == 0 and "ok" in r.stdout
                ssh_results[cid] = ok
                if not ok:
                    errors.append(f"SSH failed for cluster '{cid}': {r.stderr.strip()[:200]}")
            except Exception as e:
                ssh_results[cid] = False
                errors.append(f"SSH error for cluster '{cid}': {e}")
        else:
            ssh_results[cid] = None  # local, no SSH needed
    result["ssh"] = ssh_results

    result["errors"] = errors
    result["ok"] = len(errors) == 0
    return jsonify(result)


@app.route("/")
def index():
    return Response(HTML_TEMPLATE, mimetype="text/html")


# ---------------------------------------------------------------------------
# Chat Agent (Claude API)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a SLURM cluster expert in a GPU dashboard. Two clusters: CS (local) and Rivanna (HPC, via SSH).

GOAL: explain why jobs are pending and suggest concrete fixes (partition, resource cuts, QoS, timing, cross-cluster moves). Be concise. Use bullets. Match the user's language. If a job is running, just summarize its status.

When the user gives a job ID, you receive scontrol/sprio/squeue output plus current state of both clusters. Use the cluster context block in the user message for "now" — do not call tools to get current state.

TOOL ROUTING (pick by user INTENT, not topic):

• General knowledge / GPU specs / model VRAM / framework questions → `web_search`. Cite URLs.
• User mentions a LOCAL path/folder + asks about hardware → `analyze_workspace` (then optionally `web_search` for VRAM numbers). Recommend a specific node + sbatch flags.
• User mentions a GitHub URL (https://github.com/...) + asks about hardware → `analyze_repository` (then optionally `web_search` for VRAM numbers). Recommend a specific node + sbatch flags. NEVER pass a GitHub URL to `analyze_workspace`.
• Patterns / "typical" / "best time" / "should I wait" / "is now good" → `query_history`. These are TIMING questions, NOT submission requests.
• Current free GPUs / node status / pending jobs → answer from the context block. No tool.
• `submit_job` ONLY for direct imperatives: "submit this", "run my script", "launch it", "sbatch this". A question is NEVER a submission request.
• `cancel_job` ONLY when user says "cancel/kill/scancel" + a specific job ID.

These are NOT submit requests — give advice, do NOT call submit_job:
  "Should I submit now or wait?", "What GPU should I use?", "Can I run this here?",
  "Will my job fit?", "When is the cluster quiet?"

Rule of thumb: yes/no or WHEN question → advice. "Do X" → action. When in doubt, advise.

submit_job and cancel_job are TWO-STAGE: they return {"status": "pending_user_approval"}. The user must click Approve. After calling one, say "I've prepared it — click Approve to run." Never call the same action tool twice. Never claim a job was submitted or cancelled.

File path rule: if the command contains an absolute path (starts with /), you MUST use the exact path the user stated. Never invent or guess a path. If the user did not give a full path, ask them to confirm it before calling submit_job.
"""

openai_client = None
_openai_key_cached = None


def _read_env_file_key():
    """Re-read OPENAI_API_KEY from .env each call so users can rotate the key
    without restarting the dashboard. Falls back to os.environ."""
    env_path = os.path.join(SCRIPT_DIR, ".env")
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "OPENAI_API_KEY":
                    v = v.strip().strip('"').strip("'")
                    if v:
                        return v
    except FileNotFoundError:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


def get_openai():
    global openai_client, _openai_key_cached
    key = _read_env_file_key()
    if openai_client is None or key != _openai_key_cached:
        openai_client = OpenAI(api_key=key)
        _openai_key_cached = key
    return openai_client


def collect_job_context(job_id, cluster_id):
    """Gather detailed SLURM info about a specific job for Claude analysis."""
    ssh = CLUSTERS.get(cluster_id, {}).get("ssh")
    cmds = [
        f"scontrol show job {job_id}",
        f"sprio -j {job_id} -l",
        f"squeue -j {job_id} -o '%.18i|%P|%j|%u|%T|%M|%l|%D|%C|%m|%b|%R|%V|%S' --noheader",
        "squeue -t PD -o '%.18i|%P|%.30j|%.8u|%.2t|%.10M|%.4D|%R' --noheader | head -30",
    ]
    sections = run_batched(cmds, ssh)
    labels = ["scontrol_show_job", "sprio", "squeue_job", "pending_queue_sample"]
    result = {}
    for i, label in enumerate(labels):
        if i < len(sections):
            result[label] = "\n".join(sections[i])
        else:
            result[label] = "(no data)"
    return result


def _cluster_summary_text(cluster_id):
    """Build a text summary of a cluster's current state for the chat agent."""
    data = get_cluster_data(cluster_id)
    if "error" in data:
        return f"[{cluster_id}] (unavailable)\n"

    all_nodes = data.get("nodes", [])
    valid_partitions = sorted({n["partition"] for n in all_nodes if n.get("partition")})

    lines = [
        f"[{data['cluster_name']}]",
        f"  GPU Nodes: {data['total_gpu_nodes']}, "
        f"GPUs: {data['free_gpus']}/{data['total_gpus']} free, "
        f"CPUs: {data['free_cpus']}/{data['total_cpus']} free",
    ]
    if valid_partitions:
        lines.append(f"  Valid partitions: {', '.join(valid_partitions)}")

    free_nodes = [n for n in all_nodes if n["gpus_free"] > 0]
    if free_nodes:
        free_nodes.sort(key=lambda n: (-n["gpus_free"], n["node"]))
        lines.append("  Nodes with free GPUs:")
        for n in free_nodes[:8]:
            lines.append(
                f"    {n['node']}: "
                f"{n['gpus_free']}/{n['gpu_count']} {n['gpu_type']} free, "
                f"{n['cpus_free']}/{n['cpus']} CPUs, {n['free_mem_gb']}/{n['mem_gb']}GB mem, "
                f"state={n['state']}"
            )
    else:
        lines.append("  No free GPUs available.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Historical snapshots (SQLite)
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def db_init():
    with _db_lock, db_connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            ts          INTEGER NOT NULL,
            cluster     TEXT    NOT NULL,
            total_gpus  INTEGER,
            free_gpus   INTEGER,
            total_cpus  INTEGER,
            free_cpus   INTEGER,
            total_nodes INTEGER,
            PRIMARY KEY (ts, cluster)
        );
        CREATE INDEX IF NOT EXISTS idx_snap_cluster_ts
            ON snapshots(cluster, ts);

        CREATE TABLE IF NOT EXISTS node_snapshots (
            ts         INTEGER NOT NULL,
            cluster    TEXT    NOT NULL,
            node       TEXT    NOT NULL,
            gpu_type   TEXT,
            gpus_free  INTEGER,
            gpu_count  INTEGER,
            state      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_node_snap_lookup
            ON node_snapshots(cluster, gpu_type, ts);
        """)


def db_insert_snapshot(cluster_id, data):
    """Persist one cluster snapshot. Called by snapshot_loop()."""
    if not data or "error" in data:
        return
    ts = int(time.time())
    rows_node = [
        (ts, cluster_id, n["node"], n["gpu_type"],
         n["gpus_free"], n["gpu_count"], n["state"])
        for n in data.get("nodes", [])
    ]
    with _db_lock, db_connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO snapshots "
            "(ts, cluster, total_gpus, free_gpus, total_cpus, free_cpus, total_nodes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, cluster_id,
             data.get("total_gpus", 0), data.get("free_gpus", 0),
             data.get("total_cpus", 0), data.get("free_cpus", 0),
             data.get("total_gpu_nodes", 0)),
        )
        c.executemany(
            "INSERT INTO node_snapshots "
            "(ts, cluster, node, gpu_type, gpus_free, gpu_count, state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows_node,
        )
        # Retention: drop rows older than 30 days
        cutoff = ts - 30 * 86400
        c.execute("DELETE FROM snapshots WHERE ts < ?", (cutoff,))
        c.execute("DELETE FROM node_snapshots WHERE ts < ?", (cutoff,))


def db_query_timeseries(cluster_id, hours):
    cutoff = int(time.time()) - hours * 3600
    with _db_lock, db_connect() as c:
        rows = c.execute(
            "SELECT ts, free_gpus, total_gpus, free_cpus, total_cpus "
            "FROM snapshots WHERE cluster = ? AND ts >= ? ORDER BY ts",
            (cluster_id, cutoff),
        ).fetchall()
    return [
        {"ts": r[0], "free_gpus": r[1], "total_gpus": r[2],
         "free_cpus": r[3], "total_cpus": r[4]}
        for r in rows
    ]


def db_query_heatmap(cluster_id, days):
    """Return a 7x24 grid: average free_gpus by (day_of_week, hour_of_day).
    day_of_week: 0=Mon..6=Sun (Python convention)."""
    cutoff = int(time.time()) - days * 86400
    with _db_lock, db_connect() as c:
        rows = c.execute(
            "SELECT ts, free_gpus FROM snapshots WHERE cluster = ? AND ts >= ?",
            (cluster_id, cutoff),
        ).fetchall()
    grid = [[None] * 24 for _ in range(7)]
    counts = [[0] * 24 for _ in range(7)]
    sums = [[0.0] * 24 for _ in range(7)]
    for ts, free_gpus in rows:
        dt = datetime.fromtimestamp(ts)
        d = dt.weekday()
        h = dt.hour
        sums[d][h] += free_gpus
        counts[d][h] += 1
    for d in range(7):
        for h in range(24):
            if counts[d][h]:
                grid[d][h] = round(sums[d][h] / counts[d][h], 1)
    return grid


def db_query_history(cluster_id, gpu_type=None, lookback_hours=24):
    """Aggregated stats used by the agent's query_history tool."""
    cutoff = int(time.time()) - lookback_hours * 3600
    with _db_lock, db_connect() as c:
        if gpu_type:
            rows = c.execute(
                "SELECT ts, SUM(gpus_free), SUM(gpu_count) "
                "FROM node_snapshots WHERE cluster = ? AND gpu_type = ? AND ts >= ? "
                "GROUP BY ts ORDER BY ts",
                (cluster_id, gpu_type, cutoff),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT ts, free_gpus, total_gpus "
                "FROM snapshots WHERE cluster = ? AND ts >= ? ORDER BY ts",
                (cluster_id, cutoff),
            ).fetchall()
    if not rows:
        return {
            "cluster": cluster_id, "gpu_type": gpu_type,
            "lookback_hours": lookback_hours,
            "samples": 0,
            "message": "no historical data yet",
        }
    free_vals = [r[1] for r in rows if r[1] is not None]
    total_vals = [r[2] for r in rows if r[2] is not None]
    avg_free = round(sum(free_vals) / len(free_vals), 1) if free_vals else 0
    min_free = min(free_vals) if free_vals else 0
    max_free = max(free_vals) if free_vals else 0
    typical_total = max(total_vals) if total_vals else 0

    # Find best (highest avg free) and worst hours of day
    by_hour = {}
    for r in rows:
        ts, free, _total = r[0], r[1], r[2]
        if free is None:
            continue
        h = datetime.fromtimestamp(ts).hour
        by_hour.setdefault(h, []).append(free)
    hour_avg = {h: round(sum(v) / len(v), 1) for h, v in by_hour.items()}
    best_hours = sorted(hour_avg.items(), key=lambda kv: -kv[1])[:3]
    worst_hours = sorted(hour_avg.items(), key=lambda kv: kv[1])[:3]

    return {
        "cluster": cluster_id,
        "gpu_type": gpu_type,
        "lookback_hours": lookback_hours,
        "samples": len(rows),
        "typical_total_gpus": typical_total,
        "avg_free": avg_free,
        "min_free": min_free,
        "max_free": max_free,
        "best_hours_of_day": [{"hour": h, "avg_free": v} for h, v in best_hours],
        "worst_hours_of_day": [{"hour": h, "avg_free": v} for h, v in worst_hours],
    }


def snapshot_loop(interval_seconds=60):
    """Background thread: collect a snapshot from each cluster every minute."""
    db_init()
    print(f"[history] snapshot loop started, interval={interval_seconds}s, db={DB_PATH}", flush=True)
    while True:
        for cid in CLUSTERS.keys():
            try:
                data = get_cluster_data(cid)
                db_insert_snapshot(cid, data)
            except Exception as e:
                print(f"[history] snapshot {cid} failed: {e}", flush=True)
        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# History API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/history/<cluster_id>")
def api_history(cluster_id):
    if cluster_id not in CLUSTERS:
        return jsonify({"error": "unknown cluster"}), 404
    try:
        hours = int(request.args.get("hours", 24))
    except ValueError:
        hours = 24
    hours = max(1, min(hours, 24 * 30))
    return jsonify({"cluster": cluster_id, "hours": hours,
                    "series": db_query_timeseries(cluster_id, hours)})


@app.route("/api/history/<cluster_id>/heatmap")
def api_history_heatmap(cluster_id):
    if cluster_id not in CLUSTERS:
        return jsonify({"error": "unknown cluster"}), 404
    try:
        days = int(request.args.get("days", 7))
    except ValueError:
        days = 7
    days = max(1, min(days, 30))
    return jsonify({"cluster": cluster_id, "days": days,
                    "grid": db_query_heatmap(cluster_id, days)})


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

# --- Audit log + pending-action store -------------------------------------

PENDING_ACTIONS = {}  # action_id -> {"tool": str, "args": dict, "ts": float}
_pending_lock = threading.Lock()
PENDING_TTL_SEC = 600  # 10 minutes


def _agent_log(event, name, payload):
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "tool": name,
        "payload": payload,
    }
    try:
        with open(AGENT_LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print(f"[agent_log] write failed: {e}", file=sys.stderr)


def _stash_pending(name, args):
    aid = secrets.token_hex(8)
    with _pending_lock:
        # Reap stale entries
        now = time.time()
        for k, v in list(PENDING_ACTIONS.items()):
            if now - v["ts"] > PENDING_TTL_SEC:
                PENDING_ACTIONS.pop(k, None)
        PENDING_ACTIONS[aid] = {"tool": name, "args": args, "ts": now}
    return aid


def _take_pending(action_id):
    with _pending_lock:
        return PENDING_ACTIONS.pop(action_id, None)


# --- Mutating SLURM tools -------------------------------------------------

_VALID_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_VALID_NODE = re.compile(r"^[A-Za-z0-9._-]+$")   # node/nodelist: same charset as names
_VALID_TIME = re.compile(r"^\d+(-\d{1,2})?(:\d{1,2}){0,2}$")
_VALID_MEM = re.compile(r"^\d+[KMGT]?$", re.IGNORECASE)


def _build_sbatch_script(args):
    """Build a sanitized sbatch script from validated arg dict.
    Returns (ok, script_or_error)."""
    job_name  = (args.get("job_name") or "agent_job").strip()
    partition = (args.get("partition") or "").strip()
    node      = (args.get("node") or "").strip()
    cluster_id = (args.get("cluster_id") or "cs")
    cluster_account = (CLUSTERS.get(cluster_id, {}).get("account") or "")
    account   = (args.get("account") or cluster_account or "").strip()
    time_lim  = (args.get("time") or "01:00:00").strip()
    mem       = (args.get("mem") or "4G").strip()
    cpus      = int(args.get("cpus") or 1)
    gpus      = int(args.get("gpus") or 0)
    gpu_type  = (args.get("gpu_type") or "").strip()
    command   = args.get("command") or ""
    workdir   = (args.get("workdir") or "").strip()

    if not partition:
        return False, "partition is required"
    if not _VALID_NAME.match(partition):
        return False, f"invalid partition: {partition}"
    if node and not _VALID_NODE.match(node):
        return False, f"invalid node: {node}"
    if account and not _VALID_NAME.match(account):
        return False, f"invalid account: {account}"
    if not _VALID_NAME.match(job_name):
        return False, f"invalid job_name: {job_name}"
    if not _VALID_TIME.match(time_lim):
        return False, f"invalid time format: {time_lim}"
    if not _VALID_MEM.match(mem):
        return False, f"invalid mem format: {mem}"
    if cpus < 1 or cpus > 256:
        return False, f"cpus out of range: {cpus}"
    if gpus < 0 or gpus > 16:
        return False, f"gpus out of range: {gpus}"
    if gpu_type and not _VALID_NAME.match(gpu_type):
        return False, f"invalid gpu_type: {gpu_type}"
    if not command.strip():
        return False, "command is required"
    if workdir and not re.match(r"^[A-Za-z0-9._/~-]+$", workdir):
        return False, f"invalid workdir: {workdir}"
    # For local cluster (no SSH), check that any absolute file path in the
    # command actually exists — catches hallucinated or stale paths early.
    if not CLUSTERS.get(cluster_id, {}).get("ssh"):
        for token in command.split():
            if token.startswith("/") and not os.path.exists(token):
                return False, (
                    f"file not found on this cluster: {token!r}. "
                    "Please confirm the absolute path before submitting."
                )

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    lines.append(f"#SBATCH --partition={partition}")
    if node:
        lines.append(f"#SBATCH --nodelist={node}")
    lines += [
        f"#SBATCH --time={time_lim}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={cpus}",
    ]
    if gpus > 0:
        gres = f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}"
        lines.append(f"#SBATCH --gres={gres}")
    lines.append("")
    if workdir:
        lines.append(f"cd {shlex.quote(workdir)}")
    lines.append(command.rstrip("\n"))
    return True, "\n".join(lines) + "\n"


def submit_job(**args):
    """Build sbatch script and submit it (locally or via SSH).
    Always returns a dict, never raises."""
    cluster_id = args.get("cluster_id", "cs")
    cluster = CLUSTERS.get(cluster_id)
    if not cluster:
        return {"ok": False, "error": f"unknown cluster: {cluster_id}"}
    ok, script_or_err = _build_sbatch_script(args)
    if not ok:
        return {"ok": False, "error": script_or_err}
    script = script_or_err

    ssh = cluster["ssh"]
    print(f"[submit_job] script:\n{script}", file=sys.stderr, flush=True)
    # Pipe the script into sbatch via stdin so we don't have to write a temp file
    if ssh:
        cmd = f"{ssh} 'sbatch'"
    else:
        cmd = "sbatch"
    try:
        result = subprocess.run(
            cmd, input=script, shell=True, capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        return {"ok": False, "error": f"sbatch invocation failed: {e}"}

    out = (result.stdout or "") + (result.stderr or "")
    m = re.search(r"Submitted batch job (\d+)", out)
    if m:
        return {"ok": True, "job_id": m.group(1), "cluster": cluster_id, "output": out.strip()}
    return {"ok": False, "error": "sbatch did not return a job id", "output": out.strip(),
            "script": script}


def cancel_job(**args):
    cluster_id = args.get("cluster_id", "cs")
    job_id = str(args.get("job_id", "")).strip()
    cluster = CLUSTERS.get(cluster_id)
    if not cluster:
        return {"ok": False, "error": f"unknown cluster: {cluster_id}"}
    if not re.match(r"^\d+(_\d+)?$", job_id):
        return {"ok": False, "error": f"invalid job id: {job_id}"}
    ssh = cluster["ssh"]
    cmd = f"scancel {job_id}"
    if ssh:
        cmd = f"{ssh} {shlex.quote(cmd)}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
    except Exception as e:
        return {"ok": False, "error": f"scancel failed: {e}"}
    out = (result.stdout or "") + (result.stderr or "")
    return {"ok": result.returncode == 0, "job_id": job_id, "cluster": cluster_id, "output": out.strip()}


def execute_pending(action_id):
    """Run a previously-stashed pending action. Returns dict result."""
    rec = _take_pending(action_id)
    if not rec:
        return {"ok": False, "error": "action not found or expired"}
    name, args = rec["tool"], rec["args"]
    _agent_log("execute", name, args)
    if name == "submit_job":
        result = submit_job(**args)
    elif name == "cancel_job":
        result = cancel_job(**args)
    else:
        result = {"ok": False, "error": f"non-mutating tool sent to confirm: {name}"}
    _agent_log("result", name, result)
    return result


def _norm_gpu(s):
    """Strip spaces/underscores/dashes and lowercase for fuzzy GPU name matching."""
    return re.sub(r"[\s_\-]", "", s or "").lower()


def find_best_target(cluster_id, gpus=1, gpu_type=None, partition=None, cpus=1, node=None):
    """Validate and correct a job request against live cluster data.

    Returns a dict with:
      partition  – valid partition name (corrected if needed)
      node       – best usable node, or None if none found
      gpu_type   – canonical GPU type name from cluster data, or None (→ generic gres)
      corrections – list of human-readable correction strings
      error      – set only when no valid target exists at all
    """
    data = get_cluster_data(cluster_id)
    if "error" in data:
        return {"error": f"cannot fetch cluster data: {data['error']}"}

    nodes = data.get("nodes", [])
    corrections = []

    # Index: partition → node list
    partition_nodes: dict = {}
    for n in nodes:
        p = n.get("partition", "")
        if p:
            partition_nodes.setdefault(p, []).append(n)

    valid_partitions = set(partition_nodes)

    def _partitions_with_gpu(gpu_req):
        """Return partitions that contain at least one node with the requested GPU type."""
        norm = _norm_gpu(gpu_req)
        result = set()
        for n in nodes:
            if norm in _norm_gpu(n.get("gpu_type", "")):
                p = n.get("partition", "")
                if p:
                    result.add(p)
        return result

    def _best_partition_by_free(candidates):
        return max(candidates, key=lambda p: sum(
            n.get("gpus_free", 0) for n in partition_nodes.get(p, [])
        ))

    # ── Step 1: resolve partition ────────────────────────────────────────────
    if partition:
        # Case-insensitive exact match
        matched = next((p for p in valid_partitions if p.lower() == partition.lower()), None)
        if matched and matched != partition:
            corrections.append(f"partition name corrected: '{partition}' → '{matched}'")
            partition = matched
        elif not matched:
            # Partition doesn't exist — fall back
            if gpu_type and (gpu_parts := _partitions_with_gpu(gpu_type)):
                new_p = _best_partition_by_free(gpu_parts)
                corrections.append(
                    f"partition '{partition}' not found; using '{new_p}' (has {gpu_type})"
                )
                partition = new_p
            elif valid_partitions:
                new_p = _best_partition_by_free(valid_partitions)
                corrections.append(
                    f"partition '{partition}' not found; using '{new_p}' (most free GPUs)"
                )
                partition = new_p
            else:
                return {"error": f"partition '{partition}' not found and cluster has no GPU partitions"}
    else:
        # Auto-select partition
        if gpu_type and (gpu_parts := _partitions_with_gpu(gpu_type)):
            partition = _best_partition_by_free(gpu_parts)
        elif valid_partitions:
            partition = _best_partition_by_free(valid_partitions)
        else:
            return {"error": "no GPU partitions found in cluster"}

    # ── Step 2: resolve gpu_type within chosen partition ─────────────────────
    part_nodes = partition_nodes.get(partition, [])
    gpu_type_gres = None  # raw SLURM GRES token (e.g. "nvidia_a40") for --gres=

    def _raw_gres_for_display(display_name, node_list):
        """Return the raw SLURM GRES token for a display name from the first matching node."""
        norm = _norm_gpu(display_name)
        for n in node_list:
            if norm in _norm_gpu(n.get("gpu_type", "")):
                raw = n.get("gpu_type_raw", "")
                if raw:
                    return raw
        return ""

    if gpu_type and gpus > 0:
        norm_req = _norm_gpu(gpu_type)
        available_gpu_types = {n.get("gpu_type", "") for n in part_nodes if n.get("gpu_type")}
        matched_gpu = next(
            (ag for ag in available_gpu_types if norm_req in _norm_gpu(ag)),
            None,
        )
        if not matched_gpu:
            # GPU type absent in this partition — try another partition
            if gpu_parts := _partitions_with_gpu(gpu_type):
                new_p = _best_partition_by_free(gpu_parts)
                corrections.append(
                    f"gpu_type='{gpu_type}' not in partition '{partition}'; "
                    f"switching to partition '{new_p}'"
                )
                partition = new_p
                part_nodes = partition_nodes.get(partition, [])
                available_gpu_types = {n.get("gpu_type", "") for n in part_nodes if n.get("gpu_type")}
                matched_gpu = next(
                    (ag for ag in available_gpu_types if norm_req in _norm_gpu(ag)),
                    None,
                )
            if not matched_gpu:
                corrections.append(
                    f"gpu_type='{gpu_type}' not found anywhere; using generic --gres=gpu:{gpus}"
                )
                gpu_type = None
            else:
                gpu_type = matched_gpu
                gpu_type_gres = _raw_gres_for_display(matched_gpu, part_nodes)
        else:
            gpu_type = matched_gpu
            gpu_type_gres = _raw_gres_for_display(matched_gpu, part_nodes)

    # ── Step 3: resolve node ─────────────────────────────────────────────────
    _USABLE = {"IDLE", "MIXED"}

    def _node_ok(n):
        if n.get("state", "").upper() not in _USABLE:
            return False
        if n.get("gpus_free", 0) < (gpus or 0):
            return False
        if n.get("cpus_free", 0) < cpus:
            return False
        if gpu_type and _norm_gpu(gpu_type) not in _norm_gpu(n.get("gpu_type", "")):
            return False
        return True

    if node:
        node_in_part = next((n for n in part_nodes if n["node"] == node), None)
        if node_in_part is None:
            # Node not in this partition — find which partition it belongs to
            other_part = next(
                (p for p, ns in partition_nodes.items() if any(n["node"] == node for n in ns)),
                None,
            )
            if other_part:
                corrections.append(
                    f"node '{node}' is not in partition '{partition}'; "
                    f"it belongs to '{other_part}' — switching partition"
                )
                partition = other_part
                part_nodes = partition_nodes.get(partition, [])
                node_in_part = next((n for n in part_nodes if n["node"] == node), None)
            else:
                corrections.append(f"node '{node}' not found in any partition; auto-selecting")
                node = None
                node_in_part = None

        if node_in_part:
            state = node_in_part.get("state", "").upper()
            if state not in _USABLE:
                corrections.append(
                    f"node '{node}' is {state} (not usable); auto-selecting a better node"
                )
                node = None
            elif node_in_part.get("gpus_free", 0) < (gpus or 0):
                corrections.append(
                    f"node '{node}' has only {node_in_part['gpus_free']} free GPUs "
                    f"(need {gpus}); auto-selecting"
                )
                node = None

    if not node:
        candidates = sorted(
            [n for n in part_nodes if _node_ok(n)],
            key=lambda n: (0 if n.get("state", "").upper() == "IDLE" else 1, -n.get("gpus_free", 0)),
        )
        if candidates:
            node = candidates[0]["node"]
        # If no usable candidates, leave node=None — sbatch will pick without --nodelist

    return {
        "partition": partition,
        "node": node,
        "gpu_type": gpu_type,         # normalized display name (for corrections/messages)
        "gpu_type_gres": gpu_type_gres,  # raw SLURM GRES token (for --gres= line)
        "corrections": corrections,
    }


def web_search(query, max_results=5):
    """Search the web via DuckDuckGo (ddgs). Returns list of {title,url,snippet}
    or {"error": "..."} on failure. Never raises."""
    if DDGS is None:
        return {"error": "web search unavailable: 'ddgs' package not installed"}
    if not query or not query.strip():
        return {"error": "empty query"}
    max_results = max(1, min(int(max_results or 5), 5))

    result_holder = {"value": None}

    def _do_search():
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results))
            cleaned = []
            for h in hits[:max_results]:
                snippet = (h.get("body") or "")[:300]
                cleaned.append({
                    "title": (h.get("title") or "")[:200],
                    "url": h.get("href") or h.get("url") or "",
                    "snippet": snippet,
                })
            result_holder["value"] = cleaned
        except Exception as e:
            result_holder["value"] = {"error": f"search failed: {e}"}

    t = threading.Thread(target=_do_search, daemon=True)
    t.start()
    t.join(timeout=10)
    if t.is_alive():
        return {"error": "search timed out after 10s"}
    return result_holder["value"] if result_holder["value"] is not None else {"error": "no result"}


# --- Workspace analysis ---------------------------------------------------

_WS_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env", ".tox",
    "dist", "build", "wandb", "checkpoints", "outputs", ".mypy_cache",
    ".pytest_cache", ".idea", ".vscode", "site-packages", ".cache",
}
_WS_DEP_FILES = {
    "requirements.txt", "requirements-dev.txt", "pyproject.toml",
    "environment.yml", "environment.yaml", "Pipfile", "setup.py", "setup.cfg",
}
_WS_README_FILES = {"README.md", "README.rst", "readme.md", "Readme.md"}
_WS_CONFIG_EXTS = (".yaml", ".yml", ".json", ".toml")
_WS_SBATCH_EXTS = (".sbatch", ".slurm")
_WS_GPU_LIBS = {
    "torch", "pytorch", "transformers", "accelerate", "deepspeed", "vllm",
    "bitsandbytes", "xformers", "flash_attn", "flash-attn", "jax", "tensorflow",
    "triton", "peft", "trl", "unsloth", "diffusers", "auto_gptq", "awq",
}
_WS_MODEL_HINTS = [
    "llama", "mistral", "mixtral", "qwen", "gemma", "phi", "falcon", "yi",
    "gpt2", "gpt-j", "gpt-neox", "bloom", "bert", "roberta", "t5", "stable-diffusion",
    "sdxl", "whisper", "clip",
]
_WS_SIZE_RE = re.compile(r"\b(\d{1,3})\s*[bB]\b")  # 7b, 13B, 70b, 405B
_WS_HEAVY_SIGNALS = {"deepspeed", "fsdp", "zero_optimization", "zero3", "model_parallel"}
_WS_QUANT_SIGNALS = {"load_in_4bit", "load_in_8bit", "qlora", "lora", "bnb", "gptq", "awq"}
_WS_KEY_HINTS = [
    "batch_size", "micro_batch", "per_device_train_batch_size",
    "gradient_accumulation_steps", "fp16", "bf16",
    "gradient_checkpointing", "model_name_or_path", "model_name", "base_model",
    "max_seq_length", "seq_len",
]


def _looks_binary(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read(2048)
        return b"\x00" in chunk
    except Exception:
        return True


def _read_text(path, max_bytes=100_000):
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        return data[:max_bytes].decode("utf-8", errors="replace"), truncated
    except Exception:
        return "", False


def analyze_workspace(path, max_files=50):
    """Walk a project directory and extract GPU-relevant signals.
    Read-only. Returns a dict the LLM can reason over, or {"error": ...}."""
    if not path or not isinstance(path, str):
        return {"error": "missing path"}
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        return {"error": f"path does not exist: {path}"}
    if not os.path.isdir(expanded):
        return {"error": f"path is not a directory: {path}"}

    walked = 0
    read = 0
    deps_raw = []          # raw lines from dependency files
    gpu_libs = set()       # GPU-relevant libs we saw
    model_hints = set()    # model family names found
    size_hints = set()     # parameter sizes like '70B'
    config_signals = set() # signals from configs
    quant_signals = set()
    heavy_signals = set()
    sbatch_lines = []
    readme_excerpt = ""
    files_seen = []

    for root, dirs, files in os.walk(expanded):
        # prune ignored dirs in-place
        dirs[:] = [d for d in dirs if d not in _WS_IGNORE_DIRS and not d.startswith(".")]
        for fname in files:
            walked += 1
            if walked > 300:
                break
            full = os.path.join(root, fname)
            try:
                if os.path.getsize(full) > 1_000_000:
                    continue
            except OSError:
                continue

            is_dep = fname in _WS_DEP_FILES
            is_readme = fname in _WS_README_FILES
            is_py = fname.endswith(".py")
            is_cfg = fname.endswith(_WS_CONFIG_EXTS)
            is_sbatch = fname.endswith(_WS_SBATCH_EXTS) or fname in ("job.sh", "submit.sh", "run.sh")

            if not (is_dep or is_readme or is_py or is_cfg or is_sbatch):
                continue
            if read >= max_files:
                continue
            if _looks_binary(full):
                continue

            text, _ = _read_text(full)
            if not text:
                continue
            read += 1
            rel = os.path.relpath(full, expanded)
            files_seen.append(rel)
            low = text.lower()

            if is_dep:
                for line in text.splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    deps_raw.append(s[:200])
                for lib in _WS_GPU_LIBS:
                    if lib in low:
                        gpu_libs.add(lib)

            if is_py:
                # Extract import targets cheaply
                for m in re.finditer(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", text, re.MULTILINE):
                    top = m.group(1).split(".")[0].lower()
                    if top in _WS_GPU_LIBS or top.replace("-", "_") in _WS_GPU_LIBS:
                        gpu_libs.add(top)
                for hint in _WS_MODEL_HINTS:
                    if hint in low:
                        model_hints.add(hint)
                for m in _WS_SIZE_RE.finditer(text):
                    size_hints.add(m.group(1) + "B")
                for s in _WS_HEAVY_SIGNALS:
                    if s in low:
                        heavy_signals.add(s)
                for s in _WS_QUANT_SIGNALS:
                    if s in low:
                        quant_signals.add(s)

            if is_cfg:
                for hint in _WS_MODEL_HINTS:
                    if hint in low:
                        model_hints.add(hint)
                for m in _WS_SIZE_RE.finditer(text):
                    size_hints.add(m.group(1) + "B")
                for s in _WS_HEAVY_SIGNALS:
                    if s in low:
                        heavy_signals.add(s)
                for s in _WS_QUANT_SIGNALS:
                    if s in low:
                        quant_signals.add(s)
                for k in _WS_KEY_HINTS:
                    if k in low:
                        config_signals.add(k)

            if is_readme and not readme_excerpt:
                readme_excerpt = text[:2000]

            if is_sbatch:
                for line in text.splitlines():
                    s = line.strip()
                    if "--gres" in s or "--mem" in s or "--time" in s or "--cpus-per-task" in s or "--partition" in s:
                        sbatch_lines.append(s[:200])
        if walked > 300:
            break

    # Pressure heuristic
    largest_size = 0
    for s in size_hints:
        try:
            n = int(s.rstrip("Bb"))
            if n > largest_size:
                largest_size = n
        except ValueError:
            pass

    if heavy_signals or largest_size >= 30:
        pressure = "heavy"
    elif quant_signals or 7 <= largest_size < 30:
        pressure = "medium"
    elif gpu_libs and (model_hints or largest_size > 0):
        pressure = "light"
    elif gpu_libs:
        pressure = "light"
    else:
        pressure = "unknown"

    pressure_reasons = []
    if heavy_signals:
        pressure_reasons.append(f"heavy training signals: {sorted(heavy_signals)}")
    if largest_size:
        pressure_reasons.append(f"largest model size hint: {largest_size}B")
    if quant_signals:
        pressure_reasons.append(f"quantization/PEFT: {sorted(quant_signals)}")
    if not pressure_reasons:
        pressure_reasons.append("no strong signals; defaulting from libs/model hints")

    return {
        "path": expanded,
        "files_walked": walked,
        "files_read": read,
        "files_seen": files_seen[:30],
        "dependencies": deps_raw[:60],
        "gpu_libs": sorted(gpu_libs),
        "model_hints": sorted(model_hints),
        "size_hints": sorted(size_hints),
        "config_signals": sorted(config_signals),
        "quantization_signals": sorted(quant_signals),
        "heavy_signals": sorted(heavy_signals),
        "existing_sbatch_directives": sbatch_lines[:20],
        "readme_excerpt": readme_excerpt,
        "pressure": pressure,
        "pressure_reasons": pressure_reasons,
    }


def analyze_repository(url):
    """Fetch key files from a public GitHub repo via the GitHub API and apply
    the same GPU-signal extraction as analyze_workspace. No auth required."""
    if not url or not isinstance(url, str):
        return {"error": "missing url"}
    m = re.match(r"https?://github\.com/([^/\s#?]+)/([^/\s#?]+)", url.strip())
    if not m:
        return {"error": f"not a recognized GitHub URL: {url}"}
    owner, repo = m.group(1), m.group(2).rstrip("/")

    def _gh_get(path):
        api_url = f"https://api.github.com/repos/{owner}/{repo}" + (f"/{path}" if path else "")
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "gpu-monitor/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            return {"_error": str(e)}

    def _gh_text(path):
        data = _gh_get(f"contents/{path}")
        if isinstance(data, dict) and (data.get("_error") or data.get("encoding") != "base64"):
            return ""
        try:
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            return ""

    # Repo metadata
    meta = _gh_get("")
    if isinstance(meta, dict) and "_error" in meta:
        return {"error": f"GitHub API error: {meta['_error']}"}

    repo_desc = meta.get("description", "") or ""
    language = meta.get("language", "") or ""
    default_branch = meta.get("default_branch", "main")

    # Root file listing
    contents = _gh_get("contents/")
    if not isinstance(contents, list):
        return {"error": "could not list repo contents (private repo or API error)"}

    root_files = {f["name"] for f in contents if f.get("type") == "file"}

    gpu_libs = set()
    model_hints = set()
    size_hints = set()
    config_signals = set()
    quant_signals = set()
    heavy_signals = set()
    deps_raw = []
    readme_excerpt = ""
    files_fetched = []

    # Fetch dependency and README files
    for fname in list(_WS_DEP_FILES) + list(_WS_README_FILES):
        if fname not in root_files:
            continue
        text = _gh_text(fname)
        if not text:
            continue
        files_fetched.append(fname)
        low = text.lower()

        if fname in _WS_DEP_FILES:
            for line in text.splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    deps_raw.append(s[:200])
            for lib in _WS_GPU_LIBS:
                if lib in low:
                    gpu_libs.add(lib)

        if fname in _WS_README_FILES and not readme_excerpt:
            readme_excerpt = text[:2000]
            low_r = readme_excerpt.lower()
            for hint in _WS_MODEL_HINTS:
                if hint in low_r:
                    model_hints.add(hint)
            for ms in _WS_SIZE_RE.finditer(readme_excerpt):
                size_hints.add(ms.group(1) + "B")
            for s in _WS_HEAVY_SIGNALS:
                if s in low_r:
                    heavy_signals.add(s)
            for s in _WS_QUANT_SIGNALS:
                if s in low_r:
                    quant_signals.add(s)

    # Scan up to 5 top-level .py files for imports and signals
    for fname in sorted(f for f in root_files if f.endswith(".py"))[:5]:
        text = _gh_text(fname)
        if not text:
            continue
        files_fetched.append(fname)
        low = text.lower()
        for mi in re.finditer(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", text, re.MULTILINE):
            top = mi.group(1).split(".")[0].lower()
            if top in _WS_GPU_LIBS:
                gpu_libs.add(top)
        for hint in _WS_MODEL_HINTS:
            if hint in low:
                model_hints.add(hint)
        for ms in _WS_SIZE_RE.finditer(text):
            size_hints.add(ms.group(1) + "B")
        for s in _WS_HEAVY_SIGNALS:
            if s in low:
                heavy_signals.add(s)
        for s in _WS_QUANT_SIGNALS:
            if s in low:
                quant_signals.add(s)

    # Same pressure heuristic as analyze_workspace
    largest_size = 0
    for s in size_hints:
        try:
            n = int(s.rstrip("Bb"))
            if n > largest_size:
                largest_size = n
        except ValueError:
            pass

    if heavy_signals or largest_size >= 30:
        pressure = "heavy"
    elif quant_signals or 7 <= largest_size < 30:
        pressure = "medium"
    elif gpu_libs and (model_hints or largest_size > 0):
        pressure = "light"
    elif gpu_libs:
        pressure = "light"
    else:
        pressure = "unknown"

    pressure_reasons = []
    if heavy_signals:
        pressure_reasons.append(f"heavy training signals: {sorted(heavy_signals)}")
    if largest_size:
        pressure_reasons.append(f"largest model size hint: {largest_size}B")
    if quant_signals:
        pressure_reasons.append(f"quantization/PEFT: {sorted(quant_signals)}")
    if not pressure_reasons:
        pressure_reasons.append("no strong signals; defaulting from libs/model hints")

    return {
        "url": url,
        "repo": f"{owner}/{repo}",
        "description": repo_desc,
        "language": language,
        "default_branch": default_branch,
        "files_fetched": files_fetched,
        "dependencies": deps_raw[:60],
        "gpu_libs": sorted(gpu_libs),
        "model_hints": sorted(model_hints),
        "size_hints": sorted(size_hints),
        "config_signals": sorted(config_signals),
        "quantization_signals": sorted(quant_signals),
        "heavy_signals": sorted(heavy_signals),
        "readme_excerpt": readme_excerpt,
        "pressure": pressure,
        "pressure_reasons": pressure_reasons,
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information about GPUs, ML frameworks, "
                "CUDA versions, library compatibility, model VRAM requirements, or "
                "any topic outside the SLURM cluster state. Returns a list of "
                "{title, url, snippet} results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (1-5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_history",
            "description": (
                "Query historical cluster data collected over time (every minute). "
                "Use this when the user asks 'when is the cluster usually free', "
                "'what's typical availability', or wants to schedule based on past trends. "
                "Returns avg/min/max free GPUs over the lookback window plus the best "
                "and worst hours of day. Optionally filter by gpu_type "
                "(e.g. 'A100 80GB', 'H100', 'RTX 3090')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id":     {"type": "string", "enum": list(CLUSTERS.keys())},
                    "gpu_type":       {"type": "string", "description": "Optional GPU model filter."},
                    "lookback_hours": {"type": "integer", "default": 24, "description": "Hours of history to analyze (1-720)."},
                },
                "required": ["cluster_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_job",
            "description": (
                "Propose a new SLURM job to be submitted on behalf of the user. "
                "This DOES NOT execute immediately — it returns a pending action that "
                "the user must approve in the dashboard before it actually runs. "
                "Use this when the user explicitly asks to submit/run/launch something, "
                "or after a recommendation that the user agrees to. "
                "Always include a meaningful job_name and only the resources actually needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "string", "enum": list(CLUSTERS.keys())},
                    "partition":  {"type": "string", "description": "SLURM partition name (e.g. 'gpu', 'standard'). Never pass a node name here."},
                    "node":       {"type": "string", "description": "Optional specific node to target, e.g. 'udc-an37-13'. Generates --nodelist."},
                    "account":    {"type": "string", "description": "SLURM account/allocation to charge. Leave unset to use the cluster's configured account (or omit on clusters that don't require one, e.g. CS)."},
                    "command":    {"type": "string", "description": "Shell command(s) to run inside the job."},
                    "job_name":   {"type": "string", "default": "agent_job"},
                    "time":       {"type": "string", "default": "01:00:00", "description": "Walltime, e.g. 02:00:00 or 1-00:00:00."},
                    "mem":        {"type": "string", "default": "4G", "description": "Memory, e.g. 16G."},
                    "cpus":       {"type": "integer", "default": 1},
                    "gpus":       {"type": "integer", "default": 0},
                    "gpu_type":   {"type": "string", "description": "Optional sbatch GRES gpu type, e.g. 'a100' or 'rtx3090'."},
                    "workdir":    {"type": "string", "description": "Optional working directory to cd into before the command."},
                },
                "required": ["cluster_id", "partition", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_job",
            "description": (
                "Propose cancelling a SLURM job. DOES NOT execute immediately — returns "
                "a pending action that the user must approve. Use only when the user "
                "explicitly asks to cancel/kill/scancel a specific job."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "string", "enum": list(CLUSTERS.keys())},
                    "job_id":     {"type": "string"},
                },
                "required": ["cluster_id", "job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_workspace",
            "description": (
                "Read-only scan of a LOCAL project directory on the filesystem to "
                "understand what GPU resources the codebase will need. Extracts "
                "dependencies (requirements.txt, pyproject.toml, environment.yml), "
                "imported GPU libraries (torch, transformers, deepspeed, vllm, "
                "bitsandbytes, peft, ...), model name hints, parameter size hints "
                "(7B/13B/70B), training config signals (fp16, bf16, lora, qlora, "
                "deepspeed, fsdp, batch_size), any existing sbatch directives, "
                "and a coarse 'pressure' rating (light/medium/heavy/unknown). "
                "Call this ONLY when the user provides a local filesystem path "
                "(e.g. /home/user/myproject or ~/project). "
                "Do NOT use this for GitHub URLs — use analyze_repository instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or ~ path to the local project directory.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_repository",
            "description": (
                "Fetch and analyze a PUBLIC GitHub repository to understand what GPU "
                "resources the codebase will need. Use this when the user provides a "
                "GitHub URL (https://github.com/owner/repo). Fetches key files "
                "(requirements.txt, pyproject.toml, README, top-level Python files) "
                "via the GitHub API and applies the same GPU-signal extraction as "
                "analyze_workspace: GPU libs, model hints, parameter size hints, "
                "training signals, quantization signals, and a pressure rating. "
                "After calling this, optionally call web_search for VRAM numbers, "
                "then recommend a specific cluster node and sbatch flags. "
                "NEVER call analyze_workspace with a GitHub URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "GitHub repository URL, e.g. https://github.com/owner/repo",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


def dispatch_tool(name, args):
    """Run a tool by name with parsed JSON args. Returns a JSON-serializable result.
    Mutating tools (submit_job/cancel_job) NEVER execute here — they only stash a
    pending action that the user must approve via /api/agent/confirm."""
    if name == "web_search":
        return web_search(args.get("query", ""), args.get("max_results", 5))
    if name == "analyze_workspace":
        return analyze_workspace(args.get("path", ""))
    if name == "analyze_repository":
        return analyze_repository(args.get("url", ""))
    if name == "query_history":
        return db_query_history(
            args.get("cluster_id", "cs"),
            args.get("gpu_type"),
            int(args.get("lookback_hours", 24) or 24),
        )
    if name in ("submit_job", "cancel_job"):
        if name == "cancel_job":
            jid = str(args.get("job_id", "")).strip()
            if not re.match(r"^\d+(_\d+)?$", jid):
                return {"ok": False, "error": f"invalid job id: {jid}"}
        else:
            # Validate and correct partition / gpu_type / node against live cluster data
            # before doing string-level validation in _build_sbatch_script.
            cluster_id = args.get("cluster_id", "cs")
            gpus = int(args.get("gpus") or 0)
            target = find_best_target(
                cluster_id=cluster_id,
                gpus=gpus,
                gpu_type=args.get("gpu_type") if gpus > 0 else None,
                partition=args.get("partition"),
                cpus=int(args.get("cpus") or 1),
                node=args.get("node"),
            )
            if "error" in target:
                return {"ok": False, "error": target["error"]}

            args = dict(args)  # don't mutate the original dict
            original_node = args.get("node")  # capture before we modify args

            args["partition"] = target["partition"]

            # Never pass --nodelist for agent-submitted jobs.
            # sinfo -N deduplicates nodes to one partition (first seen), so our
            # index maps some nodes to the wrong partition bucket. The LLM reads
            # node names like "udc-ba07-38 (partition=gpu)" from the cluster
            # summary and passes them explicitly — but SLURM rejects them with
            # "Requested nodes not in this partition" because the node actually
            # lives in bii-gpu (or similar) and sinfo just happened to list it
            # under gpu first. Dropping --nodelist entirely and letting SLURM
            # schedule within the partition is always safer.
            args.pop("node", None)

            if gpus > 0:
                resolved_gpu = target.get("gpu_type")       # display name or None
                gres_token   = target.get("gpu_type_gres")  # raw SLURM token or ""
                if resolved_gpu is None:
                    # GPU type not found anywhere — emit generic --gres=gpu:N
                    args["gpu_type"] = ""
                elif gres_token:
                    # Use the exact SLURM GRES token from sinfo (e.g. "nvidia_a40")
                    # This is what sbatch needs — the display name ("A40") won't work
                    args["gpu_type"] = gres_token
                else:
                    # No raw token stored (shouldn't happen with current data, but
                    # fall back to lowercased display name as best guess)
                    args["gpu_type"] = resolved_gpu.lower().replace(" ", "_")

            ok, err = _build_sbatch_script(args)
            if not ok:
                return {"ok": False, "error": err}

            corrections = target.get("corrections", [])
            correction_note = (
                " Corrections applied: " + "; ".join(corrections) + "."
                if corrections else ""
            )
            aid = _stash_pending(name, args)
            _agent_log("propose", name, args)
            return {
                "status": "pending_user_approval",
                "action_id": aid,
                "tool": name,
                "args": args,
                "corrections": corrections,
                "message": (
                    "Action proposed. Awaiting user approval in the dashboard. "
                    "Do NOT call this tool again or assume it has run — wait for the user."
                    + correction_note
                ),
            }

        aid = _stash_pending(name, args)
        _agent_log("propose", name, args)
        return {
            "status": "pending_user_approval",
            "action_id": aid,
            "tool": name,
            "args": args,
            "message": (
                "Action proposed. Awaiting user approval in the dashboard. "
                "Do NOT call this tool again or assume it has run — wait for the user."
            ),
        }
    return {"error": f"unknown tool: {name}"}




@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True)
    msg = body.get("message", "").strip()
    history = body.get("history", [])

    if not msg:
        return jsonify({"error": "empty message"}), 400

    # Collect state of BOTH clusters
    summary_text = _cluster_summary_text("cs") + "\n\n" + _cluster_summary_text("hpc")

    # Extract job IDs from message
    job_ids = re.findall(r"\b(\d{5,})\b", msg)
    job_context_text = ""
    for jid in job_ids[:3]:
        # Try both clusters to find the job
        for cid in ("cs", "hpc"):
            ctx = collect_job_context(jid, cid)
            if ctx.get("scontrol_show_job", "").strip() and "Invalid job" not in ctx.get("scontrol_show_job", ""):
                cluster_name = CLUSTERS[cid]["name"]
                job_context_text += f"\n--- Job {jid} (found on {cluster_name}) ---\n"
                for k, v in ctx.items():
                    job_context_text += f"[{k}]\n{v}\n"
                break

    # Build messages for OpenAI
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})

    context_block = f"[Current State of All Clusters]\n{summary_text}"
    if job_context_text:
        context_block += f"\n{job_context_text}"
    messages.append({"role": "user", "content": f"{context_block}\n\n[User Question]\n{msg}"})

    tool_calls_made = []  # surfaced to the frontend for display
    MAX_ITERATIONS = 5

    try:
        client = get_openai()
        for _ in range(MAX_ITERATIONS):
            resp = client.chat.completions.create(
                model=CONFIG.get("openai_model", "gpt-4o"),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
            )
            choice = resp.choices[0].message
            tcs = getattr(choice, "tool_calls", None) or []

            if not tcs:
                return jsonify({
                    "response": choice.content or "",
                    "tool_calls": tool_calls_made,
                })

            # Append the assistant's tool-call message exactly as required by the API
            messages.append({
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tcs
                ],
            })

            for tc in tcs:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = dispatch_tool(name, args)

                # Build a compact summary for the UI
                summary = {"name": name, "args": args}
                if isinstance(result, dict) and result.get("status") == "pending_user_approval":
                    summary["pending"] = {
                        "action_id": result["action_id"],
                        "tool": result["tool"],
                        "args": result["args"],
                    }
                elif isinstance(result, dict) and "error" in result:
                    summary["error"] = result["error"]
                elif name == "web_search" and isinstance(result, list):
                    summary["results"] = [
                        {"title": r.get("title", ""), "url": r.get("url", "")}
                        for r in result[:3]
                    ]
                elif name == "analyze_workspace" and isinstance(result, dict):
                    summary["analysis"] = {
                        "path": result.get("path", ""),
                        "files_read": result.get("files_read", 0),
                        "gpu_libs": result.get("gpu_libs", []),
                        "model_hints": result.get("model_hints", []),
                        "size_hints": result.get("size_hints", []),
                        "pressure": result.get("pressure", "unknown"),
                    }
                elif name == "analyze_repository" and isinstance(result, dict):
                    summary["analysis"] = {
                        "url": result.get("url", ""),
                        "repo": result.get("repo", ""),
                        "files_fetched": len(result.get("files_fetched", [])),
                        "gpu_libs": result.get("gpu_libs", []),
                        "model_hints": result.get("model_hints", []),
                        "size_hints": result.get("size_hints", []),
                        "pressure": result.get("pressure", "unknown"),
                    }
                elif name == "query_history" and isinstance(result, dict):
                    summary["history"] = {
                        "cluster": result.get("cluster"),
                        "gpu_type": result.get("gpu_type"),
                        "lookback_hours": result.get("lookback_hours"),
                        "samples": result.get("samples"),
                        "avg_free": result.get("avg_free"),
                        "min_free": result.get("min_free"),
                        "max_free": result.get("max_free"),
                    }
                tool_calls_made.append(summary)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)[:1500],
                })

        # Hit iteration cap — make one final call without tools to force a text answer
        final = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024,
        )
        return jsonify({
            "response": final.choices[0].message.content or "",
            "tool_calls": tool_calls_made,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent/confirm", methods=["POST"])
def api_agent_confirm():
    """User clicked Approve on a pending action. Execute it."""
    body = request.get_json(force=True) or {}
    action_id = body.get("action_id", "")
    if not action_id:
        return jsonify({"ok": False, "error": "missing action_id"}), 400
    result = execute_pending(action_id)
    return jsonify(result)


@app.route("/api/agent/reject", methods=["POST"])
def api_agent_reject():
    """User clicked Reject. Drop the pending action."""
    body = request.get_json(force=True) or {}
    action_id = body.get("action_id", "")
    rec = _take_pending(action_id)
    if rec:
        _agent_log("reject", rec["tool"], rec["args"])
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# HTML Dashboard Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPU Cluster Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d2e;
    --border: #2a2d3e;
    --text: #e1e4ed;
    --text-dim: #8b8fa3;
    --accent: #6c5ce7;
    --green: #00b894;
    --yellow: #fdcb6e;
    --red: #e17055;
    --orange: #f39c12;
    --blue: #74b9ff;
    --gray: #636e72;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 20px;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .header h1 { font-size: 1.5rem; font-weight: 600; color: var(--accent); }
  .header .meta { font-size: 0.8rem; color: var(--text-dim); }

  /* ---- Tabs ---- */
  .tab-bar {
    display: flex;
    gap: 0;
    margin-bottom: 20px;
    border-bottom: 2px solid var(--border);
  }
  .tab-btn {
    padding: 10px 24px;
    background: none;
    border: none;
    color: var(--text-dim);
    font-family: inherit;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .tab-btn .tab-badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 1px 6px;
    border-radius: 8px;
    margin-left: 6px;
    vertical-align: middle;
    background: rgba(108, 92, 231, 0.15);
    color: var(--accent);
  }

  .stats-bar { display: flex; gap: 20px; margin-bottom: 24px; flex-wrap: wrap; }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 20px;
    min-width: 140px;
  }
  .stat-card .label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }
  .stat-card .value { font-size: 1.4rem; font-weight: 700; margin-top: 4px; }
  .stat-card .value.green { color: var(--green); }
  .stat-card .value.yellow { color: var(--yellow); }
  .stat-card .value.red { color: var(--red); }
  .stat-card .value.blue { color: var(--blue); }

  .filter-bar { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
  .filter-bar label { font-size: 0.75rem; color: var(--text-dim); margin-right: 4px; }
  .filter-bar select, .filter-bar input {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 6px; font-size: 0.8rem; font-family: inherit;
  }

  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  thead th {
    background: var(--surface); padding: 10px 12px; text-align: left; font-weight: 600;
    color: var(--text-dim); text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px;
    border-bottom: 2px solid var(--border); position: sticky; top: 0; cursor: pointer; user-select: none;
  }
  thead th:hover { color: var(--accent); }
  thead th .sort-arrow { margin-left: 4px; font-size: 0.6rem; }
  tbody tr { border-bottom: 1px solid var(--border); transition: background 0.15s; }
  tbody tr:hover { background: rgba(108, 92, 231, 0.08); }
  td { padding: 10px 12px; vertical-align: top; }

  .state-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }
  .state-idle { background: rgba(0, 184, 148, 0.15); color: var(--green); }
  .state-mixed { background: rgba(253, 203, 110, 0.15); color: var(--yellow); }
  .state-allocated { background: rgba(225, 112, 85, 0.15); color: var(--red); }
  .state-down { background: rgba(99, 110, 114, 0.2); color: var(--gray); }
  .state-reserved { background: rgba(116, 185, 255, 0.15); color: var(--blue); }
  .state-draining { background: rgba(243, 156, 18, 0.15); color: var(--orange); }

  .gpu-type { color: var(--blue); font-weight: 500; }
  .job-list { list-style: none; padding: 0; }
  .job-list li {
    margin-bottom: 4px; padding: 4px 8px; background: rgba(255,255,255,0.03);
    border-radius: 4px; font-size: 0.75rem; display: flex; gap: 8px; align-items: center;
  }
  .job-id { color: var(--accent); font-weight: 600; min-width: 75px; }
  .job-user { color: var(--green); min-width: 70px; }
  .job-name { color: var(--text-dim); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 200px; }
  .job-time { color: var(--yellow); min-width: 90px; text-align: right; }

  .mem-bar { width: 80px; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 6px; }
  .mem-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }

  .pulse { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse-anim 2s infinite; margin-right: 6px; }
  @keyframes pulse-anim { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .no-jobs { color: var(--text-dim); font-style: italic; font-size: 0.75rem; }
  .loading { text-align: center; padding: 40px; color: var(--text-dim); }
  footer { margin-top: 24px; text-align: center; font-size: 0.7rem; color: var(--text-dim); }

  /* ---- Chat Sidebar ---- */
  body.chat-open { padding-right: 540px; transition: padding-right 0.25s ease; }
  body { transition: padding-right 0.25s ease; }
  .chat-toggle {
    position: fixed; bottom: 24px; right: 24px; width: 52px; height: 52px;
    border-radius: 50%; background: var(--accent); color: #fff; border: none;
    font-size: 1.4rem; cursor: pointer; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    z-index: 1000; transition: transform 0.2s, right 0.25s ease;
  }
  .chat-toggle:hover { transform: scale(1.1); }
  body.chat-open .chat-toggle { right: 556px; }
  .chat-panel {
    position: fixed; top: 0; right: 0; bottom: 0; width: 520px;
    background: var(--surface); border-left: 1px solid var(--border);
    display: none; flex-direction: column; z-index: 999;
    box-shadow: -8px 0 32px rgba(0,0,0,0.5);
  }
  .chat-panel.open { display: flex; }
  .chat-header {
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    font-weight: 600; font-size: 1rem; color: var(--accent);
    display: flex; justify-content: space-between; align-items: center;
  }
  .chat-close-btn {
    background: none; border: none; color: var(--text-dim); font-size: 1.4rem;
    cursor: pointer; padding: 0 4px; line-height: 1;
  }
  .chat-close-btn:hover { color: var(--text); }
  .chat-messages {
    flex: 1; overflow-y: auto; padding: 16px 20px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .chat-msg { padding: 12px 16px; border-radius: 10px; font-size: 0.92rem; line-height: 1.6; max-width: 95%; word-wrap: break-word; white-space: pre-wrap; }
  .chat-msg.user { background: rgba(108,92,231,0.15); color: var(--text); align-self: flex-end; }
  .chat-msg.assistant { background: rgba(255,255,255,0.05); color: var(--text); align-self: flex-start; }
  .chat-msg.assistant code { background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 3px; font-size: 0.85rem; }
  .chat-msg.assistant strong { color: var(--accent); }
  .chat-msg.error { background: rgba(225,112,85,0.15); color: var(--red); }
  .chat-input-row {
    display: flex; gap: 10px; padding: 16px 20px; border-top: 1px solid var(--border);
  }
  .chat-input-row input {
    flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 12px 14px; border-radius: 8px; font-family: inherit; font-size: 0.9rem;
  }
  .chat-input-row input:focus { outline: none; border-color: var(--accent); }
  .chat-input-row button {
    background: var(--accent); color: #fff; border: none; padding: 12px 20px;
    border-radius: 8px; cursor: pointer; font-family: inherit; font-size: 0.9rem; font-weight: 600;
  }
  .chat-input-row button:disabled { opacity: 0.5; cursor: not-allowed; }
  .chat-typing { color: var(--text-dim); font-size: 0.85rem; font-style: italic; padding: 6px 16px; }
  .tool-call-card {
    align-self: flex-start; max-width: 95%; background: rgba(116,185,255,0.08);
    border: 1px solid rgba(116,185,255,0.25); border-radius: 8px;
    padding: 10px 14px; font-size: 0.85rem; color: var(--text-dim);
  }

  /* ---- Approval card ---- */
  .approval-card {
    align-self: flex-start; max-width: 95%; background: rgba(243, 156, 18, 0.10);
    border: 1px solid rgba(243, 156, 18, 0.45); border-radius: 8px;
    padding: 12px 14px; font-size: 0.88rem; color: var(--text);
  }
  .approval-card .ap-head { color: var(--orange); font-weight: 700; margin-bottom: 8px; }
  .approval-card pre {
    background: rgba(0,0,0,0.35); padding: 8px 10px; border-radius: 6px;
    font-size: 0.78rem; overflow-x: auto; margin-bottom: 8px; white-space: pre-wrap;
  }
  .approval-card .ap-buttons { display: flex; gap: 8px; margin-top: 6px; }
  .approval-card button {
    padding: 6px 14px; border-radius: 6px; border: none; font-family: inherit;
    font-size: 0.82rem; font-weight: 600; cursor: pointer;
  }
  .approval-card .ap-approve { background: var(--green); color: #fff; }
  .approval-card .ap-reject  { background: var(--gray); color: #fff; }
  .approval-card .ap-status  { font-size: 0.78rem; margin-top: 6px; color: var(--text-dim); }

  /* ---- Trends view ---- */
  .trends-view { display: none; }
  .trends-view.active { display: block; }
  .table-view.hidden { display: none; }
  .chart-row {
    display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 20px;
  }
  .chart-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .chart-card h3 { font-size: 0.9rem; color: var(--text-dim);
                   text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
  .chart-card canvas { max-height: 280px; }
  .heatmap-grid {
    display: grid; grid-template-columns: 40px repeat(24, 1fr); gap: 2px;
    font-size: 0.65rem;
  }
  .heatmap-grid .hm-corner { color: var(--text-dim); text-align: right; padding-right: 4px; }
  .heatmap-grid .hm-day-label { color: var(--text-dim); text-align: right; padding-right: 4px; }
  .heatmap-grid .hm-hour-label { color: var(--text-dim); text-align: center; }
  .heatmap-grid .hm-cell {
    aspect-ratio: 1; border-radius: 2px; background: var(--border);
  }
  .heatmap-grid .hm-cell[title]:hover { outline: 1px solid var(--accent); }
  .trends-controls {
    display: flex; gap: 10px; align-items: center; margin-bottom: 16px;
    font-size: 0.8rem; color: var(--text-dim);
  }
  .trends-controls select {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 6px; font-family: inherit; font-size: 0.8rem;
  }
  .tool-call-card .tool-head { color: var(--blue); font-weight: 600; margin-bottom: 4px; }
  .tool-call-card .tool-query { color: var(--text); font-style: italic; margin-bottom: 6px; word-break: break-word; }
  .tool-call-card a { color: var(--accent); text-decoration: none; display: block; margin: 2px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .tool-call-card a:hover { text-decoration: underline; }
  .tool-call-card .tool-error { color: var(--red); }
</style>
</head>
<body>
<div class="header">
  <h1>GPU Cluster Monitor</h1>
  <div class="meta">
    <span class="pulse"></span>
    <span id="update-time">Loading...</span>
    &nbsp;|&nbsp; Refresh: 5s
  </div>
</div>

<div class="tab-bar" id="tab-bar">
  <button class="tab-btn active" data-view="cluster" data-cluster="cs">CS <span class="tab-badge" id="badge-cs">--</span></button>
  <button class="tab-btn" data-view="cluster" data-cluster="hpc">Rivanna <span class="tab-badge" id="badge-hpc">--</span></button>
  <button class="tab-btn" data-view="trends">📈 Trends</button>
</div>

<!-- ===== Cluster (table) view ===== -->
<div class="table-view" id="table-view">
  <div class="stats-bar" id="stats-bar"></div>

  <div class="filter-bar">
    <label>State:</label>
    <select id="filter-state">
      <option value="all">All</option>
      <option value="idle">Idle</option>
      <option value="mixed">Mixed</option>
      <option value="allocated">Allocated</option>
      <option value="reserved">Reserved</option>
      <option value="down">Down</option>
    </select>
    <label>GPU Type:</label>
    <select id="filter-gpu"></select>
    <label>Search:</label>
    <input type="text" id="filter-search" placeholder="node, user, job...">
  </div>

  <table>
    <thead>
      <tr>
        <th data-sort="node">Node <span class="sort-arrow"></span></th>
        <th data-sort="partition">Partition <span class="sort-arrow"></span></th>
        <th data-sort="gpu_tier">GPU Type <span class="sort-arrow"></span></th>
        <th data-sort="gpus_free">GPUs (free/total) <span class="sort-arrow"></span></th>
        <th data-sort="cpus_free">CPUs (free/total) <span class="sort-arrow"></span></th>
        <th data-sort="mem_gb">Memory <span class="sort-arrow"></span></th>
        <th data-sort="state">State <span class="sort-arrow"></span></th>
        <th data-sort="cpu_load">Load <span class="sort-arrow"></span></th>
        <th>Running Jobs</th>
      </tr>
    </thead>
    <tbody id="node-table"><tr><td colspan="9" class="loading">Loading...</td></tr></tbody>
  </table>
</div>

<!-- ===== Trends view ===== -->
<div class="trends-view" id="trends-view">
  <div class="trends-controls">
    <label>Cluster:</label>
    <select id="trends-cluster">
      <option value="cs">CS</option>
      <option value="hpc">Rivanna</option>
    </select>
    <label>Window:</label>
    <select id="trends-hours">
      <option value="6">Last 6h</option>
      <option value="24" selected>Last 24h</option>
      <option value="72">Last 3 days</option>
      <option value="168">Last 7 days</option>
    </select>
    <label>Heatmap days:</label>
    <select id="trends-heatmap-days">
      <option value="7" selected>7</option>
      <option value="14">14</option>
      <option value="30">30</option>
    </select>
    <span id="trends-status" style="color:var(--text-dim)"></span>
  </div>

  <div class="chart-card">
    <h3>Free GPUs over time</h3>
    <canvas id="trends-line"></canvas>
  </div>

  <div class="chart-card" style="margin-top:20px">
    <h3>Average free GPUs by hour of week (Mon→Sun, 0→23h)</h3>
    <div id="trends-heatmap" class="heatmap-grid"></div>
    <div style="font-size:0.7rem;color:var(--text-dim);margin-top:8px">
      Greener = more free GPUs on average.
      Empty cells mean no data was collected at that hour yet (the dashboard records snapshots
      once per minute while it's running).
    </div>
  </div>
</div>

<footer>GPU Cluster Monitor</footer>

<!-- Chat Panel -->
<button class="chat-toggle" id="chat-toggle" title="Ask about a job">?</button>
<div class="chat-panel" id="chat-panel">
  <div class="chat-header">
    <span>SLURM Job Assistant <span style="font-size:0.75rem;color:var(--text-dim);font-weight:400">CS + Rivanna</span></span>
    <button class="chat-close-btn" id="chat-close" title="Close">×</button>
  </div>
  <div class="chat-messages" id="chat-messages">
    <div class="chat-msg assistant">Hi! Paste a job ID and I'll analyze both clusters to help you get scheduled faster.</div>
  </div>
  <div class="chat-input-row">
    <input type="text" id="chat-input" placeholder="Enter job ID or question..." autocomplete="off">
    <button id="chat-send">Send</button>
  </div>
</div>

<script>
// ---- State ----
let activeCluster = "cs";
let sortCol = "gpu_tier";
let sortAsc = true;
const clusterData = {};   // { cs: {...}, hpc: {...} }
const clusterTimers = {};

const stateOrder = {mixed:0, allocated:1, reserved:2, idle:3, draining:4, down:5, unknown:6};

const gpuTier = {
  "H200":0,
  "H100 NVL":1, "H100":1,
  "A100 80GB":2, "A100 40GB":3,
  "RTX 5090":4, "RTX 5080":4,
  "A6000":5,
  "A40":6,
  "RTX A4500":7,
  "RTX 4000 Ada":8,
  "RTX A4000":9,
  "Quadro RTX 6000":10,
  "V100":11,
  "RTX 3090":11,
  "Quadro RTX 4000":12,
  "RTX 2080 Ti":13, "RTX 2080":13,
  "Titan Xp":14, "Titan X":14,
  "GTX 1080 Ti":15,
  "GTX 1080":16,
  "A16":17,
  "Tesla P100":18,
};
function getGpuTier(t) { return gpuTier[t] !== undefined ? gpuTier[t] : 99; }

function stateClass(s) {
  s = s.toLowerCase();
  if (s.includes("idle")) return "state-idle";
  if (s.includes("mix")) return "state-mixed";
  if (s.includes("alloc")) return "state-allocated";
  if (s.includes("down")) return "state-down";
  if (s.includes("reserv")) return "state-reserved";
  if (s.includes("drain")) return "state-draining";
  return "";
}
function stateKey(s) {
  s = s.toLowerCase();
  for (const k of Object.keys(stateOrder)) { if (s.includes(k)) return stateOrder[k]; }
  return 6;
}
function memBarColor(pct) {
  if (pct > 80) return "var(--red)";
  if (pct > 50) return "var(--yellow)";
  return "var(--green)";
}

// ---- Render ----
function renderStats(data) {
  const nodes = data.nodes;
  const idle = nodes.filter(n => n.state.toLowerCase().includes("idle")).length;
  const busy = nodes.filter(n => ["mixed","allocated"].some(s => n.state.toLowerCase().includes(s))).length;
  const down = nodes.filter(n => n.state.toLowerCase().includes("down")).length;
  document.getElementById("stats-bar").innerHTML = `
    <div class="stat-card"><div class="label">GPU Nodes</div><div class="value blue">${data.total_gpu_nodes}</div></div>
    <div class="stat-card"><div class="label">GPUs</div><div class="value green">${data.free_gpus} <span style="font-size:0.7rem;color:var(--text-dim)">/ ${data.total_gpus}</span></div></div>
    <div class="stat-card"><div class="label">CPUs</div><div class="value green">${data.free_cpus} <span style="font-size:0.7rem;color:var(--text-dim)">/ ${data.total_cpus}</span></div></div>
    <div class="stat-card"><div class="label">Idle Nodes</div><div class="value green">${idle}</div></div>
    <div class="stat-card"><div class="label">Busy Nodes</div><div class="value yellow">${busy}</div></div>
    <div class="stat-card"><div class="label">Down Nodes</div><div class="value red">${down}</div></div>
  `;
}

function renderTable(data) {
  const fs = document.getElementById("filter-state").value;
  const fg = document.getElementById("filter-gpu").value;
  const fq = document.getElementById("filter-search").value.toLowerCase();

  let nodes = data.nodes.filter(n => {
    if (fs !== "all" && !n.state.toLowerCase().includes(fs)) return false;
    if (fg && fg !== "all" && n.gpu_type !== fg) return false;
    if (fq && !JSON.stringify(n).toLowerCase().includes(fq)) return false;
    return true;
  });

  nodes.sort((a,b) => {
    let va, vb;
    if (sortCol === "gpu_tier") { va = getGpuTier(a.gpu_type); vb = getGpuTier(b.gpu_type); }
    else if (sortCol === "state") { va = stateKey(a.state); vb = stateKey(b.state); }
    else if (["gpu_count","gpus_free","cpus","cpus_free","mem_gb","cpu_load"].includes(sortCol)) {
      va = a[sortCol]; vb = b[sortCol];
    } else { va = String(a[sortCol]||""); vb = String(b[sortCol]||""); }
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  document.getElementById("node-table").innerHTML = nodes.map(n => {
    const usedMem = n.mem_gb - n.free_mem_gb;
    const memPct = n.mem_gb > 0 ? (usedMem / n.mem_gb * 100) : 0;
    const jobsHtml = n.jobs.length > 0
      ? `<ul class="job-list">${n.jobs.map(j => `<li>
            <span class="job-id">${j.job_id}</span>
            <span class="job-user">${j.user}</span>
            <span class="job-name" title="${j.name}">${j.name}</span>
            <span class="job-time">${j.time}</span>
          </li>`).join("")}</ul>`
      : `<span class="no-jobs">-- none --</span>`;
    return `<tr>
      <td><strong>${n.node}</strong></td>
      <td>${n.partition}</td>
      <td class="gpu-type">${n.gpu_type}</td>
      <td><strong style="color:${n.gpus_free>0?'var(--green)':'var(--red)'}">${n.gpus_free}</strong> <span style="color:var(--text-dim)">/ ${n.gpu_count}</span></td>
      <td><strong style="color:${n.cpus_free>0?'var(--green)':'var(--red)'}">${n.cpus_free}</strong> <span style="color:var(--text-dim)">/ ${n.cpus}</span></td>
      <td>${n.mem_gb} GB <div class="mem-bar"><div class="mem-bar-fill" style="width:${memPct}%;background:${memBarColor(memPct)}"></div></div>
        <br><span style="font-size:0.7rem;color:var(--text-dim)">${n.free_mem_gb} GB free</span></td>
      <td><span class="state-badge ${stateClass(n.state)}">${n.state}</span></td>
      <td>${n.cpu_load.toFixed(1)}</td>
      <td>${jobsHtml}</td>
    </tr>`;
  }).join("");
}

function populateGpuFilter(data) {
  const sel = document.getElementById("filter-gpu");
  const types = [...new Set(data.nodes.map(n => n.gpu_type))].sort();
  const cur = sel.value;
  sel.innerHTML = `<option value="all">All</option>` + types.map(t => `<option value="${t}">${t}</option>`).join("");
  if (cur) sel.value = cur;
}

function renderActive() {
  const data = clusterData[activeCluster];
  if (!data) {
    document.getElementById("node-table").innerHTML = `<tr><td colspan="9" class="loading">Loading ${activeCluster}...</td></tr>`;
    return;
  }
  document.getElementById("update-time").textContent = "Updated: " + data.timestamp;
  renderStats(data);
  populateGpuFilter(data);
  renderTable(data);
}

function updateBadge(id, data) {
  const el = document.getElementById("badge-" + id);
  if (el && data) {
    el.textContent = data.free_gpus + " free";
  }
}

// ---- Fetch ----
async function fetchCluster(id) {
  try {
    const resp = await fetch("/api/data/" + id);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    clusterData[id] = await resp.json();
    updateBadge(id, clusterData[id]);
    if (id === activeCluster) renderActive();
  } catch(e) {
    console.error("Fetch error for " + id + ":", e);
  }
}

function fetchAll() {
  fetchCluster("cs");
  fetchCluster("hpc");
}

// ---- Tab switching ----
const tableView = document.getElementById("table-view");
const trendsView = document.getElementById("trends-view");

function showView(view) {
  if (view === "trends") {
    tableView.classList.add("hidden");
    trendsView.classList.add("active");
    refreshTrends();
  } else {
    tableView.classList.remove("hidden");
    trendsView.classList.remove("active");
  }
}

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    if (btn.dataset.view === "trends") {
      showView("trends");
    } else {
      activeCluster = btn.dataset.cluster;
      showView("cluster");
      renderActive();
    }
  });
});

// ---- Sort headers ----
document.querySelectorAll("thead th[data-sort]").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.sort;
    if (sortCol === col) { sortAsc = !sortAsc; }
    else { sortCol = col; sortAsc = true; }
    document.querySelectorAll("thead th .sort-arrow").forEach(s => s.textContent = "");
    th.querySelector(".sort-arrow").textContent = sortAsc ? " ▲" : " ▼";
    if (clusterData[activeCluster]) renderTable(clusterData[activeCluster]);
  });
});

// ---- Filter ----
["filter-state","filter-gpu","filter-search"].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener("input", () => { if (clusterData[activeCluster]) renderTable(clusterData[activeCluster]); });
  el.addEventListener("change", () => { if (clusterData[activeCluster]) renderTable(clusterData[activeCluster]); });
});

// ---- Trends ----
let trendsLineChart = null;

async function refreshTrends() {
  const cluster = document.getElementById("trends-cluster").value;
  const hours   = document.getElementById("trends-hours").value;
  const days    = document.getElementById("trends-heatmap-days").value;
  const status  = document.getElementById("trends-status");
  status.textContent = "Loading...";
  try {
    const [seriesResp, heatResp] = await Promise.all([
      fetch(`/api/history/${cluster}?hours=${hours}`).then(r => r.json()),
      fetch(`/api/history/${cluster}/heatmap?days=${days}`).then(r => r.json()),
    ]);
    renderTrendsLine(seriesResp);
    renderHeatmap(heatResp);
    status.textContent = `${seriesResp.series.length} samples`;
  } catch (e) {
    status.textContent = "Failed: " + e.message;
  }
}

function renderTrendsLine(data) {
  const ctx = document.getElementById("trends-line").getContext("2d");
  const labels = data.series.map(p => {
    const d = new Date(p.ts * 1000);
    return d.toLocaleString([], {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit"});
  });
  const free  = data.series.map(p => p.free_gpus);
  const total = data.series.map(p => p.total_gpus);

  if (trendsLineChart) trendsLineChart.destroy();
  trendsLineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        { label: "Free GPUs",  data: free,  borderColor: "#00b894",
          backgroundColor: "rgba(0,184,148,0.15)", fill: true, tension: 0.25, pointRadius: 0 },
        { label: "Total GPUs", data: total, borderColor: "#74b9ff",
          borderDash: [4,4], fill: false, tension: 0, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: {mode: "index", intersect: false},
      plugins: { legend: { labels: { color: "#e1e4ed" } } },
      scales: {
        x: { ticks: { color: "#8b8fa3", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#8b8fa3" }, grid: { color: "rgba(255,255,255,0.05)" }, beginAtZero: true },
      },
    },
  });
}

function renderHeatmap(data) {
  const grid = data.grid; // 7x24
  const container = document.getElementById("trends-heatmap");
  // Find max for scaling
  let max = 0;
  for (let d=0; d<7; d++) for (let h=0; h<24; h++) {
    if (grid[d][h] != null && grid[d][h] > max) max = grid[d][h];
  }
  const days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
  let html = `<div class="hm-corner"></div>`;
  for (let h=0; h<24; h++) html += `<div class="hm-hour-label">${h}</div>`;
  for (let d=0; d<7; d++) {
    html += `<div class="hm-day-label">${days[d]}</div>`;
    for (let h=0; h<24; h++) {
      const v = grid[d][h];
      if (v == null) {
        html += `<div class="hm-cell" title="${days[d]} ${h}:00 — no data"></div>`;
      } else {
        const ratio = max > 0 ? v / max : 0;
        // Green scale: 0=dim, 1=bright
        const alpha = 0.15 + 0.85 * ratio;
        html += `<div class="hm-cell" title="${days[d]} ${h}:00 — avg ${v} free" `
              + `style="background:rgba(0,184,148,${alpha.toFixed(2)})"></div>`;
      }
    }
  }
  container.innerHTML = html;
}

["trends-cluster","trends-hours","trends-heatmap-days"].forEach(id => {
  document.getElementById(id).addEventListener("change", refreshTrends);
});

// ---- Chat ----
const chatToggle = document.getElementById("chat-toggle");
const chatPanel = document.getElementById("chat-panel");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
let chatHistory = [];
let chatBusy = false;

function setChatOpen(open) {
  chatPanel.classList.toggle("open", open);
  document.body.classList.toggle("chat-open", open);
  if (open) chatInput.focus();
}

chatToggle.addEventListener("click", () => {
  setChatOpen(!chatPanel.classList.contains("open"));
});
document.getElementById("chat-close").addEventListener("click", () => setChatOpen(false));

function escapeHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

function appendChat(role, text) {
  const div = document.createElement("div");
  div.className = "chat-msg " + role;
  // Basic markdown: **bold**, `code`, newlines
  let html = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  div.innerHTML = html;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendApprovalCard(call) {
  const p = call.pending;
  const div = document.createElement("div");
  div.className = "approval-card";
  const tool = p.tool;
  const args = p.args || {};
  let preview = "";
  if (tool === "submit_job") {
    preview += `cluster:    ${args.cluster_id || ""}\n`;
    preview += `partition:  ${args.partition || ""}\n`;
    preview += `job_name:   ${args.job_name || "agent_job"}\n`;
    preview += `time:       ${args.time || "01:00:00"}\n`;
    preview += `cpus:       ${args.cpus || 1}\n`;
    preview += `mem:        ${args.mem || "4G"}\n`;
    if (args.gpus) preview += `gpus:       ${args.gpus}${args.gpu_type ? " (" + args.gpu_type + ")" : ""}\n`;
    if (args.workdir) preview += `workdir:    ${args.workdir}\n`;
    preview += `command:    ${args.command || ""}\n`;
  } else if (tool === "cancel_job") {
    preview += `cluster:    ${args.cluster_id || ""}\n`;
    preview += `job_id:     ${args.job_id || ""}`;
  } else {
    preview = JSON.stringify(args, null, 2);
  }
  const heading = tool === "submit_job"
    ? "⚡ Agent wants to submit a SLURM job"
    : (tool === "cancel_job" ? "⚡ Agent wants to cancel a SLURM job" : "⚡ Agent action proposal");
  div.innerHTML = `
    <div class="ap-head">${heading}</div>
    <pre>${escapeHtml(preview)}</pre>
    <div class="ap-buttons">
      <button class="ap-approve">Approve & Run</button>
      <button class="ap-reject">Reject</button>
    </div>
    <div class="ap-status"></div>
  `;
  const statusEl = div.querySelector(".ap-status");
  div.querySelector(".ap-approve").addEventListener("click", async () => {
    div.querySelector(".ap-approve").disabled = true;
    div.querySelector(".ap-reject").disabled = true;
    statusEl.textContent = "Running...";
    try {
      const resp = await fetch("/api/agent/confirm", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({action_id: p.action_id}),
      });
      const result = await resp.json();
      if (result.ok) {
        if (result.job_id) {
          statusEl.innerHTML = `<span style="color:var(--green)">✓ Done — job ${escapeHtml(result.job_id)}</span>`;
        } else {
          statusEl.innerHTML = `<span style="color:var(--green)">✓ Done</span>`;
        }
      } else {
        statusEl.innerHTML = `<span style="color:var(--red)">✗ ${escapeHtml(result.error || "failed")}</span>`;
      }
      if (result.output) {
        const out = document.createElement("pre");
        out.textContent = result.output;
        out.style.marginTop = "8px";
        div.appendChild(out);
      }
    } catch (e) {
      statusEl.innerHTML = `<span style="color:var(--red)">✗ ${escapeHtml(e.message)}</span>`;
    }
  });
  div.querySelector(".ap-reject").addEventListener("click", async () => {
    div.querySelector(".ap-approve").disabled = true;
    div.querySelector(".ap-reject").disabled = true;
    statusEl.textContent = "Rejected.";
    try {
      await fetch("/api/agent/reject", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({action_id: p.action_id}),
      });
    } catch (e) { /* ignore */ }
  });
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendToolCallCard(call) {
  // Pending mutating actions get their own card type with Approve/Reject buttons.
  if (call.pending) {
    appendApprovalCard(call);
    return;
  }
  const div = document.createElement("div");
  div.className = "tool-call-card";
  let inner = "";

  if (call.name === "query_history") {
    const h = call.history || {};
    inner += `<div class="tool-head">📈 query_history</div>`;
    inner += `<div class="tool-query">${escapeHtml(h.cluster || "")}${h.gpu_type ? " · " + escapeHtml(h.gpu_type) : ""} · last ${escapeHtml(h.lookback_hours || 24)}h</div>`;
    if (call.error) {
      inner += `<div class="tool-error">${escapeHtml(call.error)}</div>`;
    } else {
      const bits = [];
      if (h.samples != null) bits.push(escapeHtml(h.samples + " samples"));
      if (h.avg_free != null) bits.push("avg free: <strong>" + escapeHtml(h.avg_free) + "</strong>");
      if (h.min_free != null && h.max_free != null) bits.push(escapeHtml(`range ${h.min_free}–${h.max_free}`));
      inner += `<div>${bits.join(" · ")}</div>`;
    }
    div.innerHTML = inner;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return;
  }

  if (call.name === "analyze_workspace") {
    const a = call.analysis || {};
    const path = a.path || (call.args && call.args.path) || "";
    inner += `<div class="tool-head">📂 analyze_workspace</div>`;
    if (path) inner += `<div class="tool-query">${escapeHtml(path)}</div>`;
    if (call.error) {
      inner += `<div class="tool-error">${escapeHtml(call.error)}</div>`;
    } else {
      const bits = [];
      if (a.files_read != null) bits.push(escapeHtml(a.files_read + " files read"));
      if (a.gpu_libs && a.gpu_libs.length) bits.push("libs: " + escapeHtml(a.gpu_libs.join(", ")));
      if (a.model_hints && a.model_hints.length) bits.push("models: " + escapeHtml(a.model_hints.join(", ")));
      if (a.size_hints && a.size_hints.length) bits.push("sizes: " + escapeHtml(a.size_hints.join(", ")));
      if (a.pressure) bits.push("pressure: <strong>" + escapeHtml(a.pressure) + "</strong>");
      inner += `<div>${bits.join(" · ")}</div>`;
    }
  } else if (call.name === "analyze_repository") {
    const a = call.analysis || {};
    const repo = a.repo || (call.args && call.args.url) || "";
    inner += `<div class="tool-head">🌐 analyze_repository</div>`;
    if (repo) inner += `<div class="tool-query">${escapeHtml(repo)}</div>`;
    if (call.error) {
      inner += `<div class="tool-error">${escapeHtml(call.error)}</div>`;
    } else {
      const bits = [];
      if (a.files_fetched != null) bits.push(escapeHtml(a.files_fetched + " files fetched"));
      if (a.gpu_libs && a.gpu_libs.length) bits.push("libs: " + escapeHtml(a.gpu_libs.join(", ")));
      if (a.model_hints && a.model_hints.length) bits.push("models: " + escapeHtml(a.model_hints.join(", ")));
      if (a.size_hints && a.size_hints.length) bits.push("sizes: " + escapeHtml(a.size_hints.join(", ")));
      if (a.pressure) bits.push("pressure: <strong>" + escapeHtml(a.pressure) + "</strong>");
      inner += `<div>${bits.join(" · ")}</div>`;
    }
  } else {
    // web_search (default)
    const query = (call.args && call.args.query) || "";
    inner += `<div class="tool-head">🔍 ${escapeHtml(call.name)}</div>`;
    if (query) inner += `<div class="tool-query">${escapeHtml(query)}</div>`;
    if (call.error) {
      inner += `<div class="tool-error">${escapeHtml(call.error)}</div>`;
    } else if (call.results && call.results.length) {
      inner += call.results.map(r =>
        `<a href="${escapeHtml(r.url)}" target="_blank" rel="noopener" title="${escapeHtml(r.url)}">› ${escapeHtml(r.title || r.url)}</a>`
      ).join("");
    } else {
      inner += `<div>(no results)</div>`;
    }
  }
  div.innerHTML = inner;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}


async function sendChat() {
  const msg = chatInput.value.trim();
  if (!msg || chatBusy) return;
  chatBusy = true;
  chatSend.disabled = true;
  chatInput.value = "";
  appendChat("user", msg);

  const typing = document.createElement("div");
  typing.className = "chat-typing";
  typing.textContent = "Thinking...";
  chatMessages.appendChild(typing);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: msg, history: chatHistory}),
    });
    const data = await resp.json();
    typing.remove();
    if (data.error) {
      appendChat("error", "Error: " + data.error);
    } else {
      if (Array.isArray(data.tool_calls)) {
        data.tool_calls.forEach(appendToolCallCard);
      }
      appendChat("assistant", data.response);
      chatHistory.push({role:"user", content:msg});
      chatHistory.push({role:"assistant", content:data.response});
      if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
    }
  } catch(e) {
    typing.remove();
    appendChat("error", "Network error: " + e.message);
  }
  chatBusy = false;
  chatSend.disabled = false;
}

chatSend.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } });

// ---- Init ----
fetchAll();
setInterval(fetchAll, 5000);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _update_public_url(public_url):
    """Write public_url to state.json and ~/public_url."""
    print(f"Cloudflare tunnel URL: {public_url}", flush=True)
    state_file = os.path.join(os.path.dirname(__file__), "state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                state = json.load(f)
            state["public_url"] = public_url
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
    url_file = os.path.expanduser("~/public_url")
    with open(url_file, "w") as f:
        f.write(public_url + "\n")


def run_cloudflare_tunnel_forever(port):
    """Start cloudflared and restart it whenever it dies."""
    attempt = 0
    while True:
        attempt += 1
        print(f"[cloudflared] Starting tunnel (attempt {attempt})...", flush=True)
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            # Read until we find the URL
            for line in proc.stdout:
                print(f"[cloudflared] {line.rstrip()}", flush=True)
                m = re.search(r"(https://[a-zA-Z0-9_-]+\.trycloudflare\.com)", line)
                if m:
                    _update_public_url(m.group(1))
                    break
            # Drain remaining output until process exits
            for line in proc.stdout:
                print(f"[cloudflared] {line.rstrip()}", flush=True)
            proc.wait()
            print(f"[cloudflared] Tunnel exited (code {proc.returncode}), restarting in 10s...", flush=True)
        except Exception as e:
            print(f"[cloudflared] Error: {e}, restarting in 10s...", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    port = int(os.environ.get("MONITOR_PORT", 8080))
    host = os.environ.get("MONITOR_HOST", "0.0.0.0")
    print(f"Starting GPU Cluster Monitor on {host}:{port}", flush=True)

    # Start Cloudflare tunnel in background thread (needs Flask to be listening first)
    def _delayed_tunnel():
        time.sleep(3)  # wait for Flask to bind the port
        run_cloudflare_tunnel_forever(port)
    threading.Thread(target=_delayed_tunnel, daemon=True).start()

    # Start the historical snapshot collector
    threading.Thread(target=snapshot_loop, args=(60,), daemon=True).start()

    app.run(host=host, port=port, debug=False, threaded=True)
