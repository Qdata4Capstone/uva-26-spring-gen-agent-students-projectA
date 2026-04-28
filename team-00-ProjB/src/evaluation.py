#!/usr/bin/env python3
"""
GPU Cluster Monitor - Evaluation Suite
======================================

Three independent tests:

  correctness  Validate dashboard GPU/CPU/mem stats against direct SLURM queries.
  latency      Measure /api/data/<cluster> response time. Target: p95 < 5s.
  reliability  Sample auto-renewal chain health and tunnel availability.

Usage
-----
  python evaluation.py correctness
  python evaluation.py latency --samples 20
  python evaluation.py reliability --duration 10 --interval 15
  python evaluation.py reliability --duration 10 --active-tunnel-test
  python evaluation.py all

The script reads `state.json` to find the dashboard URL automatically. If your
dashboard is not running, all three tests will fail with a clear message.

This script is INTENTIONALLY independent of monitor.py — it re-parses SLURM
output with its own minimal parser so we're not just testing the parser
against itself.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: `requests` not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = SCRIPT_DIR / "state.json"

# Cluster SSH config — keep in sync with monitor.py CLUSTERS dict
CLUSTERS = {
    "cs":  {"name": "CS",      "ssh": None},
    "hpc": {"name": "Rivanna", "ssh": "ssh -o ConnectTimeout=10 -o BatchMode=yes tsx4zn@login.hpc.virginia.edu"},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_dashboard_url():
    """Read state.json and return the local dashboard base URL, e.g. http://node:8080."""
    if not STATE_FILE.exists():
        die(f"state.json not found at {STATE_FILE}. Is the dashboard running? Try `bash start.sh`.")
    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception as e:
        die(f"Failed to parse state.json: {e}")
    url = state.get("url")
    if not url:
        die("state.json has no 'url' field.")
    return url, state


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def run_shell(cmd, ssh_prefix=None, timeout=60):
    """Run a shell command (optionally via SSH) and return (stdout, stderr, returncode)."""
    if ssh_prefix:
        escaped = cmd.replace("'", "'\\''")
        cmd = f"{ssh_prefix} '{escaped}'"
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", 124


def fetch_dashboard_data(base_url, cluster_id, timeout=30):
    """GET /api/data/<cluster_id> and return the parsed JSON or raise."""
    r = requests.get(f"{base_url}/api/data/{cluster_id}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def percentile(data, pct):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# 1. Correctness — independent SLURM parser
# ---------------------------------------------------------------------------
def parse_gres(gres_str):
    """Parse SLURM Gres string like 'gpu:a100:4(S:0-1)' or 'gpu:rtx2080:8'.
    Returns the integer count of GPUs (0 if none)."""
    if not gres_str or gres_str.lower() in ("(null)", "null", "n/a"):
        return 0
    # Sum every gpu:NAME:N segment
    total = 0
    for m in re.finditer(r"gpu(?::[^:,()]+)?:(\d+)", gres_str.lower()):
        total += int(m.group(1))
    return total


def parse_cpustate(cpus_str):
    """Parse '%C' format A/I/O/T → (alloc, idle, other, total)."""
    parts = cpus_str.strip().split("/")
    if len(parts) != 4:
        return None
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def collect_ground_truth(cluster_id):
    """Run sinfo independently and return aggregated GPU/CPU totals + per-node map."""
    cluster = CLUSTERS[cluster_id]
    ssh = cluster["ssh"]
    # NodeHost | Gres | GresUsed | CPUsState (A/I/O/T) | StateLong
    cmd = 'sinfo -h -N -O "NodeHost:|,Gres:|,GresUsed:|,CPUsState:|,StateLong:|"'
    out, err, rc = run_shell(cmd, ssh, timeout=45)
    if rc != 0:
        return None, f"sinfo failed: {err.strip() or 'rc=' + str(rc)}"

    nodes = {}  # node -> {gpu_count, gpu_used, cpus_total, cpus_idle}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        node, gres, gres_used, cpus_state, _state = parts[0], parts[1], parts[2], parts[3], parts[4]
        if not node:
            continue
        gpu_total = parse_gres(gres)
        gpu_used = parse_gres(gres_used)
        cs = parse_cpustate(cpus_state)
        if cs is None:
            continue
        alloc, idle, _other, total = cs

        # A node can appear multiple times (one per partition); de-dup conservatively
        # by taking the max GPU count (it's the same node) and idle CPU count.
        existing = nodes.get(node)
        entry = {
            "gpu_count": gpu_total,
            "gpu_used": gpu_used,
            "cpus_total": total,
            "cpus_idle": idle,
        }
        if existing:
            for k in entry:
                entry[k] = max(existing[k], entry[k])
        nodes[node] = entry

    # GPU nodes only (these are what the dashboard cares about)
    gpu_nodes = {n: e for n, e in nodes.items() if e["gpu_count"] > 0}

    totals = {
        "total_gpus": sum(e["gpu_count"] for e in gpu_nodes.values()),
        "free_gpus": sum(max(0, e["gpu_count"] - e["gpu_used"]) for e in gpu_nodes.values()),
        "total_cpus": sum(e["cpus_total"] for e in gpu_nodes.values()),
        "free_cpus": sum(e["cpus_idle"] for e in gpu_nodes.values()),
        "total_gpu_nodes": len(gpu_nodes),
    }
    return {"totals": totals, "nodes": gpu_nodes}, None


def cmd_correctness(args):
    section("CORRECTNESS — dashboard vs. independent SLURM query")
    base_url, _ = load_dashboard_url()
    print(f"Dashboard: {base_url}")
    print(f"Tolerance: ±{args.gpu_tol} GPUs, ±{args.cpu_tol} CPUs (per cluster)")

    overall_pass = True
    for cluster_id in args.clusters:
        print(f"\n--- {CLUSTERS[cluster_id]['name']} ({cluster_id}) ---")
        # Fetch dashboard view
        try:
            t0 = time.time()
            dash = fetch_dashboard_data(base_url, cluster_id, timeout=60)
            dash_dt = time.time() - t0
        except Exception as e:
            print(f"  FAIL: dashboard fetch failed: {e}")
            overall_pass = False
            continue

        # Fetch ground truth
        t0 = time.time()
        gt, err = collect_ground_truth(cluster_id)
        gt_dt = time.time() - t0
        if err:
            print(f"  FAIL: ground truth failed: {err}")
            overall_pass = False
            continue

        # Compare totals with tolerance
        d = {k: dash.get(k, 0) for k in ("total_gpus", "free_gpus", "total_cpus", "free_cpus", "total_gpu_nodes")}
        g = gt["totals"]
        skew_sec = abs(dash_dt) + abs(gt_dt)
        print(f"  Dashboard fetched in {dash_dt:.2f}s, ground truth in {gt_dt:.2f}s (skew window ~{skew_sec:.1f}s)")
        print(f"  {'metric':<18} {'dashboard':>12} {'ground_truth':>14} {'delta':>8}  result")
        print(f"  {'-' * 64}")
        cluster_pass = True
        for key, tol in (
            ("total_gpus",      0),
            ("free_gpus",       args.gpu_tol),
            ("total_cpus",      0),
            ("free_cpus",       args.cpu_tol),
            ("total_gpu_nodes", 0),
        ):
            dv, gv = d[key], g[key]
            delta = dv - gv
            ok = abs(delta) <= tol
            mark = "OK" if ok else "FAIL"
            if not ok:
                cluster_pass = False
            print(f"  {key:<18} {dv:>12} {gv:>14} {delta:>+8}  {mark}")

        # Per-node disagreements (informational)
        dash_nodes = {n.get("node") or n.get("name"): n for n in dash.get("nodes", []) if n.get("gpu_count", 0) > 0}
        node_diffs = []
        for name, gnode in gt["nodes"].items():
            dn = dash_nodes.get(name)
            if not dn:
                node_diffs.append(f"missing in dashboard: {name}")
                continue
            d_free = dn.get("gpus_free", 0)
            g_free = max(0, gnode["gpu_count"] - gnode["gpu_used"])
            if abs(d_free - g_free) > args.gpu_tol:
                node_diffs.append(
                    f"{name}: dash free={d_free}, slurm free={g_free}"
                )
        if node_diffs:
            print(f"  Per-node disagreements ({len(node_diffs)}):")
            for nd in node_diffs[:10]:
                print(f"    - {nd}")
            if len(node_diffs) > 10:
                print(f"    ... and {len(node_diffs) - 10} more")

        if not cluster_pass:
            overall_pass = False
            print(f"  CLUSTER RESULT: FAIL")
        else:
            print(f"  CLUSTER RESULT: PASS")

    print()
    print(f"OVERALL CORRECTNESS: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


# ---------------------------------------------------------------------------
# 2. Latency
# ---------------------------------------------------------------------------
def cmd_latency(args):
    section(f"LATENCY — {args.samples} samples per cluster, target p95 < {args.target}s")
    base_url, _ = load_dashboard_url()
    print(f"Dashboard: {base_url}")

    overall_pass = True
    summary = []
    for cluster_id in args.clusters:
        name = CLUSTERS[cluster_id]["name"]
        ssh = CLUSTERS[cluster_id]["ssh"]
        print(f"\n--- {name} ({cluster_id}) ---")

        # Dashboard endpoint latency
        endpoint_times = []
        errors = 0
        for i in range(args.samples):
            t0 = time.perf_counter()
            try:
                r = requests.get(f"{base_url}/api/data/{cluster_id}", timeout=60)
                r.raise_for_status()
                endpoint_times.append(time.perf_counter() - t0)
                print(".", end="", flush=True)
            except Exception as e:
                errors += 1
                print("x", end="", flush=True)
            time.sleep(args.delay)
        print()

        if not endpoint_times:
            print(f"  FAIL: all {args.samples} requests errored")
            overall_pass = False
            continue

        # Raw SLURM floor (one sample) — how much is dashboard overhead vs SLURM itself
        t0 = time.perf_counter()
        _, _, _ = run_shell('sinfo -N -o "%N|%G|%C" --noheader', ssh, timeout=60)
        slurm_floor = time.perf_counter() - t0

        stats = {
            "min":  min(endpoint_times),
            "p50":  percentile(endpoint_times, 50),
            "p95":  percentile(endpoint_times, 95),
            "max":  max(endpoint_times),
            "mean": statistics.mean(endpoint_times),
        }
        print(f"  samples ok={len(endpoint_times)}/{args.samples} errors={errors}")
        print(f"  min={stats['min']:.2f}s  p50={stats['p50']:.2f}s  p95={stats['p95']:.2f}s  max={stats['max']:.2f}s  mean={stats['mean']:.2f}s")
        print(f"  raw sinfo floor (1 sample): {slurm_floor:.2f}s — dashboard overhead ≈ {stats['p50'] - slurm_floor:+.2f}s")

        cluster_pass = stats["p95"] < args.target and errors == 0
        print(f"  RESULT: {'PASS' if cluster_pass else 'FAIL'} (p95 {'<' if cluster_pass else '>='} {args.target}s)")
        if not cluster_pass:
            overall_pass = False
        summary.append((name, stats, errors, slurm_floor))

    print()
    print(f"OVERALL LATENCY: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


# ---------------------------------------------------------------------------
# 3. Reliability
# ---------------------------------------------------------------------------
def check_renewal_chain():
    """Return (status, info_dict). Healthy state = 1 RUNNING + 1 PENDING(Dependency)."""
    out, err, rc = run_shell(
        f'squeue -u $USER -n lemon -h -o "%i|%T|%R"',
        ssh_prefix=None,
        timeout=15,
    )
    if rc != 0:
        return "error", {"err": err.strip() or f"rc={rc}", "jobs": []}
    jobs = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            jobs.append({"id": parts[0].strip(), "state": parts[1].strip(), "reason": parts[2].strip()})

    running = [j for j in jobs if j["state"] == "RUNNING"]
    pending = [j for j in jobs if j["state"] == "PENDING"]

    if len(running) == 1 and len(pending) == 1 and "Dependency" in pending[0]["reason"]:
        status = "healthy"
    elif len(running) == 1 and len(pending) == 0:
        status = "no_pending"   # chain is broken — next job not queued
    elif len(running) == 0:
        status = "no_running"   # dashboard is DOWN
    elif len(running) > 1 or len(pending) > 1:
        status = "extra_jobs"   # something is weird
    else:
        status = "unhealthy"
    return status, {"jobs": jobs, "running": len(running), "pending": len(pending)}


def check_tunnel(public_url, timeout=10):
    """HEAD-request the tunnel URL. Returns (ok, latency_or_err)."""
    if not public_url:
        return False, "no public_url in state.json"
    try:
        t0 = time.perf_counter()
        # Cloudflare quick tunnels return 200 on root for our Flask app
        r = requests.get(public_url, timeout=timeout, allow_redirects=True)
        dt = time.perf_counter() - t0
        return (r.status_code < 500), f"{r.status_code} in {dt:.2f}s"
    except Exception as e:
        return False, str(e)[:80]


def active_tunnel_test():
    """Kill cloudflared and measure time-to-recover. Requires monitor.py's
    run_cloudflare_tunnel_forever() loop to be running (it is, see monitor.py:2565)."""
    print("\n  ACTIVE TEST: killing cloudflared and timing recovery...")
    out, _, _ = run_shell("pgrep -f 'cloudflared tunnel'", timeout=5)
    pids = [p for p in out.split() if p.isdigit()]
    if not pids:
        return False, "no cloudflared process found"
    print(f"  killing pids: {pids}")
    for pid in pids:
        run_shell(f"kill {pid}", timeout=5)

    # Wait for the process to come back AND respond
    _, state = load_dashboard_url()
    public_url = state.get("public_url")
    deadline = time.time() + 90
    t0 = time.time()
    last_err = None
    while time.time() < deadline:
        time.sleep(2)
        # Re-read state.json — public_url may change after a quick-tunnel restart
        try:
            state = json.loads(STATE_FILE.read_text())
            public_url = state.get("public_url") or public_url
        except Exception:
            pass
        ok, info = check_tunnel(public_url, timeout=5)
        if ok:
            return True, f"recovered in {time.time() - t0:.1f}s ({info})"
        last_err = info
    return False, f"did not recover in 90s (last: {last_err})"


def cmd_reliability(args):
    section(f"RELIABILITY — sampling for {args.duration} min every {args.interval}s")
    base_url, state = load_dashboard_url()
    public_url = state.get("public_url")
    print(f"Dashboard:  {base_url}")
    print(f"Public URL: {public_url or '(none — tunnel test will be skipped)'}")
    print(f"NOTE: a {args.duration}-minute window only catches OBVIOUS breakage. For real")
    print(f"      uptime metrics, run this for hours or days under tmux/cron.")

    end_time = time.time() + args.duration * 60
    chain_samples = []   # list of status strings
    tunnel_samples = []  # list of bools
    tunnel_failures = 0
    longest_failure = 0
    current_failure = 0
    sample_idx = 0

    print(f"\n  {'#':>3} {'time':<10} {'chain_status':<14} {'r/p':<6} {'tunnel':<10} {'detail':<30}")
    print(f"  {'-' * 78}")

    while time.time() < end_time:
        sample_idx += 1
        ts = datetime.now().strftime("%H:%M:%S")
        chain_status, chain_info = check_renewal_chain()
        chain_samples.append(chain_status)
        rp = f"{chain_info.get('running', 0)}/{chain_info.get('pending', 0)}"

        if public_url:
            ok, tunnel_info = check_tunnel(public_url, timeout=8)
            tunnel_samples.append(ok)
            if ok:
                current_failure = 0
            else:
                tunnel_failures += 1
                current_failure += 1
                if current_failure > longest_failure:
                    longest_failure = current_failure
            tunnel_str = "ok" if ok else "FAIL"
            detail = tunnel_info[:30]
        else:
            tunnel_str = "skip"
            detail = ""

        print(f"  {sample_idx:>3} {ts:<10} {chain_status:<14} {rp:<6} {tunnel_str:<10} {detail:<30}")

        if time.time() < end_time:
            time.sleep(args.interval)

    # Aggregate
    n = len(chain_samples)
    healthy_count = sum(1 for s in chain_samples if s == "healthy")
    chain_uptime = (healthy_count / n * 100) if n else 0

    print()
    print("--- Auto-renewal chain ---")
    print(f"  samples:           {n}")
    print(f"  healthy (R+P+dep): {healthy_count} ({chain_uptime:.1f}%)")
    breakdown = {}
    for s in chain_samples:
        breakdown[s] = breakdown.get(s, 0) + 1
    for status, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        if status != "healthy":
            print(f"  {status:<18} {count}")
    chain_pass = chain_uptime >= 95.0

    print()
    print("--- Tunnel availability (passive) ---")
    if not public_url:
        print("  skipped (no public_url)")
        tunnel_pass = True
    else:
        tn = len(tunnel_samples)
        ok_count = sum(1 for s in tunnel_samples if s)
        tunnel_uptime = (ok_count / tn * 100) if tn else 0
        print(f"  samples:           {tn}")
        print(f"  uptime:            {tunnel_uptime:.1f}%")
        print(f"  failures:          {tunnel_failures}")
        print(f"  longest outage:    {longest_failure} consecutive samples ({longest_failure * args.interval}s)")
        tunnel_pass = tunnel_uptime >= 95.0

    # Active tunnel kill test
    if args.active_tunnel_test:
        if not public_url:
            print("\n  ACTIVE TEST: skipped (no public_url)")
            active_pass = True
        else:
            ok, info = active_tunnel_test()
            print(f"  active recovery: {'PASS' if ok else 'FAIL'} — {info}")
            active_pass = ok
        tunnel_pass = tunnel_pass and active_pass

    overall_pass = chain_pass and tunnel_pass
    print()
    print(f"OVERALL RELIABILITY: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPU Cluster Monitor evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # correctness
    p_c = sub.add_parser("correctness", help="Validate stats vs. direct SLURM queries")
    p_c.add_argument("--clusters", nargs="+", default=["cs", "hpc"], choices=["cs", "hpc"])
    p_c.add_argument("--gpu-tol", type=int, default=2, help="Allowed GPU count drift (default 2)")
    p_c.add_argument("--cpu-tol", type=int, default=4, help="Allowed CPU count drift (default 4)")
    p_c.set_defaults(func=cmd_correctness)

    # latency
    p_l = sub.add_parser("latency", help="Measure dashboard response time")
    p_l.add_argument("--clusters", nargs="+", default=["cs", "hpc"], choices=["cs", "hpc"])
    p_l.add_argument("--samples", type=int, default=10, help="Requests per cluster (default 10)")
    p_l.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default 1.0)")
    p_l.add_argument("--target", type=float, default=5.0, help="p95 target in seconds (default 5)")
    p_l.set_defaults(func=cmd_latency)

    # reliability
    p_r = sub.add_parser("reliability", help="Sample renewal chain + tunnel uptime")
    p_r.add_argument("--duration", type=int, default=5, help="Sampling window in minutes (default 5)")
    p_r.add_argument("--interval", type=int, default=15, help="Seconds between samples (default 15)")
    p_r.add_argument("--active-tunnel-test", action="store_true",
                     help="DESTRUCTIVE: kill cloudflared and time the auto-recovery (~10-30s gap)")
    p_r.set_defaults(func=cmd_reliability)

    # all
    p_all = sub.add_parser("all", help="Run all three (with default args)")
    p_all.set_defaults(func=lambda a: cmd_all(a))

    args = parser.parse_args()
    return args.func(args)


def cmd_all(args):
    """Run all three with defaults; aggregate exit code."""
    class _A:
        clusters = ["cs", "hpc"]
        gpu_tol = 2
        cpu_tol = 4
        samples = 10
        delay = 1.0
        target = 5.0
        duration = 5
        interval = 15
        active_tunnel_test = False
    a = _A()
    rc = 0
    rc |= cmd_correctness(a)
    rc |= cmd_latency(a)
    rc |= cmd_reliability(a)
    print()
    print("=" * 70)
    print(f"  ALL TESTS: {'PASS' if rc == 0 else 'FAIL'}")
    print("=" * 70)
    return rc


if __name__ == "__main__":
    sys.exit(main())
