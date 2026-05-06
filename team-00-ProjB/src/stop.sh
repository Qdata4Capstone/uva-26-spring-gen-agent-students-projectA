#!/bin/bash
# =========================================================================
# Stop the GPU Cluster Monitor
# Cancels the current job and prevents auto-renewal.
# =========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/state.json"

echo "Stopping GPU Monitor..."

# Signal that this is an intentional stop (so the job won't let the chain continue)
touch "${SCRIPT_DIR}/.stop_requested"

# Cancel all gpu-monitor jobs belonging to this user
JOBS=$(squeue -u "$USER" -n lemon -h -o "%i" 2>/dev/null)

if [ -n "${JOBS}" ]; then
    for JOB in ${JOBS}; do
        echo "  Cancelling job ${JOB}..."
        scancel "${JOB}" 2>/dev/null
    done
    echo "All monitor jobs cancelled."
else
    echo "No running monitor jobs found."
fi

# Also cancel any pending next job
if [ -f "${SCRIPT_DIR}/.next_job_id" ]; then
    NEXT=$(cat "${SCRIPT_DIR}/.next_job_id")
    scancel "${NEXT}" 2>/dev/null
    rm -f "${SCRIPT_DIR}/.next_job_id"
fi

# Force kill any process still holding port 8080
PORT_PID=$(lsof -ti:8080 2>/dev/null)
if [ -n "${PORT_PID}" ]; then
    echo "Force killing process on port 8080 (PID: ${PORT_PID})..."
    kill -9 ${PORT_PID} 2>/dev/null
fi

# Cleanup state
rm -f "${STATE_FILE}"
rm -f "${SCRIPT_DIR}/.stop_requested"

echo "Done."
