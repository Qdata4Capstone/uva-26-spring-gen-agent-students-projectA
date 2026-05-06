#!/bin/bash
# =========================================================================
# Start the GPU Cluster Monitor
# Submits a SLURM CPU job that runs the web dashboard.
# =========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/state.json"
PORT="${1:-8080}"

# Load secrets from .env if present (e.g. OPENAI_API_KEY=...)
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    . "${SCRIPT_DIR}/.env"
    set +a
fi

# Check if already running
if [ -f "${STATE_FILE}" ]; then
    EXISTING_JOB=$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['job_id'])" 2>/dev/null)
    if [ -n "${EXISTING_JOB}" ]; then
        JOB_STATE=$(squeue -j "${EXISTING_JOB}" -h -o "%T" 2>/dev/null)
        if [ -n "${JOB_STATE}" ]; then
            echo "Monitor is already running!"
            echo "  Job ID: ${EXISTING_JOB}"
            echo "  State:  ${JOB_STATE}"
            echo "  URL:    $(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['url'])" 2>/dev/null)"
            echo ""
            echo "To stop it:  bash ${SCRIPT_DIR}/stop.sh"
            exit 0
        fi
    fi
    rm -f "${STATE_FILE}"
fi

mkdir -p "${SCRIPT_DIR}/logs"

echo "Submitting GPU Monitor job (port: ${PORT})..."
JOB_ID=$(sbatch --export=ALL,MONITOR_PORT=${PORT},OPENAI_API_KEY="${OPENAI_API_KEY}" "${SCRIPT_DIR}/job.sh" 2>&1 | grep -oP '\d+')

if [ -n "${JOB_ID}" ]; then
    echo ""
    echo "Job submitted: ${JOB_ID}"
    echo "Waiting for job to start..."

    # Wait for the job to start (up to 120 seconds)
    for i in $(seq 1 60); do
        sleep 2
        STATE=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null)
        if [ "${STATE}" = "RUNNING" ]; then
            sleep 5  # Give Flask a moment to start
            if [ -f "${STATE_FILE}" ]; then
                NODE=$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['node'])" 2>/dev/null)
                echo ""
                echo "============================================"
                echo "  GPU Monitor is RUNNING!"
                echo "  Local:  http://${NODE}:${PORT}"
                # Wait for Cloudflare tunnel to establish
                echo "  Waiting for public URL..."
                for j in $(seq 1 15); do
                    sleep 2
                    PUBLIC=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('public_url',''))" 2>/dev/null)
                    if [ -n "${PUBLIC}" ]; then
                        echo "  Public: ${PUBLIC}"
                        break
                    fi
                    printf "."
                done
                if [ -z "${PUBLIC}" ]; then
                    echo ""
                    echo "  Public URL not available yet. Check logs or state.json later."
                fi
                echo ""
                echo "  Stop: bash ${SCRIPT_DIR}/stop.sh"
                echo "============================================"
                exit 0
            fi
        elif [ -z "${STATE}" ]; then
            echo "Job ${JOB_ID} is no longer in queue. Check logs:"
            echo "  ${SCRIPT_DIR}/logs/monitor_${JOB_ID}.log"
            exit 1
        fi
        printf "."
    done
    echo ""
    echo "Job is still pending. It will start when resources are available."
    echo "Check status: squeue -j ${JOB_ID}"
else
    echo "ERROR: Failed to submit job!"
    exit 1
fi
