#!/bin/bash
#SBATCH --job-name=lemon

#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=07:00:00
#SBATCH --output=logs/monitor_%j.log
#SBATCH --error=logs/monitor_%j.err

# =========================================================================
# GPU Cluster Monitor - SLURM Job Script
# Runs a Flask web dashboard and auto-submits the next job before expiring.
# Submit from the project directory: sbatch job.sh
# =========================================================================

# Allow env-var overrides so other users don't need to edit this file.
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_ENV="${CONDA_ENV:-$HOME/miniconda3/envs/gpu_monitor}"
PORT="${MONITOR_PORT:-8080}"
STATE_FILE="${SCRIPT_DIR}/state.json"

# Create log directory
mkdir -p "${SCRIPT_DIR}/logs"

echo "============================================"
echo "GPU Monitor Job Started"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Port:      ${PORT}"
echo "Time:      $(date)"
echo "============================================"

# ----- Write state file so scripts can find us -----
cat > "${STATE_FILE}" <<SEOF
{
  "job_id": "${SLURM_JOB_ID}",
  "node": "$(hostname)",
  "port": ${PORT},
  "started": "$(date -Iseconds)",
  "url": "http://$(hostname):${PORT}"
}
SEOF

echo ""
echo ">>> Dashboard URL: http://$(hostname):${PORT}"
echo ">>> If not directly accessible, use SSH tunnel:"
echo ">>>   ssh -L ${PORT}:$(hostname):${PORT} <your-cluster-login>"
echo ">>>   Then open: http://localhost:${PORT}"
echo ""

# ----- Submit the NEXT job (dependency chain for auto-renewal) -----
NEXT_JOB_ID=$(sbatch --dependency=afterany:${SLURM_JOB_ID} \
    --export=ALL,MONITOR_PORT=${PORT} \
    "${SCRIPT_DIR}/job.sh" 2>&1 | grep -oP '\d+')

if [ -n "${NEXT_JOB_ID}" ]; then
    echo "Next job submitted: ${NEXT_JOB_ID} (will start after this job ends)"
    echo "${NEXT_JOB_ID}" > "${SCRIPT_DIR}/.next_job_id"
else
    echo "WARNING: Failed to submit next job!"
fi

# ----- Activate conda and start the monitor -----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"


export MONITOR_PORT="${PORT}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

# Trap signals for clean shutdown
cleanup() {
    echo "Received shutdown signal, cleaning up..."
    # Cancel the next pending job if this job was intentionally stopped
    if [ -f "${SCRIPT_DIR}/.stop_requested" ]; then
        if [ -n "${NEXT_JOB_ID}" ]; then
            echo "Stop requested: cancelling next job ${NEXT_JOB_ID}"
            scancel "${NEXT_JOB_ID}" 2>/dev/null
        fi
        rm -f "${SCRIPT_DIR}/.stop_requested"
    fi
    rm -f "${STATE_FILE}"
    exit 0
}
trap cleanup SIGTERM SIGINT EXIT

# Run the Flask app
python "${SCRIPT_DIR}/monitor.py"
