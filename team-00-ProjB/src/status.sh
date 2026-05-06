#!/bin/bash
# =========================================================================
# Check GPU Monitor status
# =========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/state.json"

echo "=== GPU Monitor Status ==="
echo ""

# Check for running jobs
JOBS=$(squeue -u "$USER" -n lemon -h -o "%.10i %.8T %.10M %N" 2>/dev/null)

if [ -n "${JOBS}" ]; then
    echo "Running jobs:"
    echo "  JOB_ID     STATE    TIME       NODE"
    echo "${JOBS}" | while read line; do echo "  ${line}"; done
    echo ""
    if [ -f "${STATE_FILE}" ]; then
        URL=$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['url'])" 2>/dev/null)
        PUBLIC=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('public_url',''))" 2>/dev/null)
        echo "Local:  ${URL}"
        if [ -n "${PUBLIC}" ]; then
            echo "Public: ${PUBLIC}"
        fi
    fi
else
    echo "No monitor jobs running."
fi
echo ""
