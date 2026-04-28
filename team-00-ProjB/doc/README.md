# GPU Cluster Monitor

A real-time GPU cluster monitoring dashboard that runs as a SLURM job. Tracks GPU/memory usage across multiple SLURM clusters with a web UI.

![Dashboard](https://img.shields.io/badge/SLURM-Dashboard-blue)

## Features

- Real-time GPU and memory usage per node
- Multi-cluster support (monitor local + remote clusters via SSH)
- Job queue analysis powered by OpenAI LLM
- Chat agent can search the web for GPU/library compatibility, VRAM requirements, CUDA versions, etc.
- Chat agent can scan a project directory and auto-recommend a GPU based on the codebase (`analyze_workspace` tool)
- Auto-renewing SLURM job (submits next job before expiring)
- Public URL via Cloudflare Tunnel (auto-reconnects on failure)

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
# or manually:
conda create -n gpu_monitor python=3.10 -y
conda activate gpu_monitor
pip install flask openai ddgs requests
```

### 2. Configure

Copy the example config and fill in your values:

```bash
cp config.json.example config.json
```

Edit `config.json`:

```json
{
  "script_dir": "/home/YOUR_USER/gpu_monitor",
  "conda_env": "/home/YOUR_USER/miniconda3/envs/gpu_monitor",
  "port": 8080,
  "slurm_job_name": "gpu_monitor",
  "slurm_partition": "standard",
  "slurm_time": "07:00:00",
  "openai_model": "gpt-4o",
  "clusters": {
    "cs": {
      "name": "CS",
      "ssh": null
    },
    "hpc": {
      "name": "Rivanna",
      "ssh": "ssh -o ConnectTimeout=10 -o BatchMode=yes YOUR_USER@login.hpc.virginia.edu"
    }
  }
}
```

- Set `ssh` to `null` for the local cluster (where the job runs).
- For remote clusters, set `ssh` to the full SSH command using your username. Passwordless SSH (key-based auth) must be configured.
- `account` (per cluster) is the SLURM account charged for submitted jobs. Leave it `null` on clusters that don't require an account (e.g. UVA's CS cluster).
- Remove the `hpc` block entirely if you only have one cluster.

### 3. Set OpenAI API key

Create a `.env` file in the project directory:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

Or export it as an environment variable before submitting the job:

```bash
export OPENAI_API_KEY="sk-..."
```

Get a key at [platform.openai.com](https://platform.openai.com). The key is only required for the chat/job-analysis features; the dashboard works without it.


## Usage

```bash
# Start the monitor (submits a SLURM job)

bash start.sh or python monitor.py

# Check status and current URL
bash status.sh

# Stop the monitor
bash stop.sh
```

The dashboard will be available at:
- **Local**: `http://<node>:8080` (within the cluster network)
- **SSH tunnel**: `ssh -L 8080:<node>:8080 <cluster-login>` then open `http://localhost:8080`
- **Public**: a `*.trycloudflare.com` URL (printed on startup, also in `~/public_url`)

> **Note**: The public URL changes when the Cloudflare tunnel reconnects. Check `cat ~/public_url` or run `bash status.sh` for the current URL.

### Overriding paths without editing files

You can pass `SCRIPT_DIR` and `CONDA_ENV` as environment variables when submitting, which is useful if you have the project in a non-default location:

```bash
SCRIPT_DIR=/custom/path CONDA_ENV=/custom/envs/gpu_monitor sbatch job.sh
```

## Project Structure

```
├── monitor.py          # Flask app + SLURM data collection + Cloudflare tunnel
├── job.sh              # SLURM job script (auto-renewing)
├── start.sh            # Submit the monitor job
├── stop.sh             # Cancel all monitor jobs
├── status.sh           # Check monitor status and URL
├── config.json         # Your local configuration (not committed)
├── config.json.example # Template 
└── environment.yml     # Conda environment definition
```

## Notes on conda environments

Each user needs their own conda environment unless a shared environment path is provided by the cluster admin. The `conda_env` field in `config.json` must point to the **absolute path** of an activated environment that has `flask`, `openai`, `ddgs`, and `requests` installed.

To find the path of an existing environment:

```bash
conda activate gpu_monitor
which python   # shows /path/to/envs/gpu_monitor/bin/python
# use the parent of bin/ as conda_env
```
