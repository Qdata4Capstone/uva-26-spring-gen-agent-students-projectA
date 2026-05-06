# IMPROVE.md — team-00-ProjB (GPU Cluster Monitor)

## P1 — Critical / Security

- **Hardcoded SSH username** (`evaluation.py:54`): `tsx4zn@login.hpc.virginia.edu` is hardcoded. Any other user gets silent SSH failures. Load from `config.json` like `monitor.py` already does.
- **Job name hardcoded as `lemon`** (`start.sh:16`, `stop.sh:16`, `evaluation.py:343`, `job.sh:2`): Changing `slurm_job_name` in `config.json` has no effect on the scripts. Pass job name as a parameter or read from config.
- **Missing OpenAI API key validation** (`monitor.py:590-598`): `get_openai()` creates a client even if the key is empty or absent, failing only at the first API call with a cryptic 401. Validate key is non-empty on startup.
- **XSS via innerHTML** (`monitor.py:2600`): SLURM node names are injected into HTML with template literals and no escaping. Use `document.createElement()` or an `escapeHtml()` helper.
- **Hardcoded port in stop.sh** (`stop.sh:35`): Always kills port 8080 regardless of configured port. Read port from `state.json` or `config.json`.

## P2 — Reliability

- **Silent config failure** (`monitor.py:43-55`): Missing `config.json` returns `{}` with no error, causing downstream failures with cryptic messages. Validate required fields on startup and fail fast.
- **No retry on SLURM command failures** (`monitor.py:89-99`): Transient SSH hiccups make `get_cluster_data()` return empty results silently, feeding bad data to the UI. Add 1–2 retries on timeout/SSH errors.
- **state.json not atomic** (`job.sh:36`, `monitor.py:3052`): Two separate writers (job.sh on startup, monitor.py on Cloudflare URL arrival) can produce partial/corrupt JSON. Use write-to-temp-then-mv.
- **Cloudflare tunnel spins forever** (`monitor.py:3070-3095`): If `cloudflared` binary is missing, the retry loop runs indefinitely with 10 s delays and no alerting. Add a max-retry cap and write an error flag to `state.json`.
- **No timeout configurability** (`monitor.py:89`): SLURM `scontrol` calls have a fixed 60 s timeout. Make it configurable; consider 30 s for remote HPC queries.

## P3 — Quality

- **Cluster config duplicated** (`evaluation.py:52-55`): Cluster SSH definitions are copied from `monitor.py` instead of reading `config.json`. One change point breaks the other silently.
- **GPU name mapping brittle** (`monitor.py:159-201`): 26 hardcoded GPU name strings will break when new GPU models appear. Move to a data file or `config.json`.
- **Silent tool argument errors** (`monitor.py:2028`): Malformed JSON from OpenAI tool calls is silently replaced with `{}`, dispatching the wrong tool parameters. Log and surface the error to the user.
- **No rate limiting on `/api/chat`** (`monitor.py:1950`): No per-session throttle; a script can spam requests and run up OpenAI costs. Add a simple 10 req/min limit.
- **Graceful shutdown missing** (`monitor.py:3112`): Daemon threads (cloudflare tunnel, snapshot loop) are not cleaned up on SIGTERM. Add `atexit` / signal handlers.

## P4 — Enhancements

- **`web_search` dependency undocumented**: README promises web search but doesn't warn that `ddgs` must be installed separately. Add to `environment.yml` and surface a warning in the chat UI if unavailable.
- **No cluster health alerting**: UI shows "unavailable" silently. A background thread + `/api/health` endpoint would let users or monitoring systems detect outages.
- **Agent action metrics missing**: `agent_actions.log` is append-only. A `/api/agent/stats` endpoint tracking approve/reject/expire rates would show whether the agent is useful.
- **SSH passphrase keys unsupported**: README requires passwordless SSH but gives no setup instructions. Add a troubleshooting section or support `SSH_AUTH_SOCK`.
