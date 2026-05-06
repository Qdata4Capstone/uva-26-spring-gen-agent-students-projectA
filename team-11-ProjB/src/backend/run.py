#!/usr/bin/env python3
"""Entry point for the FinSynth backend server."""
import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("FINSYNTH_RELOAD", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
    )
