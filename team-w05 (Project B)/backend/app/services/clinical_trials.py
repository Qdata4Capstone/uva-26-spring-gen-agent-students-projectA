from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CLINICAL_TRIALS_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_TIMEOUT = 15.0
_MAX_DESC_LEN = 600


def _strip_markdown_noise(text: str) -> str:
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    t = re.sub(r"[#*_`]+", "", t)
    return " ".join(t.split())


def _study_to_record(study: dict[str, Any]) -> dict[str, Any] | None:
    proto = study.get("protocolSection") or {}
    ident = proto.get("identificationModule") or {}
    status_mod = proto.get("statusModule") or {}
    desc = proto.get("descriptionModule") or {}

    nct_id = ident.get("nctId")
    if not nct_id:
        return None

    title = ident.get("briefTitle") or ident.get("officialTitle") or "Untitled study"
    raw_summary = desc.get("briefSummary") or ""
    brief = _strip_markdown_noise(raw_summary)
    if len(brief) > _MAX_DESC_LEN:
        brief = brief[: _MAX_DESC_LEN - 1] + "..."

    overall = (status_mod.get("overallStatus") or "UNKNOWN").strip()
    recruiting = overall.upper() == "RECRUITING"

    return {
        "nct_id": nct_id,
        "title": title,
        "brief_description": brief,
        "status": "Recruiting" if recruiting else "Not recruiting",
        "recruiting": recruiting,
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


async def search_clinical_trials(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    n = max(1, min(10, max_results))
    params = {"format": "json", "query.term": q, "pageSize": n}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(CLINICAL_TRIALS_STUDIES_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("ClinicalTrials.gov HTTP error: %s", exc)
        return []
    except ValueError as exc:
        logger.warning("ClinicalTrials.gov invalid JSON: %s", exc)
        return []
    except Exception as exc:
        logger.warning("ClinicalTrials.gov request failed: %s", exc)
        return []

    studies = payload.get("studies")
    if not isinstance(studies, list):
        return []

    out: list[dict[str, Any]] = []
    for item in studies:
        if not isinstance(item, dict):
            continue
        rec = _study_to_record(item)
        if rec:
            out.append(rec)
        if len(out) >= n:
            break
    return out
