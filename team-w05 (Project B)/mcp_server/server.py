"""
Mental_Health_Bot MCP Server
=====================
A standalone MCP server that exposes mental-health research tools.
Communicates over stdio (Model Context Protocol standard transport).

Run directly:
    python mcp_server/server.py

Or spawned automatically by the FastAPI backend as a subprocess.

Tools exposed
-------------
  pubmed_search      — search PubMed for peer-reviewed research articles
  crisis_resources   — return localised crisis hotline information
"""

import json
import logging
import sys

# Silence noisy loggers before anything else (MCP server writes to stdout)
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from mcp.server.fastmcp import FastMCP
from pubmed import fetch_articles  # sibling module

mcp = FastMCP(
    name="Mental_Health_Bot Tools",
    instructions=(
        "You are a mental health support assistant. "
        "Use pubmed_search when the user asks about treatments, therapies, "
        "medications, or the scientific evidence behind mental health conditions. "
        "Use crisis_resources if the user seems to be in distress or asks for help lines. "
        "Always present research findings in plain language and remind users that "
        "results are general information, not personalised medical advice."
    ),
)


# ── Tool 1: PubMed research search ────────────────────────────────────────

@mcp.tool()
async def pubmed_search(query: str, max_results: int = 5) -> str:
    """
    Search PubMed for peer-reviewed mental health research articles.

    Use this tool when the user asks about:
    - Treatments, therapies (CBT, DBT, mindfulness, EMDR, etc.)
    - Medications or clinical interventions
    - Scientific evidence for a mental health condition
    - Effectiveness of a particular approach

    Args:
        query:       A focused 2-10 keyword PubMed search query.
        max_results: Maximum articles to return (1-10, default 5).

    Returns:
        JSON string containing a list of article objects with fields:
        pmid, title, abstract, authors, journal, pub_date, url.
        Returns an empty list if no results found.
    """
    if not query.strip():
        return json.dumps([])

    max_results = max(1, min(10, max_results))
    articles = await fetch_articles(query, max_results)
    return json.dumps(articles, ensure_ascii=False)


# ── Tool 2: Crisis resources ──────────────────────────────────────────────

CRISIS_DATA = {
    "global": [
        {"name": "International Association for Suicide Prevention",
         "url": "https://www.iasp.info/resources/Crisis_Centres/",
         "note": "Directory of crisis centres worldwide"},
    ],
    "US": [
        {"name": "988 Suicide & Crisis Lifeline", "contact": "call or text 988",
         "url": "https://988lifeline.org", "hours": "24/7"},
        {"name": "Crisis Text Line", "contact": "text HOME to 741741",
         "url": "https://www.crisistextline.org", "hours": "24/7"},
        {"name": "SAMHSA National Helpline", "contact": "1-800-662-4357",
         "url": "https://www.samhsa.gov/find-help/national-helpline", "hours": "24/7"},
    ],
    "UK": [
        {"name": "Samaritans", "contact": "116 123",
         "url": "https://www.samaritans.org", "hours": "24/7"},
        {"name": "CALM", "contact": "0800 58 58 58",
         "url": "https://www.thecalmzone.net", "hours": "5pm–midnight"},
        {"name": "Crisis Text Line UK", "contact": "text SHOUT to 85258",
         "url": "https://giveusashout.org", "hours": "24/7"},
    ],
    "CA": [
        {"name": "Canada Suicide Prevention Service",
         "contact": "1-833-456-4566",
         "url": "https://www.crisisservicescanada.ca", "hours": "24/7"},
    ],
    "AU": [
        {"name": "Lifeline Australia", "contact": "13 11 14",
         "url": "https://www.lifeline.org.au", "hours": "24/7"},
        {"name": "Beyond Blue", "contact": "1300 22 4636",
         "url": "https://www.beyondblue.org.au", "hours": "24/7"},
    ],
}


@mcp.tool()
def crisis_resources(country_code: str = "US") -> str:
    """
    Return crisis support hotlines and resources for a given country.

    Use this tool when:
    - The user seems distressed, overwhelmed, or mentions self-harm
    - The user asks for mental health helplines or crisis support
    - You want to remind the user that real human support is available

    Args:
        country_code: ISO 3166-1 alpha-2 country code (US, UK, CA, AU).
                      Defaults to US. Always also returns global resources.

    Returns:
        JSON string with localised crisis resources.
    """
    code = country_code.upper().strip()
    resources = {
        "country": code,
        "local": CRISIS_DATA.get(code, CRISIS_DATA["US"]),
        "global": CRISIS_DATA["global"],
    }
    return json.dumps(resources, ensure_ascii=False)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run on stdio transport (default for MCP)
    mcp.run(transport="stdio")
