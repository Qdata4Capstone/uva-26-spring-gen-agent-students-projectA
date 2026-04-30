"""
PubMed API helper — used exclusively by the MCP server.
Fetches and parses articles from NCBI E-utilities.
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_URL  = "https://pubmed.ncbi.nlm.nih.gov"


def _base_params() -> dict:
    params: dict[str, Any] = {"db": "pubmed"}
    if key := os.getenv("NCBI_API_KEY", ""):
        params["api_key"] = key
    return params


def _clean(node: ET.Element) -> str:
    return "".join(node.itertext()).strip()


def _parse_articles(xml_text: str) -> list[dict]:
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("XML parse error: %s", exc)
        return articles

    for node in root.findall(".//PubmedArticle"):
        try:
            pmid  = (node.findtext(".//PMID") or "").strip()
            title = _clean(node.find(".//ArticleTitle")) if node.find(".//ArticleTitle") is not None else "No title"

            abstract_node = node.find(".//AbstractText")
            abstract = _clean(abstract_node) if abstract_node is not None else None

            authors = []
            for a in node.findall(".//Author"):
                last = a.findtext("LastName", "")
                fore = a.findtext("ForeName", "")
                name = f"{last}, {fore}".strip(", ")
                if name:
                    authors.append(name)

            journal  = node.findtext(".//Journal/Title")
            pub_date = node.findtext(".//PubDate/Year")

            articles.append({
                "pmid":     pmid,
                "title":    title,
                "abstract": abstract,
                "authors":  authors[:5],
                "journal":  journal,
                "pub_date": pub_date,
                "url":      f"{PUBMED_URL}/{pmid}/",
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed article: %s", exc)

    return articles


def _rank(articles: list[dict]) -> list[dict]:
    return sorted(
        articles,
        key=lambda a: (a["abstract"] is not None, a.get("pub_date") or ""),
        reverse=True,
    )


async def fetch_articles(query: str, max_results: int = 5) -> list[dict]:
    """Search PubMed and return ranked article dicts."""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # esearch — get PMIDs
            search_params = {**_base_params(), "term": query,
                             "retmax": max_results * 2, "sort": "relevance",
                             "retmode": "json"}
            r = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=search_params)
            r.raise_for_status()
            pmids = r.json().get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []

            # efetch — get full records
            fetch_params = {**_base_params(), "id": ",".join(pmids),
                            "rettype": "abstract", "retmode": "xml"}
            r2 = await client.get(f"{PUBMED_BASE}/efetch.fcgi", params=fetch_params)
            r2.raise_for_status()
            articles = _parse_articles(r2.text)

        return _rank(articles)[:max_results]

    except Exception as exc:  # noqa: BLE001
        logger.exception("PubMed fetch failed: %s", exc)
        return []
