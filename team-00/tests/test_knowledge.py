"""
Tests for src/agent/tools/knowledge.py — KnowledgeBase query logic.

Run from team-00/:
    pytest tests/test_knowledge.py -v
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.agent.tools.knowledge import KnowledgeBase


# ── Fixture: minimal in-memory knowledge DB ──────────────────────────────────

SAMPLE_DB = {
    "platforms": {
        "youtube": {
            "name": "YouTube",
            "requirements": ["no faces of minors", "blur license plates"],
            "regulation": "GDPR",
        },
        "tiktok": {
            "name": "TikTok",
            "requirements": ["face blurring required"],
            "regulation": "CCPA",
        },
    },
    "conferences": {
        "cvpr2026": {
            "name": "CVPR 2026",
            "requirements": ["de-identified dataset", "no PII in metadata"],
            "deadline": "2026-03-15",
        },
        "neurips2025": {
            "name": "NeurIPS 2025",
            "requirements": ["open-science policy", "IRB approval for human data"],
        },
    },
    "regulations": {
        "gdpr": {
            "name": "GDPR",
            "jurisdiction": "EU",
            "requirements": ["right to erasure", "data minimisation"],
        },
        "ccpa": {
            "name": "CCPA",
            "jurisdiction": "California",
            "requirements": ["opt-out rights", "no sale of personal data"],
        },
    },
}


@pytest.fixture
def kb(tmp_path):
    """Write sample DB to a temp file and return a KnowledgeBase instance."""
    db_file = tmp_path / "regulations.json"
    db_file.write_text(json.dumps(SAMPLE_DB))
    return KnowledgeBase(knowledge_path=str(db_file))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestKnowledgeBaseQuery:

    def test_query_platform_by_name(self, kb):
        results = kb.query("youtube")
        assert any("platform:youtube" in k for k in results), \
            "Should match 'youtube' platform entry"

    def test_query_conference_by_name(self, kb):
        results = kb.query("cvpr2026")
        assert any("conference:cvpr2026" in k or "cvpr2026" in str(results).lower() for k in results), \
            "Should match CVPR 2026 conference entry"

    def test_query_case_insensitive(self, kb):
        results_lower = kb.query("youtube")
        results_upper = kb.query("YouTube")
        assert results_lower == results_upper, \
            "Query should be case-insensitive"

    def test_query_partial_match(self, kb):
        results = kb.query("I want to submit my dataset to CVPR 2026 submission")
        assert results, "Should match conference from natural language query"

    def test_query_no_match_returns_empty(self, kb):
        results = kb.query("xkcd_unknown_conference_xyz")
        assert results == {}, "Unknown query should return empty dict"

    def test_query_multiple_matches(self, kb):
        # "gdpr youtube" should hit both platform and regulation
        results = kb.query("youtube gdpr compliance")
        assert len(results) >= 1, "Should return at least one match"

    def test_normalize_strips_punctuation(self, kb):
        results = kb.query("tik-tok")
        # Normalization strips non-alphanumeric → "tiktok"
        assert any("tiktok" in k for k in results), \
            "Hyphenated variant should normalize and match tiktok"


class TestKnowledgeBaseInit:

    def test_init_with_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            KnowledgeBase(knowledge_path="/nonexistent/path/regulations.json")

    def test_init_loads_all_sections(self, kb):
        # Verify the internal DB has all three top-level sections
        assert "platforms" in kb._db
        assert "conferences" in kb._db
        assert "regulations" in kb._db
