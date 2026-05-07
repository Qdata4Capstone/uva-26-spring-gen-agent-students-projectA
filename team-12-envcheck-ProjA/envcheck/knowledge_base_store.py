"""
EnvCheck Knowledge Base Store — Persistent SQLite + FTS5 backend.

Provides a persistent, searchable store for breaking change rules.
On first use, seeds itself from the builtin rules in knowledge_base.py.
Supports full-text search via SQLite FTS5 and upsert for KB enrichment.

Usage:
    from envcheck.knowledge_base_store import KnowledgeBaseStore

    store = KnowledgeBaseStore()
    rules = store.query(library="numpy")
    results = store.search_fts("trapz removed numpy 2.0")
    store.upsert(new_rule, source="web_search")
"""

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from envcheck.knowledge_base import (
    BREAKING_CHANGES,
    BreakingChangeRule,
    PatternType,
    Severity,
)

DEFAULT_DB_DIR = Path.home() / ".envcheck"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "knowledge_base.db"

_SCHEMA_VERSION = 1


class KnowledgeBaseStore:
    """Persistent knowledge base backed by SQLite with FTS5 full-text search."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._seed_if_empty()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS rules (
                rule_id       TEXT PRIMARY KEY,
                library       TEXT NOT NULL,
                removed_in    TEXT NOT NULL,
                pattern_type  TEXT NOT NULL,
                module_path   TEXT NOT NULL,
                symbol        TEXT NOT NULL,
                old_api       TEXT NOT NULL,
                new_api       TEXT NOT NULL,
                error_type    TEXT NOT NULL,
                description   TEXT NOT NULL,
                severity      TEXT NOT NULL DEFAULT 'error',
                method_kwargs TEXT,
                base_type_hint TEXT,
                source        TEXT NOT NULL DEFAULT 'builtin',
                created_at    REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rules_library ON rules(library);
            CREATE INDEX IF NOT EXISTS idx_rules_symbol  ON rules(symbol);
        """)

        # FTS5 virtual table for full-text search
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='rules_fts'
        """)
        if cur.fetchone() is None:
            cur.execute("""
                CREATE VIRTUAL TABLE rules_fts USING fts5(
                    rule_id, library, symbol, description, old_api, new_api,
                    content='rules',
                    content_rowid='rowid'
                )
            """)
            # Triggers to keep FTS in sync
            cur.executescript("""
                CREATE TRIGGER IF NOT EXISTS rules_ai AFTER INSERT ON rules BEGIN
                    INSERT INTO rules_fts(rowid, rule_id, library, symbol, description, old_api, new_api)
                    VALUES (new.rowid, new.rule_id, new.library, new.symbol,
                            new.description, new.old_api, new.new_api);
                END;

                CREATE TRIGGER IF NOT EXISTS rules_ad AFTER DELETE ON rules BEGIN
                    INSERT INTO rules_fts(rules_fts, rowid, rule_id, library, symbol, description, old_api, new_api)
                    VALUES ('delete', old.rowid, old.rule_id, old.library, old.symbol,
                            old.description, old.old_api, old.new_api);
                END;

                CREATE TRIGGER IF NOT EXISTS rules_au AFTER UPDATE ON rules BEGIN
                    INSERT INTO rules_fts(rules_fts, rowid, rule_id, library, symbol, description, old_api, new_api)
                    VALUES ('delete', old.rowid, old.rule_id, old.library, old.symbol,
                            old.description, old.old_api, old.new_api);
                    INSERT INTO rules_fts(rowid, rule_id, library, symbol, description, old_api, new_api)
                    VALUES (new.rowid, new.rule_id, new.library, new.symbol,
                            new.description, new.old_api, new.new_api);
                END;
            """)

        self._conn.commit()

    def _seed_if_empty(self) -> None:
        """Seed from builtin rules if the database is empty."""
        count = self._conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
        if count == 0:
            self.seed_from_builtin()

    def seed_from_builtin(self) -> int:
        """Insert all builtin rules from knowledge_base.py. Returns count inserted."""
        inserted = 0
        for rule in BREAKING_CHANGES:
            self.upsert(rule, source="builtin")
            inserted += 1
        return inserted

    def upsert(self, rule: BreakingChangeRule, source: str = "builtin") -> None:
        """Insert or replace a rule in the store."""
        method_kwargs_json = json.dumps(rule.method_kwargs) if rule.method_kwargs else None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO rules
                (rule_id, library, removed_in, pattern_type, module_path, symbol,
                 old_api, new_api, error_type, description, severity,
                 method_kwargs, base_type_hint, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rule.rule_id,
                rule.library,
                rule.removed_in,
                rule.pattern_type.value,
                rule.module_path,
                rule.symbol,
                rule.old_api,
                rule.new_api,
                rule.error_type,
                rule.description,
                rule.severity.value,
                method_kwargs_json,
                rule.base_type_hint,
                source,
                time.time(),
            ),
        )
        self._conn.commit()

    def query(
        self,
        library: Optional[str] = None,
        symbol: Optional[str] = None,
        pattern_type: Optional[str] = None,
    ) -> list[BreakingChangeRule]:
        """Query rules by exact match on library, symbol, or pattern_type."""
        clauses = []
        params: list[str] = []

        if library:
            clauses.append("library = ?")
            params.append(library)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if pattern_type:
            clauses.append("pattern_type = ?")
            params.append(pattern_type)

        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM rules WHERE {where}", params
        ).fetchall()
        return [self._row_to_rule(r) for r in rows]

    def search_fts(self, query_text: str, limit: int = 20) -> list[BreakingChangeRule]:
        """Full-text search across rule descriptions, APIs, and library names."""
        safe_query = query_text.replace('"', '""')
        rows = self._conn.execute(
            """
            SELECT r.* FROM rules r
            JOIN rules_fts f ON r.rowid = f.rowid
            WHERE rules_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (safe_query, limit),
        ).fetchall()
        return [self._row_to_rule(r) for r in rows]

    def get_all_for_scanner(self) -> list[BreakingChangeRule]:
        """Return all rules, compatible with the scanner's BREAKING_CHANGES list."""
        rows = self._conn.execute("SELECT * FROM rules").fetchall()
        return [self._row_to_rule(r) for r in rows]

    def get_all_libraries(self) -> set[str]:
        """Return all distinct library names in the store."""
        rows = self._conn.execute("SELECT DISTINCT library FROM rules").fetchall()
        return {r["library"] for r in rows}

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]

    def _row_to_rule(self, row: sqlite3.Row) -> BreakingChangeRule:
        method_kwargs = json.loads(row["method_kwargs"]) if row["method_kwargs"] else None
        return BreakingChangeRule(
            rule_id=row["rule_id"],
            library=row["library"],
            removed_in=row["removed_in"],
            pattern_type=PatternType(row["pattern_type"]),
            module_path=row["module_path"],
            symbol=row["symbol"],
            old_api=row["old_api"],
            new_api=row["new_api"],
            error_type=row["error_type"],
            description=row["description"],
            severity=Severity(row["severity"]),
            method_kwargs=method_kwargs,
            base_type_hint=row["base_type_hint"],
        )

    def to_dict_list(self, rules: list[BreakingChangeRule]) -> list[dict]:
        """Convert rules to serializable dicts (for MCP tool output)."""
        results = []
        for r in rules:
            results.append({
                "rule_id": r.rule_id,
                "library": r.library,
                "removed_in": r.removed_in,
                "pattern_type": r.pattern_type.value,
                "module_path": r.module_path,
                "symbol": r.symbol,
                "old_api": r.old_api,
                "new_api": r.new_api,
                "error_type": r.error_type,
                "description": r.description,
                "severity": r.severity.value,
            })
        return results

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
