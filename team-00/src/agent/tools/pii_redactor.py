"""
PII Redactor — Detect and redact sensitive information in Markdown files.

Supports: email, phone, SSN, credit card, IP address, physical address,
person names (after contextual keywords).
"""

import re
from pathlib import Path

# ── PII patterns ─────────────────────────────────────────────

_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # (label, replacement token, compiled regex)
    (
        "EMAIL",
        "[██ EMAIL]",
        re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ),
    (
        "PHONE",
        "[██ PHONE]",
        re.compile(
            r"(?<!\d)"                              # no digit before
            r"(?:\+?1[\s\-.]?)?"                    # optional +1 prefix
            r"\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
            r"(?!\d)"                               # no digit after
        ),
    ),
    (
        "SSN",
        "[██ SSN]",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "CREDIT_CARD",
        "[██ CREDIT CARD]",
        re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    ),
    (
        "IP_ADDRESS",
        "[██ IP]",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
    ),
    (
        "DATE_OF_BIRTH",
        "[██ DOB]",
        re.compile(
            r"(?i)(?:date\s*of\s*birth|dob|born\s*on|birthday)"
            r"[\s:]*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"
        ),
    ),
    (
        "ADDRESS",
        "[██ ADDRESS]",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3}"
            r"\s+(?:St(?:reet)?|Ave(?:nue)?|Blvd|Dr(?:ive)?|Rd|Road|Ln|Lane|Ct|Court|Way|Pl(?:ace)?|Cir(?:cle)?)"
            r"\.?"
            r"(?:\s*,?\s*(?:Apt|Suite|Unit|#)\s*\w+)?"
            r"(?:\s*,\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)?"    # city
            r"(?:\s*,\s*[A-Z]{2})?"                                   # state
            r"(?:\s+\d{5}(?:-\d{4})?)?"                               # zip
        ),
    ),
    (
        "PERSON_NAME",
        "[██ NAME]",
        re.compile(
            r"(?i)(?:name|patient|author|contact|investigator|employee|manager|supervisor|client)"
            r"[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
        ),
    ),
]


class PIIRedactor:
    """Detect and redact PII in text / Markdown files."""

    name = "pii_redactor"
    description = "Detect and redact sensitive personal information in Markdown documents"

    def redact_text(self, text: str) -> tuple[str, dict[str, int]]:
        """Redact all PII from text.

        Returns:
            (redacted_text, counts_dict)
        """
        counts: dict[str, int] = {}
        redacted = text

        for label, token, pattern in _PATTERNS:
            matches = pattern.findall(redacted)
            if matches:
                counts[label] = len(matches)
                redacted = pattern.sub(token, redacted)

        return redacted, counts

    def process_file(self, input_path: str, output_path: str) -> dict:
        """Redact PII from a single Markdown file.

        Returns dict with status, pii_counts, etc.
        """
        inp = Path(input_path)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            original = inp.read_text(encoding="utf-8")
            redacted, counts = self.redact_text(original)
            out.write_text(redacted, encoding="utf-8")

            total_pii = sum(counts.values())
            return {
                "status": "success",
                "file": inp.name,
                "pii_counts": counts,
                "total_pii": total_pii,
                "original_length": len(original),
                "redacted_length": len(redacted),
            }
        except Exception as e:
            return {"status": "error", "file": inp.name, "error": str(e)}

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: tuple[str, ...] = (".md", ".txt", ".markdown"),
    ) -> dict:
        """Redact PII from all matching files in a directory."""
        inp_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        aggregate_counts: dict[str, int] = {}
        processed = 0
        failed = 0

        for fpath in sorted(inp_dir.iterdir()):
            if fpath.suffix.lower() not in extensions:
                continue
            out_path = out_dir / fpath.name
            result = self.process_file(str(fpath), str(out_path))
            results.append(result)

            if result["status"] == "success":
                processed += 1
                for k, v in result["pii_counts"].items():
                    aggregate_counts[k] = aggregate_counts.get(k, 0) + v
            else:
                failed += 1

        return {
            "processed": processed,
            "failed": failed,
            "total_pii": sum(aggregate_counts.values()),
            "pii_counts": aggregate_counts,
            "files": results,
        }
