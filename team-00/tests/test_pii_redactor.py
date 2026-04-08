"""
Tests for src/agent/tools/pii_redactor.py — PII detection and redaction.

Run from team-00/:
    pytest tests/test_pii_redactor.py -v
"""
import pytest
from src.agent.tools.pii_redactor import redact_text, redact_file


class TestEmailRedaction:
    def test_simple_email(self):
        result = redact_text("Contact me at alice@example.com for details.")
        assert "alice@example.com" not in result
        assert "[██ EMAIL]" in result

    def test_email_with_subdomains(self):
        result = redact_text("Send to bob@mail.company.org please.")
        assert "bob@mail.company.org" not in result

    def test_no_email_unchanged(self):
        text = "No contact info here."
        assert redact_text(text) == text


class TestPhoneRedaction:
    def test_us_phone_dashes(self):
        result = redact_text("Call 555-867-5309 now.")
        assert "555-867-5309" not in result
        assert "[██ PHONE]" in result

    def test_us_phone_dots(self):
        result = redact_text("Phone: 800.555.1234")
        assert "800.555.1234" not in result

    def test_us_phone_with_country_code(self):
        result = redact_text("Reach me at +1 (800) 555-0199.")
        assert "555-0199" not in result


class TestSSNRedaction:
    def test_ssn_pattern(self):
        result = redact_text("SSN: 123-45-6789")
        assert "123-45-6789" not in result
        assert "[██ SSN]" in result

    def test_no_partial_ssn_false_positive(self):
        # 12-34 is not an SSN (only 2 groups not 3)
        result = redact_text("Code: 12-34")
        assert "[██ SSN]" not in result


class TestIPAddressRedaction:
    def test_ipv4_address(self):
        result = redact_text("Server at 192.168.1.100.")
        assert "192.168.1.100" not in result
        assert "[██ IP]" in result

    def test_loopback(self):
        result = redact_text("localhost is 127.0.0.1")
        assert "127.0.0.1" not in result


class TestMultiplePIITypes:
    def test_mixed_pii_all_redacted(self):
        text = "Email: user@test.com, Phone: 555-123-4567, SSN: 987-65-4321"
        result = redact_text(text)
        assert "user@test.com" not in result
        assert "555-123-4567" not in result
        assert "987-65-4321" not in result
        assert "[██ EMAIL]" in result
        assert "[██ PHONE]" in result
        assert "[██ SSN]" in result

    def test_clean_text_not_modified(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert redact_text(text) == text


class TestRedactFile:
    def test_redact_file_produces_output(self, tmp_path):
        src = tmp_path / "input.txt"
        src.write_text("Email: secret@domain.com")
        out = tmp_path / "output.txt"
        redact_file(str(src), str(out))
        content = out.read_text()
        assert "secret@domain.com" not in content
        assert "[██ EMAIL]" in content

    def test_redact_file_missing_source_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            redact_file("/nonexistent/file.txt", str(tmp_path / "out.txt"))
