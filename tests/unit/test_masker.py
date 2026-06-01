from app.rag.masker import MarkdownMasker


def test_masker_masks_common_sensitive_values() -> None:
    markdown = """
# Secret Doc

Authorization: Bearer abc.def.ghi1234567890
admin@example.com
10.0.12.3
password: my-secret
token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue123
"""

    masked = MarkdownMasker().mask(markdown)

    assert "abc.def.ghi1234567890" not in masked
    assert "admin@example.com" not in masked
    assert "10.0.12.3" not in masked
    assert "my-secret" not in masked
    assert "eyJhbGci" not in masked
    assert "[MASKED_TOKEN]" in masked
    assert "[MASKED_EMAIL]" in masked
    assert "[MASKED_PRIVATE_IP]" in masked
    assert "[MASKED_SECRET]" in masked


def test_masker_normalizes_spacing_and_balances_code_fence() -> None:
    markdown = "```bash\nkubectl get pods\n\n\n"

    masked = MarkdownMasker().mask(markdown)

    assert masked.count("```") == 2
    assert "\n\n\n" not in masked
