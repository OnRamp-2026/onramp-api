from app.rag.masker import MarkdownMasker


def test_masker_masks_common_sensitive_values() -> None:
    jwt_like = ".".join(
        [
            "eyJ" + "hbGciOiJIUzI1NiJ9",
            "eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0",
            "signature" + "value123",
        ]
    )
    markdown = f"""
# Secret Doc

Authorization: Bearer abc.def.ghi1234567890
admin@example.com
10.0.12.3
password: my-secret
token: {jwt_like}
"""

    masked = MarkdownMasker().mask(markdown)

    assert "abc.def.ghi1234567890" not in masked
    assert "admin@example.com" not in masked
    assert "10.0.12.3" not in masked
    assert "my-secret" not in masked
    assert jwt_like not in masked
    assert "[MASKED_TOKEN]" in masked
    assert "[MASKED_EMAIL]" in masked
    assert "[MASKED_PRIVATE_IP]" in masked
    assert "[MASKED_SECRET]" in masked


def test_masker_normalizes_spacing_and_balances_code_fence() -> None:
    markdown = "```bash\nkubectl get pods\n\n\n"

    masked = MarkdownMasker().mask(markdown)

    assert masked.count("```") == 2
    assert "\n\n\n" not in masked


def test_masker_masks_64_character_hex_values() -> None:
    hex_secret = "a" * 64
    masked = MarkdownMasker().mask(f"secret: {hex_secret}\n")

    assert hex_secret not in masked
    assert "[MASKED_SECRET]" in masked


def test_masker_counts_indented_code_fences() -> None:
    markdown = "   ```bash\nkubectl get pods\n   ```\n"

    masked = MarkdownMasker().mask(markdown)

    assert masked.count("```") == 2
