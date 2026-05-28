from scripts.random_confluence_page_editor import _upsert_test_section


def test_upsert_test_section_replaces_existing_section() -> None:
    first = _upsert_test_section("<h1>Doc</h1>", "Doc", "2026-05-28 15:00 KST")
    second = _upsert_test_section(first, "Doc", "2026-05-28 15:30 KST")

    assert second.count("ONRAMP_TEST_SECTION_START") == 1
    assert "2026-05-28 15:30 KST" in second
    assert "2026-05-28 15:00 KST" not in second
    assert "breakoutWidth" in second
    assert "kubectl get pod <pod-name>" in second
