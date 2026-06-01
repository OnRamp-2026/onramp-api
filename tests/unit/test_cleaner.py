from app.rag.cleaner import TextCleaner


def test_clean_removes_noise_and_preserves_content() -> None:
    html = """
    <main>
      <nav>Navigation</nav>
      <div class="td-sidebar">Sidebar</div>
      <h1>Runbook</h1>
      <p>Restart the API pod.</p>
    </main>
    """

    markdown = TextCleaner().clean(html)

    assert "# Runbook" in markdown
    assert "Restart the API pod." in markdown
    assert "Navigation" not in markdown
    assert "Sidebar" not in markdown


def test_clean_converts_confluence_code_macro_without_layout_parameters() -> None:
    html = """
    <ac:structured-macro ac:name="code" ac:schema-version="1">
      <ac:parameter ac:name="breakoutMode">wide</ac:parameter>
      <ac:parameter ac:name="breakoutWidth">760</ac:parameter>
      <ac:plain-text-body><![CDATA[kubectl get pod <pod-name>
]]></ac:plain-text-body>
    </ac:structured-macro>
    """

    markdown = TextCleaner().clean(html)

    assert "kubectl get pod <pod-name>" in markdown
    assert "wide760" not in markdown
    assert "breakoutMode" not in markdown
    assert "breakoutWidth" not in markdown


def test_clean_removes_non_code_confluence_macros_and_layout_noise() -> None:
    html = """
    <main>
      <p>#F4F5F7</p>
      <ac:structured-macro ac:name="recently-updated">
        <ac:parameter ac:name="types">page,whiteboard,database,blog</ac:parameter>
        <ac:parameter ac:name="max">10</ac:parameter>
        <ac:parameter ac:name="theme">concise</ac:parameter>
        <ac:parameter ac:name="hideHeading">true</ac:parameter>
      </ac:structured-macro>
      <p>Actual content.</p>
    </main>
    """

    markdown = TextCleaner().clean(html)

    assert "Actual content." in markdown
    assert "#F4F5F7" not in markdown
    assert "page,whiteboard,database,blog10concisetrue" not in markdown


def test_clean_converts_images_and_links() -> None:
    html = """
    <main>
      <p><img src="/diagram.png" alt="request flow"/></p>
      <p><a href="/docs/internal">Internal Doc</a></p>
      <p><a href="https://example.com/runbook">External Runbook</a></p>
    </main>
    """

    markdown = TextCleaner().clean(html)

    assert "[이미지: request flow]" in markdown
    assert "Internal Doc" in markdown
    assert "/docs/internal" not in markdown
    assert "External Runbook (https://example.com/runbook)" in markdown
