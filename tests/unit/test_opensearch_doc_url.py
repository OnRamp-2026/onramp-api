"""_doc/{id} URL의 percent-encoding 회귀 테스트.

GitHub doc_id(gh:repo:docs/x.md)의 '/'·청크의 '#'가 인코딩되지 않으면
URL이 깨져 400(경로분리)나 _id 잘림(fragment)이 발생한다.
"""

from app.db.opensearch import _doc_url


def test_doc_url_encodes_slash() -> None:
    # '/'가 %2F로 인코딩 — 아니면 OpenSearch가 경로로 해석해 400
    assert _doc_url("onramp-documents", "onramp:gh:onramp-api:docs/x.md") == (
        "/onramp-documents/_doc/onramp%3Agh%3Aonramp-api%3Adocs%2Fx.md"
    )


def test_doc_url_encodes_hash() -> None:
    # '#'가 %23로 인코딩 — 아니면 fragment로 잘려 같은 parent 청크끼리 덮어씀
    assert _doc_url("onramp-chunks", "123456#0") == "/onramp-chunks/_doc/123456%230"
    assert _doc_url("onramp-chunks", "123456#0") != _doc_url("onramp-chunks", "123456#1")
