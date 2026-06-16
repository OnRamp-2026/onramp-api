from app.rag.rrf import RankedItem, reciprocal_rank_fusion


def test_rrf_merges_by_id_and_sums_rank_scores():
    fused = reciprocal_rank_fusion(
        (
            (
                "dense",
                [RankedItem(id="a", score=0.9, payload={"chunk_id": "a"}), RankedItem(id="b", score=0.8, payload={})],
            ),
            (
                "bm25",
                [RankedItem(id="b", score=12.0, payload={"chunk_id": "b"}), RankedItem(id="c", score=8.0, payload={})],
            ),
        ),
        k=60,
    )

    assert [item.id for item in fused] == ["b", "a", "c"]
    assert fused[0].source_scores == {"dense": 0.8, "bm25": 12.0}


def test_rrf_applies_limit():
    fused = reciprocal_rank_fusion(
        (
            (
                "dense",
                [
                    RankedItem(id="a", score=1.0, payload={}),
                    RankedItem(id="b", score=0.9, payload={}),
                ],
            ),
        ),
        limit=1,
    )

    assert [item.id for item in fused] == ["a"]
