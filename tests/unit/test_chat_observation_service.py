from app.services.chat_observation_service import classify_result_bucket


def test_classify_result_bucket_success() -> None:
    assert classify_result_bucket("answerable", 0) == "success"
    assert classify_result_bucket("partially_answerable", 0) == "success"


def test_classify_result_bucket_requery() -> None:
    assert classify_result_bucket("answerable", 1) == "requery"


def test_classify_result_bucket_failure() -> None:
    assert classify_result_bucket("not_enough_evidence", 0) == "failure"
    assert classify_result_bucket("", 0, failed=True) == "failure"
