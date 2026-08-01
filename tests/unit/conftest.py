"""유닛 테스트 격리 — 로컬 `.env`가 새어들지 않게 한다.

`Settings`는 pydantic-settings라 `Settings()`를 인자 없이 부르면
`model_config["env_file"]`(=`.env`)을 읽어 **코드 기본값 위에 덮어쓴다**.
유닛 테스트는 "코드에 적힌 기본값"을 검증하므로 이게 섞이면 안 된다.

이걸 막지 않으면 `.env`를 채워둔 로컬에서만 테스트가 깨진다(CI는 `.env`가
없어서 통과). 예: `test_langfuse_disabled_by_default`가 `assert True is False`.

환경변수(`os.environ`)는 그대로 둔다 — CI가 의도적으로 주입하는 경로이고,
`client` 픽스처도 `DEBUG`를 환경변수로 세팅한다. 차단 대상은 `.env` 파일뿐이다.

통합 테스트에는 적용하지 않는다(`tests/integration/`은 실 자격증명·호스트가 필요).
"""

import pytest


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    from app.config import Settings, get_settings

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    # get_settings는 lru_cache라 이전 테스트가 만든 .env 반영 인스턴스가 남을 수 있다.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
