from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.router import build_v1_router


def test_dev_auth_enables_auth_router_without_slack(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_JWT_SECRET", "a" * 32)
    monkeypatch.setenv("AUTH_DEV_TOKEN_ENABLED", "true")
    monkeypatch.setenv("DEBUG", "false")

    from app.config import get_settings

    get_settings.cache_clear()
    app = FastAPI()
    app.include_router(build_v1_router(enable_slack_auth=False, enable_dev_auth=True), prefix="/v1")

    client = TestClient(app)
    response = client.post("/v1/auth/dev-token", json={"tenant_id": "onramp", "subject": "dev-user"})

    assert response.status_code == 200


def test_auth_router_excluded_when_all_auth_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_JWT_SECRET", "a" * 32)
    monkeypatch.setenv("AUTH_DEV_TOKEN_ENABLED", "false")
    monkeypatch.setenv("DEBUG", "false")

    from app.config import get_settings

    get_settings.cache_clear()
    app = FastAPI()
    app.include_router(build_v1_router(enable_slack_auth=False, enable_dev_auth=False), prefix="/v1")

    client = TestClient(app)
    response = client.post("/v1/auth/dev-token", json={"tenant_id": "onramp", "subject": "dev-user"})

    assert response.status_code == 404
