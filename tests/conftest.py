import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    """FastAPI 테스트 클라이언트. chat/asset이 인증 필수(#163)이므로 fake 인증 user를 주입한다.

    subject=""로 두면 chat이 대화 기록 persist를 건너뛰어 DB 의존 없이 통과한다.
    미인증(401) 검증은 테스트 내에서 이 override를 제거하고 호출한다.
    """
    import os

    os.environ["DEBUG"] = "false"
    from app.config import get_settings

    get_settings.cache_clear()
    from app.auth.session import SessionUser, get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: SessionUser(
        tenant_id="test-tenant", subject="", provider="test", name="Test", email="test@example.com", claims={}
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.pop(get_current_user, None)
