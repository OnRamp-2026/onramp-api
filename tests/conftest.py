import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    """FastAPI 테스트 클라이언트."""
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
