import pytest

from app.auth.tenant_registry import TenantContext, make_registry_key, resolve_internal_tenant_id, resolve_tenant_context
from app.config import Settings


def test_make_registry_key_normalizes_provider() -> None:
    assert make_registry_key(provider="Slack", external_tenant="T12345") == "slack:T12345"


def test_resolve_internal_tenant_id_uses_registry_mapping() -> None:
    settings = Settings(_env_file=None, debug=False, tenant_registry={"slack:T12345": "tenant1-onramp"})

    assert (
        resolve_internal_tenant_id(provider="slack", external_tenant="T12345", settings=settings) == "tenant1-onramp"
    )


def test_resolve_internal_tenant_id_rejects_missing_mapping() -> None:
    settings = Settings(_env_file=None, debug=False, tenant_registry={"slack:T12345": "tenant1-onramp"})

    with pytest.raises(ValueError, match="tenant registry에 매핑이 없습니다"):
        resolve_internal_tenant_id(provider="slack", external_tenant="T99999", settings=settings)


def test_resolve_internal_tenant_id_rejects_invalid_target_tenant_id() -> None:
    settings = Settings(_env_file=None, debug=False, tenant_registry={"slack:T12345": "tenant/invalid"})

    with pytest.raises(ValueError, match="tenant_id는 영문자, 숫자, '_', '-'만 사용할 수 있습니다"):
        resolve_internal_tenant_id(provider="slack", external_tenant="T12345", settings=settings)


def test_resolve_tenant_context_supports_extended_registry_entry() -> None:
    settings = Settings(
        _env_file=None,
        debug=False,
        tenant_registry={
            "slack:T12345": {
                "tenant_id": "tenant1-onramp",
                "tenant_api_base_url": "https://tenant1-onramp-api.dev.example.com/",
            }
        },
    )

    assert resolve_tenant_context(provider="slack", external_tenant="T12345", settings=settings) == TenantContext(
        tenant_id="tenant1-onramp",
        tenant_api_base_url="https://tenant1-onramp-api.dev.example.com",
    )


def test_resolve_tenant_context_rejects_invalid_base_url() -> None:
    settings = Settings(
        _env_file=None,
        debug=False,
        tenant_registry={
            "slack:T12345": {
                "tenant_id": "tenant1-onramp",
                "tenant_api_base_url": "tenant1-onramp-api.dev.example.com",
            }
        },
    )

    with pytest.raises(ValueError, match="tenant_api_base_url은 http/https 절대 URL이어야 합니다"):
        resolve_tenant_context(provider="slack", external_tenant="T12345", settings=settings)
