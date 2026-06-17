from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from app.config import Settings

_TENANT_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{1,128}$")
_PROVIDER_PATTERN = re.compile(r"^[0-9A-Za-z_-]{1,64}$")


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    tenant_api_base_url: str = ""
    allowed_hosts: tuple[str, ...] = ()


def _validate_tenant_id(value: str, *, field_name: str) -> str:
    if not _TENANT_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name}는 영문자, 숫자, '_', '-'만 사용할 수 있습니다.")
    return value


def _validate_provider(value: str) -> str:
    normalized = value.strip().lower()
    if not _PROVIDER_PATTERN.fullmatch(normalized):
        raise ValueError("provider는 영문자, 숫자, '_', '-'만 사용할 수 있습니다.")
    return normalized


def make_registry_key(*, provider: str, external_tenant: str) -> str:
    normalized_provider = _validate_provider(provider)
    normalized_external_tenant = external_tenant.strip()
    if not normalized_external_tenant:
        raise ValueError("external_tenant은 비어 있을 수 없습니다.")
    return f"{normalized_provider}:{normalized_external_tenant}"


def _validate_tenant_api_base_url(value: object) -> str:
    if value in ("", None):
        return ""
    if not isinstance(value, str):
        raise ValueError("tenant_api_base_url은 문자열이어야 합니다.")
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("tenant_api_base_url은 http/https 절대 URL이어야 합니다.")
    return normalized


def _validate_allowed_hosts(value: object) -> tuple[str, ...]:
    if value in ("", None):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("allowed_hosts는 문자열 배열이어야 합니다.")
    hosts: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("allowed_hosts 항목은 문자열이어야 합니다.")
        normalized = item.strip().lower().rstrip(".")
        if not normalized:
            continue
        if "://" in normalized:
            parsed = urlparse(normalized)
            normalized = parsed.netloc.lower().rstrip(".")
        if "/" in normalized or not normalized:
            raise ValueError("allowed_hosts 항목은 host 또는 host:port 형식이어야 합니다.")
        hosts.append(normalized)
    return tuple(dict.fromkeys(hosts))


def resolve_tenant_context(*, provider: str, external_tenant: str, settings: Settings) -> TenantContext:
    registry_key = make_registry_key(provider=provider, external_tenant=external_tenant)
    entry = settings.tenant_registry.get(registry_key)
    if entry is None:
        raise ValueError(f"tenant registry에 매핑이 없습니다: {registry_key}")
    if isinstance(entry, str):
        return TenantContext(tenant_id=_validate_tenant_id(entry, field_name="tenant_id"))
    if not isinstance(entry, dict):
        raise ValueError("tenant registry 값은 문자열 또는 객체여야 합니다.")

    tenant_id = _validate_tenant_id(str(entry.get("tenant_id", "")), field_name="tenant_id")
    tenant_api_base_url = _validate_tenant_api_base_url(entry.get("tenant_api_base_url", ""))
    allowed_hosts = _validate_allowed_hosts(entry.get("allowed_hosts", ()))
    return TenantContext(
        tenant_id=tenant_id,
        tenant_api_base_url=tenant_api_base_url,
        allowed_hosts=allowed_hosts,
    )


def resolve_internal_tenant_id(*, provider: str, external_tenant: str, settings: Settings) -> str:
    return resolve_tenant_context(
        provider=provider,
        external_tenant=external_tenant,
        settings=settings,
    ).tenant_id
