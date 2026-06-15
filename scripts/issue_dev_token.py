"""개발용 STT API 인증 토큰 발급 스크립트.

#98(OIDC 연동) 완료 전까지 /v1/transcriptions* 엔드포인트를 수동으로 호출하기 위한
HS256 JWT를 발급한다. AUTH_JWT_SECRET이 설정된 환경에서만 사용한다.

사용법:
    python scripts/issue_dev_token.py --tenant-id tenant_123
    python scripts/issue_dev_token.py --tenant-id tenant_123 --ttl-seconds 7200
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, help="tenant_id 클레임 값")
    parser.add_argument("--ttl-seconds", type=int, default=3600, help="토큰 유효 기간(초), 기본 3600")
    args = parser.parse_args()

    settings = get_settings()
    secret = settings.auth_jwt_secret.get_secret_value()
    if len(secret) < 32:
        raise SystemExit("AUTH_JWT_SECRET이 설정되지 않았거나 32자 미만입니다.")

    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "tenant_id": args.tenant_id,
        "iat": now,
        "exp": now + timedelta(seconds=args.ttl_seconds),
    }
    if settings.auth_jwt_audience:
        claims["aud"] = settings.auth_jwt_audience
    if settings.auth_jwt_issuer:
        claims["iss"] = settings.auth_jwt_issuer

    token = jwt.encode(claims, secret, algorithm="HS256")
    print(token)


if __name__ == "__main__":
    main()
