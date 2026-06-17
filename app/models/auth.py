from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SlackAuthorizeResponse(BaseModel):
    authorization_url: str
    state_expires_at: datetime


class AuthSessionResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_at: datetime
    tenant_id: str
    tenant_api_base_url: str = Field(default="", description="tenant 앱 진입용 base URL")
    provider: str = "slack"
    external_tenant: str = Field(description="외부 IdP tenant/team 식별값")
    user_id: str
    email: str = ""
    email_verified: bool = False
    name: str = ""
