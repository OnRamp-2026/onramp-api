from app.api.v1.router import build_v1_router


def test_dev_auth_enables_auth_router_without_slack() -> None:
    router = build_v1_router(enable_slack_auth=False, enable_dev_auth=True)

    paths = {route.path for route in router.routes}
    assert "/auth/dev-token" in paths


def test_auth_router_excluded_when_all_auth_disabled() -> None:
    router = build_v1_router(enable_slack_auth=False, enable_dev_auth=False)

    paths = {route.path for route in router.routes}
    assert "/auth/dev-token" not in paths
