from app.api.v1.router import build_v1_router


def _route_paths(router) -> set[str]:
    return {route.path for route in router.routes if hasattr(route, "path")}


def test_dev_auth_enables_auth_router_without_slack() -> None:
    router = build_v1_router(enable_slack_auth=False, enable_dev_auth=True)

    paths = _route_paths(router)
    assert "/auth/dev-token" in paths


def test_auth_router_excluded_when_all_auth_disabled() -> None:
    router = build_v1_router(enable_slack_auth=False, enable_dev_auth=False)

    paths = _route_paths(router)
    assert "/auth/dev-token" not in paths
