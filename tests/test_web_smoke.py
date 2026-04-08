"""Web mode smoke tests: cookie auth flow, protected API, logout.

These tests verify the web auth session lifecycle via real HTTP requests to
the FastAPI application using Starlette's TestClient.  They do NOT require a
running Telegram bot or an actual database — the minimal test config and
mock DB layer are enough to exercise the auth plumbing.

Run with:  BOT_TOKEN=test:token python -m pytest tests/test_web_smoke.py -v
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import unittest
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Minimal fake config
# ---------------------------------------------------------------------------

@dataclass
class _FakeConfig:
    bot_token: str = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    web_auth_secret: str = "test_secret_key_for_smoke_tests_only"
    web_auth_enabled: bool = True
    web_auth_token_ttl: int = 3600
    miniapp_auth_max_age: int = 3600
    enforce_origin_check: bool = False
    allowed_cors_origins: tuple = ("http://testserver",)
    allow_media_query_auth: bool = True
    trusted_hosts: tuple = ("*",)
    api_rate_limit_rpm: int = 1000
    api_write_rate_limit_rpm: int = 500
    enable_docs: bool = False
    miniapp_public_origin: str = "http://testserver"
    bot_username: str = "testbot"
    miniapp_url: str = "http://testserver"
    app_env: str = "test"


def _fake_load_config():
    return _FakeConfig()


def _make_tg_login_data(bot_token: str = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", **overrides) -> dict:
    """Build Telegram Login Widget data with a valid HMAC hash."""
    now = int(time.time())
    data: dict[str, Any] = {
        "id": 999888777,
        "first_name": "Smoke",
        "last_name": "Test",
        "username": "smoketestuser",
        "auth_date": now,
        **overrides,
    }
    check_fields = {str(k): str(v) for k, v in data.items() if k != "hash" and v is not None}
    check_string = "\n".join(f"{k}={check_fields[k]}" for k in sorted(check_fields))
    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    data["hash"] = hmac.new(secret_key, check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    return data


_SESSION_COOKIE_NAME = "neurosmm_session"


class TestWebSmokeWithTestClient(unittest.TestCase):
    """Smoke tests for the web auth cookie session lifecycle using Starlette TestClient."""

    @classmethod
    def setUpClass(cls):
        import config
        config.load_config = _fake_load_config
        import auth as auth_module
        auth_module.load_config = _fake_load_config
        import miniapp_server
        cls._app = miniapp_server.app

    def setUp(self):
        import config
        config.load_config = _fake_load_config
        import auth as auth_module
        auth_module.load_config = _fake_load_config

    def _client(self):
        from starlette.testclient import TestClient
        return TestClient(self._app, raise_server_exceptions=False)

    # --- 1. Health check (no auth needed) ---

    def test_healthz_ok(self):
        """GET /healthz should return 200 without auth."""
        with self._client() as c:
            r = c.get("/healthz")
            self.assertEqual(r.status_code, 200)

    # --- 2. web-auth/config (public) ---

    def test_auth_config_public(self):
        """GET /api/web-auth/config should return enabled=True and bot info."""
        with self._client() as c:
            r = c.get("/api/web-auth/config")
            self.assertEqual(r.status_code, 200)
            d = r.json()
            self.assertTrue(d["enabled"])
            self.assertEqual(d["bot_username"], "testbot")

    # --- 3. Login sets HttpOnly cookie ---

    def test_login_sets_httponly_cookie(self):
        """POST /api/web-auth/telegram-login should set an HttpOnly session cookie."""
        login_data = _make_tg_login_data()
        with self._client() as c:
            r = c.post("/api/web-auth/telegram-login", json=login_data)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertTrue(r.json().get("ok"))
            raw = r.headers.get("set-cookie", "")
            self.assertIn("neurosmm_session", raw)
            self.assertIn("httponly", raw.lower())
            self.assertIn("samesite=lax", raw.lower())

    # --- 4. Invalid login data rejected ---

    def test_login_bad_hash_401(self):
        """POST /api/web-auth/telegram-login with invalid hash should fail."""
        data = _make_tg_login_data()
        data["hash"] = "bad" * 21 + "ab"  # 64 chars but wrong
        with self._client() as c:
            r = c.post("/api/web-auth/telegram-login", json=data)
            self.assertEqual(r.status_code, 401)

    # --- 5. Protected API fails without cookie ---

    def test_no_cookie_gets_401(self):
        """GET /api/bootstrap/core without session cookie should return 401."""
        with self._client() as c:
            r = c.get("/api/bootstrap/core")
            self.assertEqual(r.status_code, 401)

    # --- 6. Protected API works with valid cookie ---

    def test_valid_cookie_not_401(self):
        """A valid JWT cookie should pass auth (may get 500 from DB but never 401)."""
        from auth import create_web_jwt, WEB_SESSION_COOKIE
        token = create_web_jwt({"sub": 999888777, "user_id": 999888777, "user": {"id": 999888777}})
        with self._client() as c:
            c.cookies.set(WEB_SESSION_COOKIE, token)
            r = c.get("/api/bootstrap/core")
            self.assertNotEqual(r.status_code, 401, "Valid cookie should not get 401")

    # --- 7. Expired token returns 401 ---

    def test_expired_cookie_gets_401(self):
        """A request with an expired JWT cookie should return 401."""
        import auth as auth_module
        cfg = _fake_load_config()
        now = int(time.time())
        payload = {"sub": 999, "user_id": 999, "iat": now - 7200, "exp": now - 3600}
        header_b64 = auth_module._b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
        payload_b64 = auth_module._b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
        signing_input = f"{header_b64}.{payload_b64}"
        sig = hmac.new(cfg.web_auth_secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
        expired_token = f"{signing_input}.{auth_module._b64url_encode(sig)}"

        from auth import WEB_SESSION_COOKIE
        with self._client() as c:
            c.cookies.set(WEB_SESSION_COOKIE, expired_token)
            r = c.get("/api/bootstrap/core")
            self.assertEqual(r.status_code, 401)

    # --- 8. Logout clears cookie ---

    def test_logout_clears_cookie(self):
        """POST /api/web-auth/logout should delete the session cookie."""
        with self._client() as c:
            r = c.post("/api/web-auth/logout")
            self.assertEqual(r.status_code, 200)
            raw = r.headers.get("set-cookie", "")
            self.assertIn("neurosmm_session", raw)
            # Starlette delete_cookie sets max-age=0
            low = raw.lower()
            self.assertTrue(
                "max-age=0" in low or 'expires=' in low,
                f"Cookie not properly cleared: {raw}"
            )

    # --- 9. Full lifecycle: login -> protected -> logout ---

    def test_full_login_then_protected_then_logout(self):
        """Full session lifecycle: login -> access protected endpoint -> logout."""
        login_data = _make_tg_login_data()
        with self._client() as c:
            # Step 1: Login -- get cookie from response
            r = c.post("/api/web-auth/telegram-login", json=login_data)
            self.assertEqual(r.status_code, 200)
            # Extract the cookie from Set-Cookie header
            from auth import WEB_SESSION_COOKIE
            session_cookie = None
            for val in r.headers.get_list("set-cookie"):
                if WEB_SESSION_COOKIE in val:
                    session_cookie = val.split("=", 1)[1].split(";")[0]
                    break
            self.assertIsNotNone(session_cookie, "Session cookie not found in login response")

            # Step 2: Access protected endpoint with the cookie
            c.cookies.set(WEB_SESSION_COOKIE, session_cookie)
            r2 = c.get("/api/bootstrap/core")
            self.assertNotEqual(r2.status_code, 401, "Should be authenticated after login")

            # Step 3: Logout
            r3 = c.post("/api/web-auth/logout")
            self.assertEqual(r3.status_code, 200)

    # --- 10. Web-auth disabled returns 403 for login ---

    def test_login_when_web_auth_disabled(self):
        """Login should return 403 when WEB_AUTH_ENABLED=false."""
        def _disabled_config():
            return _FakeConfig(web_auth_enabled=False)

        import auth as auth_module
        import config
        import miniapp_server
        auth_module.load_config = _disabled_config
        config.load_config = _disabled_config
        miniapp_server.load_config = _disabled_config
        try:
            login_data = _make_tg_login_data()
            with self._client() as c:
                r = c.post("/api/web-auth/telegram-login", json=login_data)
                self.assertEqual(r.status_code, 403)
        finally:
            auth_module.load_config = _fake_load_config
            config.load_config = _fake_load_config
            miniapp_server.load_config = _fake_load_config


if __name__ == "__main__":
    unittest.main()
