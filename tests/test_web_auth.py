"""Tests for web auth: JWT create/verify, Telegram Login Widget verification.

Run with:  python -m pytest tests/test_web_auth.py -v
    or:    pytest tests/test_web_auth.py -v
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import unittest
from unittest.mock import patch

# Stub the config before importing auth (load_config is called at import time
# in some paths, but our functions always call load_config() internally).
import sys
import os

# Ensure the project root is on sys.path so `auth` and `config` can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# We need to set env vars *before* importing config, because load_config is
# decorated with lru_cache.  We patch at the function level instead.

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FakeConfig:
    bot_token: str = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    web_auth_secret: str = "test_secret_key_for_unit_tests_only"
    web_auth_enabled: bool = True
    web_auth_token_ttl: int = 3600
    miniapp_auth_max_age: int = 3600
    enforce_origin_check: bool = False
    allowed_cors_origins: tuple = ()
    allow_media_query_auth: bool = True


def _fake_load_config():
    return _FakeConfig()


# Patch load_config globally before importing auth module
with patch("config.load_config", _fake_load_config):
    pass  # ensures config module sees our fake during initial import if needed

import auth as auth_module

# Patch load_config in auth module for all test methods
_original_load_config = auth_module.load_config


class TestJWT(unittest.TestCase):
    """Tests for create_web_jwt / verify_web_jwt."""

    def setUp(self):
        auth_module.load_config = _fake_load_config

    def tearDown(self):
        auth_module.load_config = _original_load_config

    def test_create_and_verify_roundtrip(self):
        """A token created by create_web_jwt should be verifiable."""
        payload = {"sub": 12345, "user_id": 12345, "user": {"id": 12345}}
        token = auth_module.create_web_jwt(payload)
        self.assertIsInstance(token, str)
        parts = token.split(".")
        self.assertEqual(len(parts), 3, "JWT must have 3 dot-separated parts")

        result = auth_module.verify_web_jwt(token)
        self.assertEqual(result["sub"], 12345)
        self.assertEqual(result["user_id"], 12345)
        self.assertIn("iat", result)
        self.assertIn("exp", result)

    def test_expired_token(self):
        """An expired JWT must be rejected."""
        cfg = _fake_load_config()
        payload = {"sub": 99, "iat": int(time.time()) - 7200, "exp": int(time.time()) - 3600}
        # Build the token manually with an expired exp
        header_b64 = auth_module._b64url_encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
        )
        payload_b64 = auth_module._b64url_encode(
            json.dumps(payload, separators=(",", ":")).encode()
        )
        signing_input = f"{header_b64}.{payload_b64}"
        sig = hmac.new(
            cfg.web_auth_secret.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        token = f"{signing_input}.{auth_module._b64url_encode(sig)}"

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_web_jwt(token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("expired", ctx.exception.detail.lower())

    def test_invalid_signature(self):
        """A token with a tampered signature must be rejected."""
        payload = {"sub": 12345}
        token = auth_module.create_web_jwt(payload)
        # Flip the last character of the signature
        parts = token.rsplit(".", 1)
        tampered_sig = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
        tampered_token = f"{parts[0]}.{tampered_sig}"

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_web_jwt(tampered_token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("signature", ctx.exception.detail.lower())

    def test_malformed_token_no_dots(self):
        """A string without dots should be rejected."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_web_jwt("notavalidtoken")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_malformed_token_two_dots_bad_base64(self):
        """A three-part string with invalid base64 should be rejected."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_web_jwt("aaa.bbb.!!!")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_token_with_wrong_secret(self):
        """A token signed with a different secret must be rejected."""
        payload = {"sub": 42}
        token = auth_module.create_web_jwt(payload)

        # Now verify with a different secret
        def _different_secret_config():
            return _FakeConfig(web_auth_secret="completely_different_secret")

        auth_module.load_config = _different_secret_config
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_web_jwt(token)
        self.assertEqual(ctx.exception.status_code, 401)


class TestTelegramLoginWidget(unittest.TestCase):
    """Tests for verify_telegram_login_widget."""

    def setUp(self):
        auth_module.load_config = _fake_load_config

    def tearDown(self):
        auth_module.load_config = _original_load_config

    def _make_valid_login_data(self, **overrides):
        """Generate a valid Telegram Login Widget data dict with correct hash."""
        cfg = _fake_load_config()
        now = int(time.time())
        data = {
            "id": 123456789,
            "first_name": "Test",
            "last_name": "User",
            "username": "testuser",
            "auth_date": now,
            **overrides,
        }
        # Compute correct hash
        check_fields = {str(k): str(v) for k, v in data.items() if k != "hash" and v is not None}
        check_string = "\n".join(f"{k}={check_fields[k]}" for k in sorted(check_fields))
        secret_key = hashlib.sha256(cfg.bot_token.encode("utf-8")).digest()
        data["hash"] = hmac.new(secret_key, check_string.encode("utf-8"), hashlib.sha256).hexdigest()
        return data

    def test_valid_login(self):
        """Valid login data should return user info."""
        data = self._make_valid_login_data()
        result = auth_module.verify_telegram_login_widget(data)
        self.assertEqual(result["id"], 123456789)
        self.assertEqual(result["raw_user"]["username"], "testuser")

    def test_missing_hash(self):
        """Data without hash should be rejected."""
        data = self._make_valid_login_data()
        del data["hash"]
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_telegram_login_widget(data)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_invalid_hash(self):
        """Data with wrong hash should be rejected."""
        data = self._make_valid_login_data()
        data["hash"] = "0" * 64  # Wrong hash
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_telegram_login_widget(data)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("invalid", ctx.exception.detail.lower())

    def test_expired_login_data(self):
        """Login data older than max_age should be rejected."""
        data = self._make_valid_login_data(auth_date=int(time.time()) - 7200)
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_telegram_login_widget(data)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("expired", ctx.exception.detail.lower())

    def test_missing_user_id(self):
        """Login data without 'id' should be rejected."""
        data = self._make_valid_login_data()
        # Remove id and recompute hash
        del data["hash"]
        data.pop("id")
        cfg = _fake_load_config()
        check_fields = {str(k): str(v) for k, v in data.items() if v is not None}
        check_string = "\n".join(f"{k}={check_fields[k]}" for k in sorted(check_fields))
        secret_key = hashlib.sha256(cfg.bot_token.encode("utf-8")).digest()
        data["hash"] = hmac.new(secret_key, check_string.encode("utf-8"), hashlib.sha256).hexdigest()

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_telegram_login_widget(data)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_tampered_field(self):
        """Changing a field after hashing should invalidate the data."""
        data = self._make_valid_login_data()
        data["first_name"] = "Hacker"  # Tampered field, hash is now wrong
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth_module.verify_telegram_login_widget(data)
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
