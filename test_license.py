"""
test_license.py — Comprehensive tests for license.py (Gumroad backend)

Coverage:
  - load_license: missing file, corrupt JSON, non-dict, forward-compat keys
  - save_license: success, disk error, dir creation
  - check_startup: grace math (3 launches free), activated, required state
  - activate_license: empty key, success, save fails, verify fails, API errors
  - validate_license_silent: whitespace key, valid, revoked, network down → True
  - deactivate_license: no license, success (local-only), save fails
  - _gr_post: timeout, SSL, URLError (DNS/refused), HTTPError, bad JSON response
  - _extract_error: network flag, message field, status codes, unknown fallback
"""
import io
import json
import os
import socket
import ssl
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import license as lic
from license import (
    GRACE_LAUNCHES,
    _DEFAULT_LICENSE,
    _extract_error,
    _gr_post,
    _verify_license,
    _PRODUCT_ID_RE,
    activate_license,
    check_startup,
    deactivate_license,
    load_license,
    save_license,
    validate_license_silent,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_license(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _activated_license(key="ABCD-1234"):
    d = _DEFAULT_LICENSE.copy()
    d.update({"key": key, "activated": True,
               "activation_date": "2024-01-01T00:00:00"})
    return d


def _make_urlopen_mock(payload, status=200):
    """Return a context-manager mock that yields payload as JSON."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(code, body=None):
    body_bytes = json.dumps(body or {}).encode() if body is not None else b""
    return urllib.error.HTTPError(
        url="https://api.gumroad.com/v2/licenses/verify",
        code=code, msg="Error", hdrs=None,
        fp=io.BytesIO(body_bytes),
    )


# ── TestLoadLicense ───────────────────────────────────────────────────────────

class TestLoadLicense(unittest.TestCase):

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def test_missing_file_returns_default(self):
        self.assertEqual(load_license(self.path), _DEFAULT_LICENSE)

    def test_valid_file_returns_data(self):
        _write_license(self.path, _activated_license())
        result = load_license(self.path)
        self.assertEqual(result["key"], "ABCD-1234")
        self.assertTrue(result["activated"])

    def test_corrupt_json_returns_default(self):
        with open(self.path, "w") as f:
            f.write("{not valid json!!!")
        self.assertEqual(load_license(self.path), _DEFAULT_LICENSE)

    def test_non_dict_json_returns_default(self):
        with open(self.path, "w") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(load_license(self.path), _DEFAULT_LICENSE)

    def test_null_json_returns_default(self):
        with open(self.path, "w") as f:
            f.write("null")
        self.assertEqual(load_license(self.path), _DEFAULT_LICENSE)

    def test_forward_compat_missing_keys_added(self):
        """Partial file (old version) gets missing keys filled in."""
        _write_license(self.path, {"key": "OLD-KEY", "activated": False})
        result = load_license(self.path)
        for k in _DEFAULT_LICENSE:
            self.assertIn(k, result, f"Missing key: {k}")
        self.assertEqual(result["key"], "OLD-KEY")
        self.assertEqual(result["launch_count"], 0)

    def test_empty_file_returns_default(self):
        open(self.path, "w").close()
        self.assertEqual(load_license(self.path), _DEFAULT_LICENSE)

    def test_all_default_keys_present(self):
        result = load_license(self.path)
        for k in _DEFAULT_LICENSE:
            self.assertIn(k, result)

    def test_does_not_raise_on_read_error(self):
        with patch("builtins.open", side_effect=OSError("disk error")):
            with patch("os.path.exists", return_value=True):
                result = load_license(self.path)
        self.assertEqual(result, _DEFAULT_LICENSE)


# ── TestSaveLicense ───────────────────────────────────────────────────────────

class TestSaveLicense(unittest.TestCase):

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def test_save_and_reload(self):
        ok = save_license(_activated_license(), self.path)
        self.assertTrue(ok)
        self.assertEqual(load_license(self.path)["key"], "ABCD-1234")

    def test_returns_true_on_success(self):
        self.assertTrue(save_license(_DEFAULT_LICENSE.copy(), self.path))

    def test_returns_false_on_write_error(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            self.assertFalse(save_license(_DEFAULT_LICENSE.copy(), self.path))

    def test_creates_parent_directory(self):
        nested = os.path.join(self.td.name, "deep", "nested", "license.json")
        self.assertTrue(save_license(_DEFAULT_LICENSE.copy(), nested))
        self.assertTrue(os.path.exists(nested))

    def test_does_not_raise_on_makedirs_error(self):
        with patch("os.makedirs", side_effect=OSError("perm denied")):
            try:
                save_license(_DEFAULT_LICENSE.copy(), self.path)
            except Exception as e:
                self.fail(f"save_license raised: {e}")

    def test_saved_file_is_valid_json(self):
        save_license(_activated_license(), self.path)
        with open(self.path, encoding="utf-8") as f:
            self.assertIsInstance(json.load(f), dict)


# ── TestCheckStartup ──────────────────────────────────────────────────────────

class TestCheckStartup(unittest.TestCase):

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def test_activated_returns_ok(self):
        _write_license(self.path, _activated_license())
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "ok")
        self.assertEqual(remaining, 0)

    def test_first_launch_is_grace(self):
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "grace")
        self.assertEqual(remaining, GRACE_LAUNCHES - 1)

    def test_second_launch_decrements_remaining(self):
        check_startup(self.path)
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "grace")
        self.assertEqual(remaining, GRACE_LAUNCHES - 2)

    def test_last_free_launch(self):
        for _ in range(GRACE_LAUNCHES - 1):
            check_startup(self.path)
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "grace")
        self.assertEqual(remaining, 0)

    def test_launch_after_grace_is_required(self):
        for _ in range(GRACE_LAUNCHES):
            check_startup(self.path)
        state, _ = check_startup(self.path)
        self.assertEqual(state, "required")

    def test_launch_count_persists(self):
        check_startup(self.path)
        check_startup(self.path)
        self.assertEqual(load_license(self.path)["launch_count"], 2)

    def test_activated_does_not_increment_count(self):
        d = _activated_license()
        d["launch_count"] = 5
        _write_license(self.path, d)
        check_startup(self.path)
        self.assertEqual(load_license(self.path)["launch_count"], 5)

    def test_missing_file_treated_as_launch_1(self):
        state, _ = check_startup(self.path)
        self.assertEqual(state, "grace")

    def test_grace_launches_constant_is_3(self):
        self.assertEqual(GRACE_LAUNCHES, 3)

    def test_exactly_grace_launches_free(self):
        free = 0
        for _ in range(GRACE_LAUNCHES + 2):
            state, _ = check_startup(self.path)
            if state == "grace":
                free += 1
        self.assertEqual(free, GRACE_LAUNCHES)


# ── TestActivateLicense ───────────────────────────────────────────────────────

class TestActivateLicense(unittest.TestCase):

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def _mock_success(self, key="TEST-KEY-1234"):
        """Gumroad success response: {success: true, purchase: {...}}"""
        return _make_urlopen_mock({
            "success": True,
            "purchase": {"license_key": key, "email": "user@example.com"},
        })

    def test_empty_key_returns_error(self):
        ok, msg = activate_license("", self.path)
        self.assertFalse(ok)
        self.assertIn("enter a license key", msg.lower())

    def test_whitespace_key_returns_error(self):
        ok, msg = activate_license("   ", self.path)
        self.assertFalse(ok)
        self.assertIn("enter a license key", msg.lower())

    def test_none_key_returns_error(self):
        ok, msg = activate_license(None, self.path)
        self.assertFalse(ok)

    @patch("urllib.request.urlopen")
    def test_successful_activation(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY")
        ok, msg = activate_license("GOOD-KEY", self.path)
        self.assertTrue(ok)
        self.assertIn("activated successfully", msg.lower())

    @patch("urllib.request.urlopen")
    def test_activation_persists_to_file(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY")
        activate_license("GOOD-KEY", self.path)
        data = load_license(self.path)
        self.assertTrue(data["activated"])
        self.assertEqual(data["key"], "GOOD-KEY")

    @patch("urllib.request.urlopen")
    def test_activation_date_set(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY")
        activate_license("GOOD-KEY", self.path)
        self.assertIsNotNone(load_license(self.path)["activation_date"])

    @patch("urllib.request.urlopen")
    def test_save_failure_returns_error(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY")
        with patch("license.save_license", return_value=False):
            ok, msg = activate_license("GOOD-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("could not be saved", msg.lower())

    @patch("urllib.request.urlopen")
    def test_verify_failure_returns_error(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY")
        with patch("license.save_license", return_value=True):
            with patch("license.load_license", side_effect=[
                _DEFAULT_LICENSE.copy(), _DEFAULT_LICENSE.copy()
            ]):
                ok, msg = activate_license("GOOD-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("verification failed", msg.lower())

    @patch("urllib.request.urlopen")
    def test_api_returns_success_false(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({
            "success": False,
            "message": "That license does not exist.",
        })
        ok, msg = activate_license("BAD-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("does not exist", msg.lower())

    @patch("urllib.request.urlopen")
    def test_http_404_returns_friendly_message(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(404)
        ok, msg = activate_license("MISSING-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("not found", msg.lower())

    @patch("urllib.request.urlopen")
    def test_http_403_revoked_key(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(403)
        ok, msg = activate_license("REVOKED-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("disabled", msg.lower())

    @patch("urllib.request.urlopen")
    def test_http_429_rate_limit(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(429)
        ok, msg = activate_license("ANY-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("too many", msg.lower())

    @patch("urllib.request.urlopen")
    def test_network_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = socket.timeout("timed out")
        ok, msg = activate_license("ANY-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("timed out", msg.lower())

    @patch("urllib.request.urlopen")
    def test_ssl_error(self, mock_urlopen):
        mock_urlopen.side_effect = ssl.SSLError("certificate verify failed")
        ok, msg = activate_license("ANY-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("secure connection", msg.lower())

    @patch("urllib.request.urlopen")
    def test_dns_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError(
            "[Errno -2] Name or service not known"
        )
        ok, msg = activate_license("ANY-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("internet connection", msg.lower())

    @patch("urllib.request.urlopen")
    def test_key_stripped_before_use(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("CLEAN-KEY")
        activate_license("  CLEAN-KEY  ", self.path)
        self.assertEqual(load_license(self.path)["key"], "CLEAN-KEY")


# ── TestValidateLicenseSilent ─────────────────────────────────────────────────

class TestValidateLicenseSilent(unittest.TestCase):
    """Gumroad validate: POST /v2/licenses/verify with increment_uses_count=false."""

    @patch("urllib.request.urlopen")
    def test_valid_key_returns_true(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({"success": True})
        self.assertTrue(validate_license_silent("KEY-1234"))

    @patch("urllib.request.urlopen")
    def test_revoked_key_returns_false(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({"success": False})
        self.assertFalse(validate_license_silent("KEY-1234"))

    @patch("urllib.request.urlopen")
    def test_network_error_returns_true(self, mock_urlopen):
        """Offline users get benefit of the doubt."""
        mock_urlopen.side_effect = urllib.error.URLError("network unreachable")
        self.assertTrue(validate_license_silent("KEY-1234"))

    @patch("urllib.request.urlopen")
    def test_timeout_returns_true(self, mock_urlopen):
        mock_urlopen.side_effect = socket.timeout()
        self.assertTrue(validate_license_silent("KEY-1234"))

    def test_empty_key_returns_false(self):
        self.assertFalse(validate_license_silent(""))

    def test_whitespace_key_returns_false(self):
        self.assertFalse(validate_license_silent("   "))

    def test_none_key_returns_false(self):
        self.assertFalse(validate_license_silent(None))

    @patch("urllib.request.urlopen")
    def test_http_403_returns_false(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(403)
        self.assertFalse(validate_license_silent("KEY-1234"))


# ── TestDeactivateLicense ─────────────────────────────────────────────────────

class TestDeactivateLicense(unittest.TestCase):
    """Gumroad deactivation is local-only (no API call)."""

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def test_no_license_returns_error(self):
        ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)
        self.assertIn("no active license", msg.lower())

    def test_unactivated_license_returns_error(self):
        _write_license(self.path, _DEFAULT_LICENSE.copy())
        ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)
        self.assertIn("no active license", msg.lower())

    def test_successful_deactivation_clears_file(self):
        _write_license(self.path, _activated_license())
        ok, msg = deactivate_license(self.path)
        self.assertTrue(ok)
        data = load_license(self.path)
        self.assertFalse(data["activated"])
        self.assertIsNone(data["key"])

    def test_successful_deactivation_preserves_launch_count(self):
        d = _activated_license()
        d["launch_count"] = 7
        _write_license(self.path, d)
        deactivate_license(self.path)
        self.assertEqual(load_license(self.path)["launch_count"], 7)

    def test_save_failure_returns_error(self):
        _write_license(self.path, _activated_license())
        with patch("license.save_license", return_value=False):
            ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)
        self.assertIn("could not clear", msg.lower())

    def test_deactivation_message_mentions_support(self):
        _write_license(self.path, _activated_license())
        ok, msg = deactivate_license(self.path)
        self.assertTrue(ok)
        self.assertIn("support", msg.lower())

    def test_no_network_call_made(self):
        """Gumroad deactivation must be local-only — no HTTP request."""
        _write_license(self.path, _activated_license())
        with patch("urllib.request.urlopen") as mock_urlopen:
            deactivate_license(self.path)
            mock_urlopen.assert_not_called()


# ── TestGrPost ────────────────────────────────────────────────────────────────

class TestGrPost(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_success_returns_true_and_json(self, mock_urlopen):
        payload = {"success": True, "purchase": {}}
        mock_urlopen.return_value = _make_urlopen_mock(payload)
        ok, resp = _gr_post({"license_key": "KEY", "product_id": "tts-studio"})
        self.assertTrue(ok)
        self.assertTrue(resp["success"])

    @patch("urllib.request.urlopen")
    def test_bad_json_response(self, mock_urlopen):
        resp_mock = MagicMock()
        resp_mock.read.return_value = b"not json"
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock
        ok, resp = _gr_post({"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertIn("unreadable", resp["message"].lower())

    @patch("urllib.request.urlopen")
    def test_http_error_stores_status_code(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(404, {"message": "not found"})
        ok, resp = _gr_post({"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertEqual(resp["_status_code"], 404)

    @patch("urllib.request.urlopen")
    def test_socket_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = socket.timeout("timed out")
        ok, resp = _gr_post({"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertTrue(resp.get("_network_error"))
        self.assertIn("timed out", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_ssl_error(self, mock_urlopen):
        mock_urlopen.side_effect = ssl.SSLError("cert verify failed")
        ok, resp = _gr_post({"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertTrue(resp.get("_network_error"))
        self.assertIn("secure connection", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_url_error_dns(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("[Errno -2] Name or service not known")
        ok, resp = _gr_post({"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertTrue(resp.get("_network_error"))
        self.assertIn("internet connection", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_url_error_timed_out(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timed out")
        ok, resp = _gr_post({"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertIn("timed out", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_unexpected_exception(self, mock_urlopen):
        mock_urlopen.side_effect = RuntimeError("unexpected kaboom")
        ok, resp = _gr_post({"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertIn("unexpected", resp["error"].lower())


# ── TestExtractError ──────────────────────────────────────────────────────────

class TestExtractError(unittest.TestCase):

    def test_network_error_flag_with_message(self):
        msg = _extract_error({"_network_error": True, "error": "Request timed out."})
        self.assertIn("timed out", msg.lower())

    def test_network_error_flag_no_message(self):
        msg = _extract_error({"_network_error": True})
        self.assertIn("internet", msg.lower())

    def test_gumroad_message_field(self):
        """Gumroad returns {success: false, message: '...'}"""
        msg = _extract_error({"message": "That license does not exist."})
        self.assertEqual(msg, "That license does not exist.")

    def test_message_stripped(self):
        msg = _extract_error({"message": "  Some error.  "})
        self.assertEqual(msg, "Some error.")

    def test_status_404(self):
        msg = _extract_error({"_status_code": 404})
        self.assertIn("not found", msg.lower())

    def test_status_403(self):
        msg = _extract_error({"_status_code": 403})
        self.assertIn("disabled", msg.lower())

    def test_status_429(self):
        msg = _extract_error({"_status_code": 429})
        self.assertIn("too many", msg.lower())

    def test_unknown_response_fallback(self):
        msg = _extract_error({})
        self.assertIn("check your key", msg.lower())


# ── Integration: full activation round-trip ───────────────────────────────────

class TestActivationRoundTrip(unittest.TestCase):

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    @patch("urllib.request.urlopen")
    def test_full_round_trip(self, mock_urlopen):
        # 1. Grace period
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "grace")
        self.assertEqual(remaining, 2)

        # 2. Activate
        mock_urlopen.return_value = _make_urlopen_mock({"success": True, "purchase": {}})
        ok, msg = activate_license("ROUND-TRIP-KEY", self.path)
        self.assertTrue(ok, msg)

        # 3. Startup after activation → ok
        state, _ = check_startup(self.path)
        self.assertEqual(state, "ok")

        # 4. Silent validate → True
        mock_urlopen.return_value = _make_urlopen_mock({"success": True})
        self.assertTrue(validate_license_silent("ROUND-TRIP-KEY"))

        # 5. Deactivate (local-only)
        ok, msg = deactivate_license(self.path)
        self.assertTrue(ok, msg)

        # 6. After deactivation, startup increments count again
        state, _ = check_startup(self.path)
        self.assertEqual(state, "grace")

    @patch("urllib.request.urlopen")
    def test_grace_expires_then_activate(self, mock_urlopen):
        for _ in range(GRACE_LAUNCHES + 1):
            state, _ = check_startup(self.path)
        self.assertEqual(state, "required")

        mock_urlopen.return_value = _make_urlopen_mock({"success": True, "purchase": {}})
        ok, msg = activate_license("LATE-KEY", self.path)
        self.assertTrue(ok, msg)

        state, _ = check_startup(self.path)
        self.assertEqual(state, "ok")


class TestFreemiumUsage(unittest.TestCase):
    """Tests for can_use_natural, record_natural_use, can_use_enhance, record_enhance_use, is_pro."""

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    # ── is_pro ────────────────────────────────────────────────────────────────
    def test_is_pro_false_for_unactivated(self):
        self.assertFalse(lic.is_pro(self.path))

    def test_is_pro_true_for_activated(self):
        _write_license(self.path, _activated_license())
        self.assertTrue(lic.is_pro(self.path))

    def test_is_pro_false_for_missing_file(self):
        self.assertFalse(lic.is_pro(self.path))

    # ── can_use_natural ───────────────────────────────────────────────────────
    def test_can_use_natural_true_when_no_uses(self):
        self.assertTrue(lic.can_use_natural(self.path))

    def test_can_use_natural_true_within_limit(self):
        _write_license(self.path, {**_DEFAULT_LICENSE, "natural_uses": 2})
        self.assertTrue(lic.can_use_natural(self.path))

    def test_can_use_natural_false_at_limit(self):
        _write_license(self.path, {**_DEFAULT_LICENSE, "natural_uses": 3})
        self.assertFalse(lic.can_use_natural(self.path))

    def test_can_use_natural_false_over_limit(self):
        _write_license(self.path, {**_DEFAULT_LICENSE, "natural_uses": 99})
        self.assertFalse(lic.can_use_natural(self.path))

    def test_can_use_natural_true_for_pro_regardless(self):
        d = _activated_license()
        d["natural_uses"] = 999
        _write_license(self.path, d)
        self.assertTrue(lic.can_use_natural(self.path))

    # ── natural_uses_remaining ────────────────────────────────────────────────
    def test_natural_remaining_starts_at_free_max(self):
        self.assertEqual(lic.natural_uses_remaining(self.path), lic.FREE_NATURAL_USES)

    def test_natural_remaining_decrements(self):
        _write_license(self.path, {**_DEFAULT_LICENSE, "natural_uses": 1})
        self.assertEqual(lic.natural_uses_remaining(self.path), 2)

    def test_natural_remaining_zero_at_limit(self):
        _write_license(self.path, {**_DEFAULT_LICENSE, "natural_uses": 3})
        self.assertEqual(lic.natural_uses_remaining(self.path), 0)

    def test_natural_remaining_none_for_pro(self):
        _write_license(self.path, _activated_license())
        self.assertIsNone(lic.natural_uses_remaining(self.path))

    # ── record_natural_use ────────────────────────────────────────────────────
    def test_record_natural_increments_count(self):
        lic.record_natural_use(self.path)
        data = load_license(self.path)
        self.assertEqual(data["natural_uses"], 1)

    def test_record_natural_increments_multiple_times(self):
        for _ in range(3):
            lic.record_natural_use(self.path)
        data = load_license(self.path)
        self.assertEqual(data["natural_uses"], 3)

    def test_record_natural_noop_for_pro(self):
        _write_license(self.path, _activated_license())
        lic.record_natural_use(self.path)
        data = load_license(self.path)
        # natural_uses should remain 0 (default) — pro users aren't tracked
        self.assertEqual(data.get("natural_uses", 0), 0)

    def test_record_natural_does_not_raise_on_bad_path(self):
        bad_path = "/nonexistent/path/license.json"
        try:
            lic.record_natural_use(bad_path)
        except Exception as e:
            self.fail(f"record_natural_use raised {e}")

    # ── can_use_enhance ───────────────────────────────────────────────────────
    def test_can_use_enhance_true_when_no_uses(self):
        self.assertTrue(lic.can_use_enhance(self.path))

    def test_can_use_enhance_false_at_limit(self):
        _write_license(self.path, {**_DEFAULT_LICENSE, "enhance_uses": 3})
        self.assertFalse(lic.can_use_enhance(self.path))

    def test_can_use_enhance_true_for_pro(self):
        d = _activated_license()
        d["enhance_uses"] = 999
        _write_license(self.path, d)
        self.assertTrue(lic.can_use_enhance(self.path))

    # ── enhance_uses_remaining ────────────────────────────────────────────────
    def test_enhance_remaining_starts_at_free_max(self):
        self.assertEqual(lic.enhance_uses_remaining(self.path), lic.FREE_ENHANCE_USES)

    def test_enhance_remaining_zero_at_limit(self):
        _write_license(self.path, {**_DEFAULT_LICENSE, "enhance_uses": 3})
        self.assertEqual(lic.enhance_uses_remaining(self.path), 0)

    def test_enhance_remaining_none_for_pro(self):
        _write_license(self.path, _activated_license())
        self.assertIsNone(lic.enhance_uses_remaining(self.path))

    # ── record_enhance_use ────────────────────────────────────────────────────
    def test_record_enhance_increments_count(self):
        lic.record_enhance_use(self.path)
        data = load_license(self.path)
        self.assertEqual(data["enhance_uses"], 1)

    def test_record_enhance_noop_for_pro(self):
        _write_license(self.path, _activated_license())
        lic.record_enhance_use(self.path)
        data = load_license(self.path)
        self.assertEqual(data.get("enhance_uses", 0), 0)

    def test_record_enhance_does_not_raise_on_bad_path(self):
        try:
            lic.record_enhance_use("/nonexistent/path/license.json")
        except Exception as e:
            self.fail(f"record_enhance_use raised {e}")

    # ── end-to-end freemium flow ──────────────────────────────────────────────
    def test_natural_uses_exhaust_then_blocked(self):
        for _ in range(lic.FREE_NATURAL_USES):
            self.assertTrue(lic.can_use_natural(self.path))
            lic.record_natural_use(self.path)
        self.assertFalse(lic.can_use_natural(self.path))

    def test_enhance_uses_exhaust_then_blocked(self):
        for _ in range(lic.FREE_ENHANCE_USES):
            self.assertTrue(lic.can_use_enhance(self.path))
            lic.record_enhance_use(self.path)
        self.assertFalse(lic.can_use_enhance(self.path))

    def test_default_license_has_usage_fields(self):
        self.assertIn("natural_uses", _DEFAULT_LICENSE)
        self.assertIn("enhance_uses", _DEFAULT_LICENSE)
        self.assertEqual(_DEFAULT_LICENSE["natural_uses"], 0)
        self.assertEqual(_DEFAULT_LICENSE["enhance_uses"], 0)


# ── TestDualPermalink ──────────────────────────────────────────────────────────

class TestDualPermalink(unittest.TestCase):
    """
    activate_license and validate_license_silent must try both Gumroad
    permalinks (monthly TTSStudioPro, lifetime TTSStudioProLifetime) because
    each product generates keys tied only to its own permalink.
    """

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    # ── activate_license ──────────────────────────────────────────────────────

    @patch("license._verify_license")
    def test_activate_monthly_key_succeeds_first_try(self, mock_verify):
        """Monthly key: first product succeeds → second never called."""
        mock_verify.return_value = (True, {"success": True, "purchase": {}})
        ok, msg = activate_license("MONTHLY-KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(mock_verify.call_count, 1)

    @patch("license._verify_license")
    def test_activate_lifetime_key_succeeds_on_second_product(self, mock_verify):
        """Lifetime key: monthly returns success=False, lifetime returns True."""
        mock_verify.side_effect = [
            (True, {"success": False, "message": "That license does not exist."}),
            (True, {"success": True, "purchase": {}}),
        ]
        ok, msg = activate_license("LIFETIME-KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(mock_verify.call_count, 2)
        self.assertEqual(load_license(self.path)["key"], "LIFETIME-KEY")

    @patch("license._verify_license")
    def test_activate_both_products_fail_returns_error(self, mock_verify):
        """Both products reject key → error returned, file not touched."""
        mock_verify.side_effect = [
            (True, {"success": False, "message": "That license does not exist."}),
            (True, {"success": False, "message": "That license does not exist."}),
        ]
        ok, msg = activate_license("BAD-KEY", self.path)
        self.assertFalse(ok)
        self.assertEqual(mock_verify.call_count, 2)
        self.assertIn("does not exist", msg.lower())
        self.assertFalse(load_license(self.path)["activated"])

    @patch("license._verify_license")
    def test_activate_network_error_first_still_tries_second(self, mock_verify):
        """Network error on first product still attempts second."""
        mock_verify.side_effect = [
            (False, {"_network_error": True, "error": "Could not reach server."}),
            (True, {"success": True, "purchase": {}}),
        ]
        ok, msg = activate_license("KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(mock_verify.call_count, 2)

    @patch("license._verify_license")
    def test_activate_both_network_errors_returns_friendly_message(self, mock_verify):
        """Both products hit network error → friendly network error message returned."""
        err = {"_network_error": True, "error": "Could not reach activation server. Check your internet connection."}
        mock_verify.side_effect = [(False, err), (False, err)]
        ok, msg = activate_license("KEY", self.path)
        self.assertFalse(ok)
        self.assertEqual(mock_verify.call_count, 2)
        self.assertIn("internet connection", msg.lower())

    @patch("license._verify_license")
    def test_activate_tries_both_permalinks(self, mock_verify):
        """Both _ALL_PERMALINKS values must be tried."""
        mock_verify.side_effect = [
            (True, {"success": False, "message": "Not found."}),
            (True, {"success": True, "purchase": {}}),
        ]
        activate_license("KEY", self.path)
        permalinks_used = [call[0][0] for call in mock_verify.call_args_list]
        self.assertIn(lic.PRODUCT_PERMALINK, permalinks_used)
        self.assertIn(lic.PRODUCT_PERMALINK_LIFETIME, permalinks_used)

    @patch("license._verify_license")
    def test_activate_first_fails_tries_second(self, mock_verify):
        """First product fails → tries second."""
        mock_verify.side_effect = [
            (False, {"_status_code": 404}),
            (True, {"success": True, "purchase": {}}),
        ]
        ok, msg = activate_license("KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(mock_verify.call_count, 2)

    # ── validate_license_silent ───────────────────────────────────────────────

    @patch("license._verify_license")
    def test_silent_valid_on_first_permalink_returns_true(self, mock_verify):
        """Key valid on first product → True, second not attempted."""
        mock_verify.return_value = (True, {"success": True})
        result = validate_license_silent("KEY-1234")
        self.assertTrue(result)
        self.assertEqual(mock_verify.call_count, 1)

    @patch("license._verify_license")
    def test_silent_valid_on_second_permalink_returns_true(self, mock_verify):
        """Key valid only on lifetime → True after two calls."""
        mock_verify.side_effect = [
            (True, {"success": False}),
            (True, {"success": True}),
        ]
        result = validate_license_silent("KEY-1234")
        self.assertTrue(result)
        self.assertEqual(mock_verify.call_count, 2)

    @patch("license._verify_license")
    def test_silent_both_invalid_returns_false(self, mock_verify):
        """Key rejected by both products → False."""
        mock_verify.return_value = (True, {"success": False})
        result = validate_license_silent("KEY-1234")
        self.assertFalse(result)
        self.assertEqual(mock_verify.call_count, 2)

    @patch("license._verify_license")
    def test_silent_network_error_returns_true_immediately(self, mock_verify):
        """Network error → benefit of the doubt, True without trying second."""
        mock_verify.return_value = (False, {"_network_error": True, "error": "No network."})
        result = validate_license_silent("KEY-1234")
        self.assertTrue(result)
        self.assertEqual(mock_verify.call_count, 1)

    @patch("license._verify_license")
    def test_silent_increment_uses_count_is_false(self, mock_verify):
        """Silent validation must pass increment='false'."""
        mock_verify.return_value = (True, {"success": True})
        validate_license_silent("KEY-1234")
        self.assertEqual(mock_verify.call_args[0][2], "false")


# ── TestDeactivatePreservesFreemium ───────────────────────────────────────────

class TestDeactivatePreservesFreemium(unittest.TestCase):
    """
    Deactivating Pro must NOT reset freemium counters.
    Otherwise: user exhausts 3 free Natural uses → buys Pro → deactivates → gets 3 more free.
    """

    def setUp(self):
        self.td   = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def test_deactivate_preserves_natural_uses(self):
        d = _activated_license()
        d["natural_uses"] = 3
        _write_license(self.path, d)
        ok, _ = deactivate_license(self.path)
        self.assertTrue(ok)
        self.assertEqual(load_license(self.path)["natural_uses"], 3)

    def test_deactivate_preserves_enhance_uses(self):
        d = _activated_license()
        d["enhance_uses"] = 3
        _write_license(self.path, d)
        ok, _ = deactivate_license(self.path)
        self.assertTrue(ok)
        self.assertEqual(load_license(self.path)["enhance_uses"], 3)

    def test_deactivate_still_blocks_natural_after_limit(self):
        """After deactivation, exhausted freemium users remain blocked."""
        d = _activated_license()
        d["natural_uses"] = lic.FREE_NATURAL_USES
        _write_license(self.path, d)
        deactivate_license(self.path)
        self.assertFalse(lic.can_use_natural(self.path))

    def test_deactivate_still_blocks_enhance_after_limit(self):
        d = _activated_license()
        d["enhance_uses"] = lic.FREE_ENHANCE_USES
        _write_license(self.path, d)
        deactivate_license(self.path)
        self.assertFalse(lic.can_use_enhance(self.path))

    def test_deactivate_zero_uses_stays_zero(self):
        """Pro user who never used Natural/Enhance deactivates — counters still 0."""
        _write_license(self.path, _activated_license())
        deactivate_license(self.path)
        data = load_license(self.path)
        self.assertEqual(data["natural_uses"], 0)
        self.assertEqual(data["enhance_uses"], 0)


# ── TestVerifyLicense — product_id discovery ─────────────────────────────────

class TestVerifyLicense(unittest.TestCase):
    """Tests for _verify_license and product_id auto-discovery."""

    def setUp(self):
        # Clear cached product IDs between tests
        lic._PRODUCT_IDS.clear()

    def tearDown(self):
        lic._PRODUCT_IDS.clear()

    # ── regex ────────────────────────────────────────────────────────────────

    def test_regex_extracts_product_id_with_special_chars(self):
        """Gumroad product IDs contain -, +, = — regex must capture all."""
        msg = ("The 'product_id' parameter is required to verify the license "
               "for this product. Please set 'product_id' to "
               "'cwSJcg1w4rgcNO-T6K732w==' in the request.")
        m = _PRODUCT_ID_RE.search(msg)
        self.assertIsNotNone(m, "Regex did not match")
        self.assertEqual(m.group(1), "cwSJcg1w4rgcNO-T6K732w==")

    def test_regex_extracts_simple_product_id(self):
        msg = "Please set 'product_id' to 'abc123' in the request."
        m = _PRODUCT_ID_RE.search(msg)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "abc123")

    def test_regex_no_match_on_unrelated_message(self):
        msg = "That license does not exist for the provided product."
        m = _PRODUCT_ID_RE.search(msg)
        self.assertIsNone(m)

    # ── hardcoded product_id succeeds immediately ────────────────────────────

    @patch("license._gr_post")
    def test_hardcoded_id_succeeds_first_try(self, mock_post):
        """When hardcoded product_id works, only one API call is made."""
        mock_post.return_value = (True, {"success": True, "purchase": {}})
        ok, resp = _verify_license("TTSStudioProLifetime", "KEY", "false")
        self.assertTrue(ok)
        self.assertTrue(resp["success"])
        self.assertEqual(mock_post.call_count, 1)
        # Verify it used the hardcoded product_id
        call_params = mock_post.call_args[0][0]
        self.assertEqual(call_params["product_id"],
                         lic._GUMROAD_PRODUCT_IDS["TTSStudioProLifetime"])

    # ── hardcoded fails, permalink fallback works ────────────────────────────

    @patch("license._gr_post")
    def test_permalink_fallback_when_hardcoded_fails(self, mock_post):
        """Hardcoded ID fails → falls back to product_permalink."""
        mock_post.side_effect = [
            (True, {"success": False, "message": "Not found."}),  # hardcoded
            (True, {"success": True, "purchase": {}}),             # permalink
        ]
        ok, resp = _verify_license("TTSStudioPro", "KEY", "false")
        self.assertTrue(ok)
        self.assertEqual(mock_post.call_count, 2)
        call_params = mock_post.call_args_list[1][0][0]
        self.assertIn("product_permalink", call_params)

    # ── product_id discovery from error message ──────────────────────────────

    @patch("license._gr_post")
    def test_discovers_product_id_from_error_message(self, mock_post):
        """When permalink returns product_id hint, extract it and retry."""
        mock_post.side_effect = [
            # hardcoded ID fails
            (True, {"success": False, "message": "Not found."}),
            # permalink fails with product_id hint
            (False, {"success": False,
                     "message": "Please set 'product_id' to 'abc123XYZ==' in the request."}),
            # retry with discovered ID succeeds
            (True, {"success": True, "purchase": {}}),
        ]
        ok, resp = _verify_license("TTSStudioPro", "KEY", "true")
        self.assertTrue(ok)
        self.assertEqual(mock_post.call_count, 3)
        # Discovered ID should be cached
        self.assertEqual(lic._PRODUCT_IDS["TTSStudioPro"], "abc123XYZ==")
        # Third call should use discovered ID
        call_params = mock_post.call_args_list[2][0][0]
        self.assertEqual(call_params["product_id"], "abc123XYZ==")

    @patch("license._gr_post")
    def test_cached_id_used_on_subsequent_calls(self, mock_post):
        """Once a product_id is discovered, it's used directly next time."""
        lic._PRODUCT_IDS["TTSStudioPro"] = "cached-id-123"
        mock_post.side_effect = [
            (True, {"success": False}),  # hardcoded fails
            (True, {"success": True}),   # cached works
        ]
        ok, resp = _verify_license("TTSStudioPro", "KEY", "false")
        self.assertTrue(ok)
        call_params = mock_post.call_args_list[1][0][0]
        self.assertEqual(call_params["product_id"], "cached-id-123")

    # ── all attempts fail ────────────────────────────────────────────────────

    @patch("license._gr_post")
    def test_all_attempts_fail_returns_last_error(self, mock_post):
        """When everything fails (no product_id hint), return the error."""
        mock_post.return_value = (True, {
            "success": False,
            "message": "That license does not exist for the provided product.",
        })
        ok, resp = _verify_license("TTSStudioPro", "BAD-KEY", "false")
        self.assertFalse(resp["success"])
        self.assertIn("does not exist", resp["message"])

    # ── network error propagates ─────────────────────────────────────────────

    @patch("license._gr_post")
    def test_network_error_propagates(self, mock_post):
        """Network error on hardcoded ID check propagates up."""
        mock_post.return_value = (False, {"_network_error": True, "error": "timeout"})
        ok, resp = _verify_license("TTSStudioPro", "KEY", "false")
        self.assertFalse(ok)
        # On network error, hardcoded fails, permalink fails, no hint → return error
        self.assertTrue(resp.get("_network_error"))


# ── TestMachineActivationLimit ────────────────────────────────────────────────

class TestMachineActivationLimit(unittest.TestCase):
    """Tests for machine-based activation limit (MAX_MACHINES=2)."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    @patch("license._verify_license")
    def test_first_machine_activates(self, mock_verify):
        """Brand new key on first machine → uses=1, success."""
        mock_verify.return_value = (True, {"success": True, "purchase": {}, "uses": 1})
        ok, msg = activate_license("NEW-KEY", self.path)
        self.assertTrue(ok, msg)
        data = load_license(self.path)
        self.assertTrue(data["activated"])
        self.assertIsNotNone(data["machine_id"])
        # New machine → should have called with increment="true"
        self.assertEqual(mock_verify.call_args[0][2], "true")

    @patch("license._verify_license")
    def test_same_machine_reinstall_no_increment(self, mock_verify):
        """Reinstall on same machine → increment=false, doesn't burn activation."""
        # Pre-seed with matching machine_id
        _write_license(self.path, {
            "key": "MY-KEY",
            "machine_id": lic._get_machine_id(),
        })
        mock_verify.return_value = (True, {"success": True, "purchase": {}, "uses": 1})
        ok, msg = activate_license("MY-KEY", self.path)
        self.assertTrue(ok, msg)
        # Same machine → increment should be "false"
        self.assertEqual(mock_verify.call_args[0][2], "false")

    @patch("license._verify_license")
    def test_second_machine_activates(self, mock_verify):
        """Second machine → uses=2, still under limit, success."""
        mock_verify.return_value = (True, {"success": True, "purchase": {}, "uses": 2})
        ok, msg = activate_license("SHARED-KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertTrue(load_license(self.path)["activated"])

    @patch("license._verify_license")
    def test_third_machine_rejected(self, mock_verify):
        """Third machine → uses=3, over MAX_MACHINES=2, rejected."""
        mock_verify.return_value = (True, {"success": True, "purchase": {}, "uses": 3})
        ok, msg = activate_license("SHARED-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("already active on 2 machines", msg)
        self.assertFalse(load_license(self.path).get("activated", False))

    @patch("license._verify_license")
    def test_same_machine_bypasses_limit_check(self, mock_verify):
        """Same machine reinstall with uses=5 still works (limit check skipped)."""
        _write_license(self.path, {
            "key": "MY-KEY",
            "machine_id": lic._get_machine_id(),
        })
        mock_verify.return_value = (True, {"success": True, "purchase": {}, "uses": 5})
        ok, msg = activate_license("MY-KEY", self.path)
        self.assertTrue(ok, msg)  # same machine → limit not checked

    @patch("license._verify_license")
    def test_different_key_same_machine_increments(self, mock_verify):
        """Activating a DIFFERENT key on same machine → treated as new, increments."""
        _write_license(self.path, {
            "key": "OLD-KEY",
            "machine_id": lic._get_machine_id(),
        })
        mock_verify.return_value = (True, {"success": True, "purchase": {}, "uses": 1})
        ok, msg = activate_license("NEW-KEY", self.path)
        self.assertTrue(ok, msg)
        # Different key → increment should be "true"
        self.assertEqual(mock_verify.call_args[0][2], "true")

    def test_machine_id_is_stable(self):
        """_get_machine_id returns the same value across calls."""
        id1 = lic._get_machine_id()
        id2 = lic._get_machine_id()
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 16)  # 16-char hex hash prefix


# ── TestLiveGumroadAPI — real API integration (skipped without network) ──────

class TestLiveGumroadAPI(unittest.TestCase):
    """Integration tests that hit the real Gumroad API.

    These verify the actual product_id, regex extraction, and end-to-end flow.
    Skipped if the network is unreachable.
    """

    TEST_KEY = os.environ.get("TTS_TEST_LICENSE_KEY", "")

    @classmethod
    def setUpClass(cls):
        """Skip if no test key or API unreachable."""
        if not cls.TEST_KEY:
            raise unittest.SkipTest("TTS_TEST_LICENSE_KEY not set — skipping live tests")
        try:
            import urllib.request
            urllib.request.urlopen("https://api.gumroad.com", timeout=5)
        except urllib.error.HTTPError:
            pass  # 404 is fine — server is reachable
        except Exception:
            raise unittest.SkipTest("Gumroad API unreachable — skipping live tests")

    def setUp(self):
        lic._PRODUCT_IDS.clear()

    def tearDown(self):
        lic._PRODUCT_IDS.clear()

    def _require_active_key(self):
        """Skip test if the test key has been revoked/disabled on Gumroad."""
        ok, resp = _gr_post({
            "product_id": lic._GUMROAD_PRODUCT_IDS["TTSStudioProLifetime"],
            "license_key": self.TEST_KEY,
            "increment_uses_count": "false",
        })
        if not (ok and resp.get("success")):
            self.skipTest("Test key is revoked/disabled on Gumroad — re-enable to run live tests")
        return resp

    def test_hardcoded_product_id_validates_lifetime_key(self):
        """Hardcoded lifetime product_id successfully validates a real key."""
        resp = self._require_active_key()
        self.assertEqual(resp["purchase"]["product_name"], "TTS Studio Pro Lifetime")

    def test_product_permalink_returns_product_id_hint(self):
        """Sending product_permalink returns error with real product_id."""
        ok, resp = _gr_post({
            "product_permalink": "TTSStudioProLifetime",
            "license_key": self.TEST_KEY,
            "increment_uses_count": "false",
        })
        # Should fail with a message containing the product_id
        msg = resp.get("message", "")
        m = _PRODUCT_ID_RE.search(msg)
        self.assertIsNotNone(m, f"No product_id in error: {msg}")
        self.assertEqual(m.group(1), lic._GUMROAD_PRODUCT_IDS["TTSStudioProLifetime"])

    def test_verify_license_end_to_end(self):
        """_verify_license finds the right product and validates the key."""
        self._require_active_key()
        ok, resp = _verify_license("TTSStudioProLifetime", self.TEST_KEY, "false")
        self.assertTrue(ok)
        self.assertTrue(resp.get("success"))

    def test_verify_license_bad_key_returns_failure(self):
        """Invalid key returns success=False, not an exception."""
        ok, resp = _verify_license("TTSStudioProLifetime", "FAKE-KEY-1234", "false")
        self.assertFalse(resp.get("success", True))

    def test_full_activate_flow(self):
        """Full activate_license with a real key writes license.json."""
        self._require_active_key()
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "test_live_lic.json")
        try:
            # Pre-seed with current machine_id so this counts as a reinstall
            # (doesn't increment uses — avoids burning test activations)
            _write_license(tmp, {
                "key": self.TEST_KEY,
                "machine_id": lic._get_machine_id(),
            })
            success, msg = activate_license(self.TEST_KEY, path=tmp)
            self.assertTrue(success, f"Activation failed: {msg}")
            data = load_license(tmp)
            self.assertTrue(data["activated"])
            self.assertEqual(data["key"], self.TEST_KEY)
            self.assertEqual(data["machine_id"], lic._get_machine_id())
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
