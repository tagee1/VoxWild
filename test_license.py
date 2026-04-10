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
        ok, resp = _gr_post({"license_key": "KEY", "product_permalink": "tts-studio"})
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

    @patch("license._gr_post")
    def test_activate_monthly_key_succeeds_first_try(self, mock_post):
        """Monthly key: first permalink succeeds → second never called."""
        mock_post.return_value = (True, {"success": True, "purchase": {}})
        ok, msg = activate_license("MONTHLY-KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(mock_post.call_count, 1)

    @patch("license._gr_post")
    def test_activate_lifetime_key_succeeds_on_second_permalink(self, mock_post):
        """Lifetime key: monthly returns success=False, lifetime returns True."""
        mock_post.side_effect = [
            (True, {"success": False, "message": "That license does not exist."}),
            (True, {"success": True, "purchase": {}}),
        ]
        ok, msg = activate_license("LIFETIME-KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(load_license(self.path)["key"], "LIFETIME-KEY")

    @patch("license._gr_post")
    def test_activate_both_permalinks_fail_returns_error(self, mock_post):
        """Both permalinks reject key → error returned, file not touched."""
        mock_post.side_effect = [
            (True, {"success": False, "message": "That license does not exist."}),
            (True, {"success": False, "message": "That license does not exist."}),
        ]
        ok, msg = activate_license("BAD-KEY", self.path)
        self.assertFalse(ok)
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("does not exist", msg.lower())
        self.assertFalse(load_license(self.path)["activated"])

    @patch("license._gr_post")
    def test_activate_network_error_first_still_tries_second(self, mock_post):
        """Network error on monthly permalink still attempts lifetime permalink."""
        mock_post.side_effect = [
            (False, {"_network_error": True, "error": "Could not reach server."}),
            (True, {"success": True, "purchase": {}}),
        ]
        ok, msg = activate_license("KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(mock_post.call_count, 2)

    @patch("license._gr_post")
    def test_activate_both_network_errors_returns_friendly_message(self, mock_post):
        """Both permalinks hit network error → friendly network error message returned."""
        err = {"_network_error": True, "error": "Could not reach activation server. Check your internet connection."}
        mock_post.side_effect = [(False, err), (False, err)]
        ok, msg = activate_license("KEY", self.path)
        self.assertFalse(ok)
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("internet connection", msg.lower())

    @patch("license._gr_post")
    def test_activate_uses_both_permalink_constants(self, mock_post):
        """Both _ALL_PERMALINKS values must be passed to _gr_post."""
        mock_post.side_effect = [
            (True, {"success": False, "message": "Not found."}),
            (True, {"success": True, "purchase": {}}),
        ]
        activate_license("KEY", self.path)
        permalinks_used = [call[0][0]["product_permalink"] for call in mock_post.call_args_list]
        self.assertIn(lic.PRODUCT_PERMALINK, permalinks_used)
        self.assertIn(lic.PRODUCT_PERMALINK_LIFETIME, permalinks_used)

    @patch("license._gr_post")
    def test_activate_http_404_on_first_tries_second(self, mock_post):
        """HTTP 404 on monthly (wrong product) → tries lifetime permalink."""
        mock_post.side_effect = [
            (False, {"_status_code": 404}),
            (True, {"success": True, "purchase": {}}),
        ]
        ok, msg = activate_license("KEY", self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(mock_post.call_count, 2)

    # ── validate_license_silent ───────────────────────────────────────────────

    @patch("license._gr_post")
    def test_silent_valid_on_first_permalink_returns_true(self, mock_post):
        """Key valid on monthly → True, second permalink not attempted."""
        mock_post.return_value = (True, {"success": True})
        result = validate_license_silent("KEY-1234")
        self.assertTrue(result)
        self.assertEqual(mock_post.call_count, 1)

    @patch("license._gr_post")
    def test_silent_valid_on_second_permalink_returns_true(self, mock_post):
        """Key valid only on lifetime permalink → True after two calls."""
        mock_post.side_effect = [
            (True, {"success": False}),
            (True, {"success": True}),
        ]
        result = validate_license_silent("KEY-1234")
        self.assertTrue(result)
        self.assertEqual(mock_post.call_count, 2)

    @patch("license._gr_post")
    def test_silent_both_invalid_returns_false(self, mock_post):
        """Key rejected by both permalinks → False."""
        mock_post.return_value = (True, {"success": False})
        result = validate_license_silent("KEY-1234")
        self.assertFalse(result)
        self.assertEqual(mock_post.call_count, 2)

    @patch("license._gr_post")
    def test_silent_network_error_returns_true_immediately(self, mock_post):
        """Network error → benefit of the doubt, True returned without trying second permalink."""
        mock_post.return_value = (False, {"_network_error": True, "error": "No network."})
        result = validate_license_silent("KEY-1234")
        self.assertTrue(result)
        self.assertEqual(mock_post.call_count, 1)

    @patch("license._gr_post")
    def test_silent_increment_uses_count_is_false(self, mock_post):
        """Silent validation must never increment use count."""
        mock_post.return_value = (True, {"success": True})
        validate_license_silent("KEY-1234")
        call_params = mock_post.call_args[0][0]
        self.assertEqual(call_params.get("increment_uses_count"), "false")


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
