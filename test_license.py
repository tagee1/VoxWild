"""
test_license.py — Comprehensive tests for license.py

Coverage:
  - load_license: missing file, corrupt JSON, non-dict, forward-compat keys
  - save_license: success, disk error, dir creation
  - check_startup: grace math (3 launches free), activated, required state
  - activate_license: empty key, success, no instance_id, save fails, verify fails, API errors
  - validate_license_silent: whitespace keys, valid, revoked, network down → True
  - deactivate_license: no license, success, save fails, API errors
  - _ls_post: timeout, SSL, URLError (DNS/refused), HTTPError, bad JSON response
  - _extract_error: all branches (network, plain fields, errors array, status codes)
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
from unittest.mock import MagicMock, patch, mock_open

import license as lic
from license import (
    GRACE_LAUNCHES,
    _DEFAULT_LICENSE,
    _extract_error,
    activate_license,
    check_startup,
    deactivate_license,
    load_license,
    save_license,
    validate_license_silent,
    _ls_post,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_license(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _activated_license(key="ABCD-1234", instance_id="inst-xyz"):
    d = _DEFAULT_LICENSE.copy()
    d.update({"key": key, "instance_id": instance_id, "activated": True,
               "activation_date": "2024-01-01T00:00:00"})
    return d


def _make_urlopen_mock(payload, status=200):
    """Return a context-manager mock for urllib.request.urlopen that yields payload."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(code, body=None):
    body_bytes = json.dumps(body or {}).encode() if body is not None else b""
    return urllib.error.HTTPError(
        url="https://example.com",
        code=code,
        msg="Error",
        hdrs=None,
        fp=io.BytesIO(body_bytes),
    )


# ── TestLoadLicense ───────────────────────────────────────────────────────────

class TestLoadLicense(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def test_missing_file_returns_default(self):
        result = load_license(self.path)
        self.assertEqual(result, _DEFAULT_LICENSE)

    def test_valid_file_returns_data(self):
        data = _activated_license()
        _write_license(self.path, data)
        result = load_license(self.path)
        self.assertEqual(result["key"], "ABCD-1234")
        self.assertTrue(result["activated"])

    def test_corrupt_json_returns_default(self):
        with open(self.path, "w") as f:
            f.write("{not valid json!!!")
        result = load_license(self.path)
        self.assertEqual(result, _DEFAULT_LICENSE)

    def test_non_dict_json_returns_default(self):
        with open(self.path, "w") as f:
            json.dump([1, 2, 3], f)
        result = load_license(self.path)
        self.assertEqual(result, _DEFAULT_LICENSE)

    def test_null_json_returns_default(self):
        with open(self.path, "w") as f:
            f.write("null")
        result = load_license(self.path)
        self.assertEqual(result, _DEFAULT_LICENSE)

    def test_forward_compat_missing_keys_added(self):
        """Partial file (old version) gets missing keys filled in."""
        partial = {"key": "OLD-KEY", "activated": False}
        _write_license(self.path, partial)
        result = load_license(self.path)
        for k in _DEFAULT_LICENSE:
            self.assertIn(k, result, f"Missing key: {k}")
        self.assertEqual(result["key"], "OLD-KEY")
        self.assertIsNone(result["instance_id"])
        self.assertEqual(result["launch_count"], 0)

    def test_empty_file_returns_default(self):
        open(self.path, "w").close()
        result = load_license(self.path)
        self.assertEqual(result, _DEFAULT_LICENSE)

    def test_all_default_keys_present(self):
        result = load_license(self.path)
        for k in _DEFAULT_LICENSE:
            self.assertIn(k, result)

    def test_does_not_raise_on_read_error(self):
        """Even if open() raises OSError, load_license never raises."""
        with patch("builtins.open", side_effect=OSError("disk error")):
            with patch("os.path.exists", return_value=True):
                result = load_license(self.path)
        self.assertEqual(result, _DEFAULT_LICENSE)


# ── TestSaveLicense ───────────────────────────────────────────────────────────

class TestSaveLicense(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def test_save_and_reload(self):
        data = _activated_license()
        ok = save_license(data, self.path)
        self.assertTrue(ok)
        result = load_license(self.path)
        self.assertEqual(result["key"], "ABCD-1234")

    def test_returns_true_on_success(self):
        ok = save_license(_DEFAULT_LICENSE.copy(), self.path)
        self.assertTrue(ok)

    def test_returns_false_on_write_error(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            ok = save_license(_DEFAULT_LICENSE.copy(), self.path)
        self.assertFalse(ok)

    def test_creates_parent_directory(self):
        nested = os.path.join(self.td.name, "deep", "nested", "license.json")
        ok = save_license(_DEFAULT_LICENSE.copy(), nested)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(nested))

    def test_does_not_raise_on_makedirs_error(self):
        """If makedirs fails, save_license still attempts the write."""
        with patch("os.makedirs", side_effect=OSError("perm denied")):
            # write may or may not succeed — just must not raise
            try:
                save_license(_DEFAULT_LICENSE.copy(), self.path)
            except Exception as e:
                self.fail(f"save_license raised: {e}")

    def test_saved_file_is_valid_json(self):
        save_license(_activated_license(), self.path)
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)


# ── TestCheckStartup ──────────────────────────────────────────────────────────

class TestCheckStartup(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
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

    def test_second_launch_grace_remaining_decrements(self):
        check_startup(self.path)
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "grace")
        self.assertEqual(remaining, GRACE_LAUNCHES - 2)

    def test_last_free_launch(self):
        """Launch GRACE_LAUNCHES (3rd) is still free with 0 remaining."""
        for _ in range(GRACE_LAUNCHES - 1):
            check_startup(self.path)
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "grace")
        self.assertEqual(remaining, 0)

    def test_launch_after_grace_is_required(self):
        """Launch GRACE_LAUNCHES+1 (4th) must be blocked."""
        for _ in range(GRACE_LAUNCHES):
            check_startup(self.path)
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "required")
        self.assertEqual(remaining, 0)

    def test_launch_count_persists(self):
        check_startup(self.path)
        check_startup(self.path)
        data = load_license(self.path)
        self.assertEqual(data["launch_count"], 2)

    def test_activated_does_not_increment_count(self):
        d = _activated_license()
        d["launch_count"] = 5
        _write_license(self.path, d)
        check_startup(self.path)
        data = load_license(self.path)
        self.assertEqual(data["launch_count"], 5)

    def test_missing_file_treated_as_launch_1(self):
        """No file → first launch → grace."""
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "grace")

    def test_grace_launches_constant_is_3(self):
        self.assertEqual(GRACE_LAUNCHES, 3)

    def test_exactly_grace_launches_free_launches(self):
        """Verify that exactly GRACE_LAUNCHES are free and no more."""
        free = 0
        for _ in range(GRACE_LAUNCHES + 2):
            state, _ = check_startup(self.path)
            if state == "grace":
                free += 1
        self.assertEqual(free, GRACE_LAUNCHES)


# ── TestActivateLicense ───────────────────────────────────────────────────────

class TestActivateLicense(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def _mock_success(self, key="TEST-KEY-1234", instance_id="inst-abc"):
        return _make_urlopen_mock({
            "activated": True,
            "instance": {"id": instance_id},
            "license_key": {"key": key, "status": "active"},
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
        mock_urlopen.return_value = self._mock_success("GOOD-KEY", "inst-1")
        ok, msg = activate_license("GOOD-KEY", self.path)
        self.assertTrue(ok)
        self.assertIn("activated successfully", msg.lower())

    @patch("urllib.request.urlopen")
    def test_activation_persists_to_file(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY", "inst-1")
        activate_license("GOOD-KEY", self.path)
        data = load_license(self.path)
        self.assertTrue(data["activated"])
        self.assertEqual(data["key"], "GOOD-KEY")
        self.assertEqual(data["instance_id"], "inst-1")

    @patch("urllib.request.urlopen")
    def test_activation_date_set(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY", "inst-1")
        activate_license("GOOD-KEY", self.path)
        data = load_license(self.path)
        self.assertIsNotNone(data["activation_date"])

    @patch("urllib.request.urlopen")
    def test_no_instance_id_returns_error(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({
            "activated": True,
            "instance": {},
        })
        ok, msg = activate_license("GOOD-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("instance id", msg.lower())

    @patch("urllib.request.urlopen")
    def test_no_instance_field_returns_error(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({
            "activated": True,
        })
        ok, msg = activate_license("GOOD-KEY", self.path)
        self.assertFalse(ok)

    @patch("urllib.request.urlopen")
    def test_save_failure_returns_error(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY", "inst-1")
        with patch("license.save_license", return_value=False):
            ok, msg = activate_license("GOOD-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("could not be saved", msg.lower())

    @patch("urllib.request.urlopen")
    def test_verify_failure_returns_error(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_success("GOOD-KEY", "inst-1")
        # Use side_effect so each call gets a fresh copy (avoids aliasing from lic.update())
        # First call (load existing data) → fresh default; second call (verify) → stale default
        with patch("license.save_license", return_value=True):
            with patch("license.load_license", side_effect=[_DEFAULT_LICENSE.copy(),
                                                            _DEFAULT_LICENSE.copy()]):
                ok, msg = activate_license("GOOD-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("verification failed", msg.lower())

    @patch("urllib.request.urlopen")
    def test_api_returns_activated_false(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({
            "activated": False,
            "error": "License key not valid.",
        })
        ok, msg = activate_license("BAD-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("not valid", msg.lower())

    @patch("urllib.request.urlopen")
    def test_http_404_returns_friendly_message(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(404)
        ok, msg = activate_license("MISSING-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("not found", msg.lower())

    @patch("urllib.request.urlopen")
    def test_http_422_activation_limit(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(422)
        ok, msg = activate_license("MAXED-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("activation limit", msg.lower())

    @patch("urllib.request.urlopen")
    def test_http_403_revoked_key(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(403)
        ok, msg = activate_license("REVOKED-KEY", self.path)
        self.assertFalse(ok)
        self.assertIn("disabled or revoked", msg.lower())

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
        """Leading/trailing whitespace in key must be stripped."""
        mock_urlopen.return_value = self._mock_success("CLEAN-KEY", "inst-1")
        ok, msg = activate_license("  CLEAN-KEY  ", self.path)
        self.assertTrue(ok)
        data = load_license(self.path)
        self.assertEqual(data["key"], "CLEAN-KEY")


# ── TestValidateLicenseSilent ─────────────────────────────────────────────────

class TestValidateLicenseSilent(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_valid_key_returns_true(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({"valid": True})
        result = validate_license_silent("KEY-1234", "inst-abc")
        self.assertTrue(result)

    @patch("urllib.request.urlopen")
    def test_revoked_key_returns_false(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({"valid": False})
        result = validate_license_silent("KEY-1234", "inst-abc")
        self.assertFalse(result)

    @patch("urllib.request.urlopen")
    def test_network_error_returns_true(self, mock_urlopen):
        """Offline users get benefit of the doubt."""
        mock_urlopen.side_effect = urllib.error.URLError("network unreachable")
        result = validate_license_silent("KEY-1234", "inst-abc")
        self.assertTrue(result)

    @patch("urllib.request.urlopen")
    def test_timeout_returns_true(self, mock_urlopen):
        mock_urlopen.side_effect = socket.timeout()
        result = validate_license_silent("KEY-1234", "inst-abc")
        self.assertTrue(result)

    def test_empty_key_returns_false(self):
        result = validate_license_silent("", "inst-abc")
        self.assertFalse(result)

    def test_empty_instance_id_returns_false(self):
        result = validate_license_silent("KEY-1234", "")
        self.assertFalse(result)

    def test_whitespace_key_returns_false(self):
        result = validate_license_silent("   ", "inst-abc")
        self.assertFalse(result)

    def test_none_key_returns_false(self):
        result = validate_license_silent(None, "inst-abc")
        self.assertFalse(result)

    def test_none_instance_id_returns_false(self):
        result = validate_license_silent("KEY-1234", None)
        self.assertFalse(result)

    @patch("urllib.request.urlopen")
    def test_http_403_returns_false(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(403)
        result = validate_license_silent("KEY-1234", "inst-abc")
        self.assertFalse(result)


# ── TestDeactivateLicense ─────────────────────────────────────────────────────

class TestDeactivateLicense(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "license.json")

    def tearDown(self):
        self.td.cleanup()

    def test_no_license_returns_error(self):
        ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)
        self.assertIn("no active license", msg.lower())

    def test_missing_key_returns_error(self):
        d = _DEFAULT_LICENSE.copy()
        d["instance_id"] = "inst-1"
        _write_license(self.path, d)
        ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)
        self.assertIn("no active license", msg.lower())

    @patch("urllib.request.urlopen")
    def test_successful_deactivation_clears_file(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({"deactivated": True})
        _write_license(self.path, _activated_license())
        ok, msg = deactivate_license(self.path)
        self.assertTrue(ok)
        data = load_license(self.path)
        self.assertFalse(data["activated"])
        self.assertIsNone(data["key"])
        self.assertIsNone(data["instance_id"])

    @patch("urllib.request.urlopen")
    def test_successful_deactivation_preserves_launch_count(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({"deactivated": True})
        d = _activated_license()
        d["launch_count"] = 7
        _write_license(self.path, d)
        deactivate_license(self.path)
        data = load_license(self.path)
        self.assertEqual(data["launch_count"], 7)

    @patch("urllib.request.urlopen")
    def test_api_returns_deactivated_false(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({
            "deactivated": False,
            "error": "Already deactivated.",
        })
        _write_license(self.path, _activated_license())
        ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)

    @patch("urllib.request.urlopen")
    def test_save_failure_after_deactivation(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_mock({"deactivated": True})
        _write_license(self.path, _activated_license())
        with patch("license.save_license", return_value=False):
            ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)
        self.assertIn("could not be saved", msg.lower())

    @patch("urllib.request.urlopen")
    def test_network_error_returns_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("network unreachable")
        _write_license(self.path, _activated_license())
        ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)
        # Generic URLError falls through to "Network error: <reason>" message
        self.assertIn("network error", msg.lower())

    @patch("urllib.request.urlopen")
    def test_http_404_returns_friendly_message(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(404)
        _write_license(self.path, _activated_license())
        ok, msg = deactivate_license(self.path)
        self.assertFalse(ok)


# ── TestLsPost ────────────────────────────────────────────────────────────────

class TestLsPost(unittest.TestCase):

    URL = "https://api.lemonsqueezy.com/v1/licenses/activate"

    @patch("urllib.request.urlopen")
    def test_success_returns_true_and_json(self, mock_urlopen):
        payload = {"activated": True, "instance": {"id": "abc"}}
        mock_urlopen.return_value = _make_urlopen_mock(payload)
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertTrue(ok)
        self.assertEqual(resp["activated"], True)

    @patch("urllib.request.urlopen")
    def test_bad_json_response(self, mock_urlopen):
        resp_mock = MagicMock()
        resp_mock.read.return_value = b"not json"
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertIn("unreadable", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_http_error_with_json_body(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(422, {"error": "limit reached"})
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertEqual(resp["_status_code"], 422)
        self.assertEqual(resp.get("error"), "limit reached")

    @patch("urllib.request.urlopen")
    def test_http_error_with_bad_body(self, mock_urlopen):
        err = urllib.error.HTTPError(
            url=self.URL, code=500, msg="Server Error",
            hdrs=None, fp=io.BytesIO(b"<html>crash</html>")
        )
        mock_urlopen.side_effect = err
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertEqual(resp["_status_code"], 500)

    @patch("urllib.request.urlopen")
    def test_socket_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = socket.timeout("timed out")
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertTrue(resp.get("_network_error"))
        self.assertIn("timed out", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_timeout_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timeout")
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertTrue(resp.get("_network_error"))

    @patch("urllib.request.urlopen")
    def test_ssl_error(self, mock_urlopen):
        mock_urlopen.side_effect = ssl.SSLError("certificate verify failed")
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertTrue(resp.get("_network_error"))
        self.assertIn("secure connection", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_url_error_dns(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError(
            "[Errno -2] Name or service not known"
        )
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertTrue(resp.get("_network_error"))
        self.assertIn("internet connection", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_url_error_timed_out_in_reason(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timed out")
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertIn("timed out", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_url_error_connection_refused(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertIn("network error", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_unexpected_exception(self, mock_urlopen):
        mock_urlopen.side_effect = RuntimeError("unexpected kaboom")
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertIn("unexpected", resp["error"].lower())

    @patch("urllib.request.urlopen")
    def test_http_429_rate_limit(self, mock_urlopen):
        mock_urlopen.side_effect = _make_http_error(429)
        ok, resp = _ls_post(self.URL, {"license_key": "KEY"})
        self.assertFalse(ok)
        self.assertEqual(resp["_status_code"], 429)


# ── TestExtractError ──────────────────────────────────────────────────────────

class TestExtractError(unittest.TestCase):

    def test_network_error_flag(self):
        resp = {"_network_error": True, "error": "Request timed out."}
        msg = _extract_error(resp)
        self.assertIn("timed out", msg.lower())

    def test_network_error_no_message(self):
        resp = {"_network_error": True}
        msg = _extract_error(resp)
        self.assertIn("internet", msg.lower())

    def test_plain_error_field(self):
        resp = {"error": "License key not valid."}
        self.assertEqual(_extract_error(resp), "License key not valid.")

    def test_plain_message_field(self):
        resp = {"message": "Invalid key format."}
        self.assertEqual(_extract_error(resp), "Invalid key format.")

    def test_error_takes_priority_over_message(self):
        resp = {"error": "Error msg.", "message": "Other msg."}
        self.assertEqual(_extract_error(resp), "Error msg.")

    def test_jsonapi_errors_array_detail(self):
        resp = {"errors": [{"detail": "Key has expired.", "title": "Expired"}]}
        msg = _extract_error(resp)
        self.assertEqual(msg, "Key has expired.")

    def test_jsonapi_errors_array_title_fallback(self):
        resp = {"errors": [{"title": "Not Found"}]}
        msg = _extract_error(resp)
        self.assertEqual(msg, "Not Found")

    def test_jsonapi_empty_errors_list(self):
        resp = {"errors": [], "_status_code": 400}
        msg = _extract_error(resp)
        self.assertIn("invalid", msg.lower())

    def test_status_400(self):
        msg = _extract_error({"_status_code": 400})
        self.assertIn("invalid", msg.lower())

    def test_status_404(self):
        msg = _extract_error({"_status_code": 404})
        self.assertIn("not found", msg.lower())

    def test_status_422(self):
        msg = _extract_error({"_status_code": 422})
        self.assertIn("activation limit", msg.lower())

    def test_status_403(self):
        msg = _extract_error({"_status_code": 403})
        self.assertIn("disabled or revoked", msg.lower())

    def test_status_429(self):
        msg = _extract_error({"_status_code": 429})
        self.assertIn("too many", msg.lower())

    def test_unknown_response_fallback(self):
        msg = _extract_error({})
        self.assertIn("check your key", msg.lower())

    def test_strips_whitespace_from_message(self):
        resp = {"error": "  Some error.  "}
        self.assertEqual(_extract_error(resp), "Some error.")


# ── Integration: full activation round-trip ───────────────────────────────────

class TestActivationRoundTrip(unittest.TestCase):
    """End-to-end flow: grace period → activate → validate → deactivate."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
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
        mock_urlopen.return_value = _make_urlopen_mock({
            "activated": True,
            "instance": {"id": "inst-rt"},
        })
        ok, msg = activate_license("ROUND-TRIP-KEY", self.path)
        self.assertTrue(ok, msg)

        # 3. Startup after activation → ok
        mock_urlopen.return_value = _make_urlopen_mock({"valid": True})
        state, _ = check_startup(self.path)
        self.assertEqual(state, "ok")

        # 4. Silent validate → True
        mock_urlopen.return_value = _make_urlopen_mock({"valid": True})
        valid = validate_license_silent("ROUND-TRIP-KEY", "inst-rt")
        self.assertTrue(valid)

        # 5. Deactivate
        mock_urlopen.return_value = _make_urlopen_mock({"deactivated": True})
        ok, msg = deactivate_license(self.path)
        self.assertTrue(ok, msg)

        # 6. After deactivation, startup increments count again
        state, remaining = check_startup(self.path)
        self.assertEqual(state, "grace")

    @patch("urllib.request.urlopen")
    def test_grace_expires_then_activate(self, mock_urlopen):
        """Exhaust grace period, then successfully activate."""
        for _ in range(GRACE_LAUNCHES + 1):
            state, _ = check_startup(self.path)
        self.assertEqual(state, "required")

        mock_urlopen.return_value = _make_urlopen_mock({
            "activated": True,
            "instance": {"id": "inst-late"},
        })
        ok, msg = activate_license("LATE-KEY", self.path)
        self.assertTrue(ok, msg)

        state, _ = check_startup(self.path)
        self.assertEqual(state, "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
