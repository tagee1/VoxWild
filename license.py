"""
license.py — Lemon Squeezy license key validation for TTS Studio.

Flow:
  - First GRACE_LAUNCHES launches: app runs freely, status bar shows a countdown.
  - After grace period: blocking activation dialog shown before app is usable.
  - On activation: calls Lemon Squeezy /activate, stores key + instance_id locally.
  - Subsequent launches: validated locally; silent background re-validation once
    per session to catch revoked keys.

Grace period math (GRACE_LAUNCHES = 3):
  Launch 1 → grace, 2 remaining
  Launch 2 → grace, 1 remaining
  Launch 3 → grace, 0 remaining  (last free launch)
  Launch 4 → required            (blocked)
"""
import os
import json
import socket
import ssl
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
_USER_DIR    = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "TTS Studio")
LICENSE_FILE = os.path.join(_USER_DIR, "license.json")

# ── Constants ─────────────────────────────────────────────────────────────────
GRACE_LAUNCHES = 3

# Replace with your real Lemon Squeezy store URL once the product is live.
STORE_URL = "https://yourstore.lemonsqueezy.com"

_LS_ACTIVATE   = "https://api.lemonsqueezy.com/v1/licenses/activate"
_LS_VALIDATE   = "https://api.lemonsqueezy.com/v1/licenses/validate"
_LS_DEACTIVATE = "https://api.lemonsqueezy.com/v1/licenses/deactivate"

_TIMEOUT = 10  # seconds for all API calls

_DEFAULT_LICENSE = {
    "key":             None,
    "instance_id":     None,
    "activated":       False,
    "activation_date": None,
    "launch_count":    0,
}


# ── License file I/O ──────────────────────────────────────────────────────────
def load_license(path=None):
    """
    Load license data from disk. Always returns a complete dict with all
    expected keys, even if the file is missing, corrupt, or partially written.
    Never raises.
    """
    target = path or LICENSE_FILE
    if os.path.exists(target):
        try:
            with open(target, encoding="utf-8") as f:
                data = json.load(f)
            # Reject non-dict values (e.g. a list or null from corrupt writes)
            if not isinstance(data, dict):
                return _DEFAULT_LICENSE.copy()
            # Forward-compatible: add missing keys introduced in later versions
            for k, v in _DEFAULT_LICENSE.items():
                if k not in data:
                    data[k] = v
            return data
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return _DEFAULT_LICENSE.copy()


def save_license(data, path=None):
    """
    Persist license data to disk.
    Returns True on success, False if the write failed (disk full, read-only, etc.).
    Never raises.
    """
    target = path or LICENSE_FILE
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
    except OSError:
        pass  # attempt the write anyway; open() will fail if dir truly missing

    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


# ── Internal helpers ──────────────────────────────────────────────────────────
def _extract_error(resp):
    """
    Pull the most useful error string from a Lemon Squeezy error response.
    Handles several LS response shapes and maps HTTP status codes to plain
    English when no message is present.
    """
    # Network / transport errors added by _ls_post
    if resp.get("_network_error"):
        return resp.get("error", "Network error. Check your internet connection.")

    # Plain string fields
    for field in ("error", "message"):
        val = resp.get(field)
        if val and isinstance(val, str):
            return val.strip()

    # JSON:API errors array  {"errors": [{"detail": "..."}]}
    errors = resp.get("errors")
    if isinstance(errors, list) and errors:
        item = errors[0]
        for key in ("detail", "title"):
            if item.get(key):
                return str(item[key])

    # Fall back to HTTP status code hints
    code = resp.get("_status_code")
    if code == 400:
        return "Invalid license key format."
    if code == 404:
        return "License key not found."
    if code == 422:
        return "This license key has reached its activation limit or is invalid."
    if code == 403:
        return "This license key has been disabled or revoked."
    if code == 429:
        return "Too many activation attempts. Please wait a moment and try again."

    return "Activation failed. Check your key and try again."


def _ls_post(url, params):
    """
    POST form data to a Lemon Squeezy endpoint.
    Returns (ok: bool, response: dict).
    Never raises — all exceptions are caught and returned as error dicts.
    """
    try:
        data = urllib.parse.urlencode(params).encode()
        req  = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            try:
                return True, json.loads(resp.read())
            except (json.JSONDecodeError, ValueError):
                return False, {"error": "Server returned an unreadable response."}

    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        body["_status_code"] = e.code
        return False, body

    except (socket.timeout, TimeoutError):
        return False, {
            "error": "Request timed out. Check your internet connection and try again.",
            "_network_error": True,
        }

    except ssl.SSLError as e:
        return False, {
            "error": f"Secure connection failed ({e}). Check your system clock.",
            "_network_error": True,
        }

    except urllib.error.URLError as e:
        reason = str(e.reason).lower()
        if "timed out" in reason:
            msg = "Request timed out. Check your internet connection and try again."
        elif any(w in reason for w in ("name or service", "nodename", "getaddrinfo")):
            msg = "Could not reach activation server. Check your internet connection."
        else:
            msg = f"Network error: {e.reason}"
        return False, {"error": msg, "_network_error": True}

    except Exception as e:
        return False, {"error": f"Unexpected error: {e}"}


# ── Public API ────────────────────────────────────────────────────────────────
def activate_license(key, path=None):
    """
    Activate a license key on this machine via Lemon Squeezy.
    Saves the result to license.json on success.

    Args:
        key:  The license key string entered by the user.
        path: Override for the license file path (used in tests).

    Returns:
        (success: bool, message: str)
    """
    key = (key or "").strip()
    if not key:
        return False, "Please enter a license key."

    instance_name = socket.gethostname() or "TTS Studio"
    ok, resp = _ls_post(_LS_ACTIVATE, {
        "license_key":   key,
        "instance_name": instance_name,
    })

    if ok and resp.get("activated") is True:
        instance_id = (resp.get("instance") or {}).get("id")
        if not instance_id:
            return False, (
                "Server confirmed activation but did not return an instance ID. "
                "Please try again or contact support."
            )

        lic = load_license(path)
        lic.update({
            "key":             key,
            "instance_id":     instance_id,
            "activated":       True,
            "activation_date": datetime.now().isoformat(),
        })

        saved = save_license(lic, path)
        if not saved:
            return False, (
                "License was activated on the server but could not be saved locally. "
                "Check that your disk is not full or write-protected."
            )

        # Double-check the file actually persisted
        verify = load_license(path)
        if not verify.get("activated") or verify.get("key") != key:
            return False, (
                "License was activated but verification failed. "
                "Please restart the app and try again."
            )

        return True, "License activated successfully. Thank you!"

    return False, _extract_error(resp)


def validate_license_silent(key, instance_id):
    """
    Re-validate against Lemon Squeezy (call from a background thread).
    Returns True if confirmed valid, False if revoked OR if network is down.
    The caller should treat False as "could not confirm" rather than "definitely revoked"
    when network errors are possible — check `_network_error` if you need the distinction.
    """
    key         = (key or "").strip()
    instance_id = (instance_id or "").strip()
    if not key or not instance_id:
        return False

    ok, resp = _ls_post(_LS_VALIDATE, {
        "license_key": key,
        "instance_id": instance_id,
    })
    # Network errors are not revocations — treat them as "unable to confirm"
    # (caller should decide whether to penalise the user)
    if resp.get("_network_error"):
        return True   # benefit of the doubt; don't punish offline users

    return ok and resp.get("valid") is True


def deactivate_license(path=None):
    """
    Deactivate this machine's license (e.g. before moving to a new PC).
    Clears the local license file on success.

    Returns:
        (success: bool, message: str)
    """
    lic = load_license(path)
    key         = (lic.get("key") or "").strip()
    instance_id = (lic.get("instance_id") or "").strip()

    if not key or not instance_id:
        return False, "No active license found on this machine."

    ok, resp = _ls_post(_LS_DEACTIVATE, {
        "license_key": key,
        "instance_id": instance_id,
    })

    if ok and resp.get("deactivated") is True:
        cleared = {
            "key":             None,
            "instance_id":     None,
            "activated":       False,
            "activation_date": None,
            "launch_count":    lic.get("launch_count", 0),
        }
        saved = save_license(cleared, path)
        if not saved:
            return False, (
                "License was deactivated on the server but could not be saved locally. "
                "Check that your disk is not full or write-protected."
            )
        return True, "License deactivated. You can now activate on another machine."

    return False, _extract_error(resp)


def check_startup(path=None):
    """
    Call once at startup to determine license state.
    Increments launch_count when not yet activated.

    Returns:
        ("ok",       0)           — activated, proceed normally
        ("grace",    remaining)   — within free trial (remaining = launches after this one)
        ("required", 0)           — trial expired, activation required
    """
    lic = load_license(path)
    if lic.get("activated"):
        return "ok", 0

    count = lic.get("launch_count", 0) + 1
    lic["launch_count"] = count
    save_license(lic, path)

    # Launch 1..GRACE_LAUNCHES are free; launch GRACE_LAUNCHES+1 is blocked.
    if count <= GRACE_LAUNCHES:
        remaining = GRACE_LAUNCHES - count   # free launches left AFTER this one
        return "grace", remaining

    return "required", 0
