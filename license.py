"""
license.py — Gumroad license key validation for TTS Studio.

Flow:
  - First GRACE_LAUNCHES launches: app runs freely, status bar shows a countdown.
  - Freemium: Fast mode free forever. Natural: 3 free generations. Enhancement: 3 free uses.
  - On activation: calls Gumroad /v2/licenses/verify (increment_uses_count=true),
    stores key locally.
  - Subsequent launches: validated locally; silent background re-validation once
    per session to catch revoked keys.

Gumroad license API:
  POST https://api.gumroad.com/v2/licenses/verify
  Body: product_permalink=<id>&license_key=<key>&increment_uses_count=<true|false>
  Response: { success: true/false, purchase: {...}, uses: N }
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
GRACE_LAUNCHES    = 3
FREE_NATURAL_USES = 3   # lifetime free Natural (Chatterbox) generations
FREE_ENHANCE_USES = 3   # lifetime free Resemble Enhance uses

# Gumroad product permalinks — one per product (monthly vs lifetime)
PRODUCT_PERMALINK          = "TTSStudioPro"          # monthly
PRODUCT_PERMALINK_LIFETIME = "TTSStudioProLifetime"  # lifetime
_ALL_PERMALINKS = (PRODUCT_PERMALINK, PRODUCT_PERMALINK_LIFETIME)

# Store URLs
STORE_URL          = "https://cookiestudios.gumroad.com"
STORE_URL_MONTHLY  = "https://cookiestudios.gumroad.com/l/TTSStudioPro"
STORE_URL_LIFETIME = "https://cookiestudios.gumroad.com/l/TTSStudioProLifetime"

_GR_VERIFY = "https://api.gumroad.com/v2/licenses/verify"

_TIMEOUT = 10  # seconds for all API calls

_DEFAULT_LICENSE = {
    "key":             None,
    "activated":       False,
    "activation_date": None,
    "launch_count":    0,
    "natural_uses":    0,
    "enhance_uses":    0,
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
            if not isinstance(data, dict):
                return _DEFAULT_LICENSE.copy()
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
    Returns True on success, False if the write failed.
    Never raises.
    """
    target = path or LICENSE_FILE
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
    except OSError:
        pass

    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


# ── Internal helpers ──────────────────────────────────────────────────────────
def _extract_error(resp):
    """Pull the most useful error string from a Gumroad error response."""
    if resp.get("_network_error"):
        return resp.get("error", "Network error. Check your internet connection.")

    # Gumroad returns {"success": false, "message": "..."} on failure
    msg = resp.get("message")
    if msg and isinstance(msg, str):
        return msg.strip()

    code = resp.get("_status_code")
    if code == 404:
        return "License key not found. Check that you entered it correctly."
    if code == 403:
        return "This license key has been disabled or refunded."
    if code == 429:
        return "Too many attempts. Please wait a moment and try again."

    return "Activation failed. Check your key and try again."


def _gr_post(params):
    """
    POST to Gumroad /v2/licenses/verify.
    Returns (ok: bool, response: dict). Never raises.
    """
    try:
        data = urllib.parse.urlencode(params).encode()
        req  = urllib.request.Request(_GR_VERIFY, data=data, method="POST")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            try:
                return True, json.loads(resp.read())
            except (json.JSONDecodeError, ValueError):
                return False, {"message": "Server returned an unreadable response."}

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
    Activate a license key via Gumroad (increments use count).
    Tries monthly permalink first, then lifetime — covers both products.
    Saves the result to license.json on success.

    Returns:
        (success: bool, message: str)
    """
    key = (key or "").strip()
    if not key:
        return False, "Please enter a license key."

    ok, resp = False, {}
    for permalink in _ALL_PERMALINKS:
        ok, resp = _gr_post({
            "product_permalink":    permalink,
            "license_key":          key,
            "increment_uses_count": "true",
        })
        if ok and resp.get("success") is True:
            break  # found the right product

    if ok and resp.get("success") is True:
        lic = load_license(path)
        lic.update({
            "key":             key,
            "activated":       True,
            "activation_date": datetime.now().isoformat(),
        })

        saved = save_license(lic, path)
        if not saved:
            return False, (
                "License was activated but could not be saved locally. "
                "Check that your disk is not full or write-protected."
            )

        verify = load_license(path)
        if not verify.get("activated") or verify.get("key") != key:
            return False, (
                "License was activated but verification failed. "
                "Please restart the app and try again."
            )

        return True, "License activated successfully. Thank you!"

    return False, _extract_error(resp)


def validate_license_silent(key):
    """
    Re-validate against Gumroad (call from a background thread).
    Tries monthly permalink first, then lifetime — covers both products.
    Returns True if confirmed valid, False if revoked.
    Network errors return True (benefit of the doubt — don't punish offline users).
    """
    key = (key or "").strip()
    if not key:
        return False

    for permalink in _ALL_PERMALINKS:
        ok, resp = _gr_post({
            "product_permalink":    permalink,
            "license_key":          key,
            "increment_uses_count": "false",
        })
        if resp.get("_network_error"):
            return True  # can't reach server — assume valid, check again next session
        if ok and resp.get("success") is True:
            return True

    return False


def deactivate_license(path=None):
    """
    Deactivate (forget) the license on this machine.
    Gumroad has no server-side deactivation endpoint, so we clear locally.
    The key's use count on Gumroad is not decremented — user should contact
    support if they need a seat freed.

    Returns:
        (success: bool, message: str)
    """
    lic = load_license(path)
    if not lic.get("activated"):
        return False, "No active license found on this machine."

    cleared = {
        "key":             None,
        "activated":       False,
        "activation_date": None,
        "launch_count":    lic.get("launch_count", 0),
        "natural_uses":    lic.get("natural_uses", 0),
        "enhance_uses":    lic.get("enhance_uses", 0),
    }
    saved = save_license(cleared, path)
    if not saved:
        return False, (
            "Could not clear license file locally. "
            "Check that your disk is not full or write-protected."
        )
    return True, (
        "License removed from this machine. "
        "To move to a new PC, contact support to reset your seat."
    )


# ── Freemium usage helpers ────────────────────────────────────────────────────
def is_pro(path=None):
    """Return True if this machine has an activated Pro license."""
    return load_license(path).get("activated", False)


def can_use_natural(path=None):
    """Return True if the user may start a Natural mode generation."""
    lic = load_license(path)
    if lic.get("activated"):
        return True
    return lic.get("natural_uses", 0) < FREE_NATURAL_USES


def natural_uses_remaining(path=None):
    """Return how many free Natural uses remain, or None if Pro (unlimited)."""
    lic = load_license(path)
    if lic.get("activated"):
        return None
    return max(0, FREE_NATURAL_USES - lic.get("natural_uses", 0))


def record_natural_use(path=None):
    """Increment the Natural use counter. No-op for Pro users. Never raises."""
    try:
        lic = load_license(path)
        if not lic.get("activated"):
            lic["natural_uses"] = lic.get("natural_uses", 0) + 1
            save_license(lic, path)
    except Exception:
        pass


def can_use_enhance(path=None):
    """Return True if the user may run Resemble Enhance."""
    lic = load_license(path)
    if lic.get("activated"):
        return True
    return lic.get("enhance_uses", 0) < FREE_ENHANCE_USES


def enhance_uses_remaining(path=None):
    """Return how many free Enhance uses remain, or None if Pro (unlimited)."""
    lic = load_license(path)
    if lic.get("activated"):
        return None
    return max(0, FREE_ENHANCE_USES - lic.get("enhance_uses", 0))


def record_enhance_use(path=None):
    """Increment the Enhance use counter. No-op for Pro users. Never raises."""
    try:
        lic = load_license(path)
        if not lic.get("activated"):
            lic["enhance_uses"] = lic.get("enhance_uses", 0) + 1
            save_license(lic, path)
    except Exception:
        pass


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

    if count <= GRACE_LAUNCHES:
        remaining = GRACE_LAUNCHES - count
        return "grace", remaining

    return "required", 0
