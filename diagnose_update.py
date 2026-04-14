"""Run this on the 4GB test machine to diagnose the update check.

Double-click or run: python diagnose_update.py
Output goes to: %APPDATA%\TTS Studio\update_diagnostic.log (and stdout)
"""
import os
import sys
import urllib.request
import urllib.error
import json
import ssl
import socket
import traceback
from datetime import datetime

LOG = os.path.join(os.environ.get("APPDATA", ""), "TTS Studio", "update_diagnostic.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

VERSION = "1.0.0"  # pretend we're v1.0.0
GITHUB_REPO = "tagee1/voxwild"

log("=" * 60)
log(f"Update diagnostic — pretending to be v{VERSION}")
log(f"Python: {sys.version}")
log(f"Platform: {sys.platform}")

try:
    log("Step 1: DNS lookup for api.github.com")
    ip = socket.gethostbyname("api.github.com")
    log(f"  → resolved to {ip}")
except Exception as e:
    log(f"  FAIL: {e}")

try:
    log("Step 2: HTTP request to GitHub API")
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    req = urllib.request.Request(
        api_url, headers={"User-Agent": f"VoxWild/{VERSION}"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = resp.read()
    log(f"  → HTTP 200, {len(body)} bytes")
except urllib.error.HTTPError as e:
    log(f"  HTTP {e.code}: {e.reason}")
    sys.exit(1)
except ssl.SSLError as e:
    log(f"  SSL FAIL: {e}")
    sys.exit(1)
except Exception as e:
    log(f"  FAIL: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    sys.exit(1)

try:
    log("Step 3: Parse JSON")
    data = json.loads(body)
    log(f"  → tag_name={data.get('tag_name')}")
    log(f"  → draft={data.get('draft')}, prerelease={data.get('prerelease')}")
except Exception as e:
    log(f"  FAIL: {e}")
    sys.exit(1)

try:
    log("Step 4: Version comparison")
    tag = data.get("tag_name", "").lstrip("v")
    current = tuple(int(x) for x in VERSION.split(".") if x.isdigit())
    latest  = tuple(int(x) for x in tag.split(".")  if x.isdigit())
    log(f"  → current={current}  latest={latest}  newer={latest > current}")
    if latest > current:
        log(f"  ✅ UPDATE BANNER SHOULD SHOW for {data['tag_name']}")
    else:
        log(f"  ❌ No update (you're already on latest or newer)")
except Exception as e:
    log(f"  FAIL: {e}")

log("=" * 60)
log("Done. Share this log file if banner still not showing.")
input("Press Enter to exit...")
