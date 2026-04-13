"""
Local end-to-end test harness for chatterbox_worker.py.

Runs the worker as a subprocess using chatterbox_env's Python (which has a
real chatterbox install), sends a generate request via stdin, and checks
the output WAV.

This bypasses the installer/PyInstaller/python_embed entirely so we can
iterate on chatterbox_worker.py changes in seconds instead of minutes.

Usage:
    python test_worker_local.py
    python test_worker_local.py --cfg-weight 0.5 --text "Hello world"
    python test_worker_local.py --keep-output  # don't delete the wav

Exit code 0 if generation succeeds, 1 otherwise.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORKER = REPO / "chatterbox_worker.py"

# Default to chatterbox_env (the dev install). Override with --python-embed to
# test against the bit-for-bit end-user environment.
DEV_ENV_PY = REPO / "chatterbox_env" / "Scripts" / "python.exe"
EMBED_PY = Path(os.environ.get("APPDATA", "")) / "TTS Studio" / "python_embed" / "python.exe"
ENV_PY = DEV_ENV_PY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--audio-prompt", default=None,
                    help="Optional path to a voice clone wav.")
    ap.add_argument("--keep-output", action="store_true")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Seconds to wait for the worker to finish (default 600).")
    ap.add_argument("--python-embed", action="store_true",
                    help="Run against the python_embed (end-user env) instead of chatterbox_env.")
    args = ap.parse_args()

    global ENV_PY
    if args.python_embed:
        ENV_PY = EMBED_PY

    if not ENV_PY.exists():
        print(f"FAIL: chatterbox_env Python not found at {ENV_PY}", file=sys.stderr)
        return 1
    if not WORKER.exists():
        print(f"FAIL: worker not found at {WORKER}", file=sys.stderr)
        return 1

    out_wav = Path(tempfile.gettempdir()) / "test_worker_output.wav"
    if out_wav.exists():
        out_wav.unlink()

    print(f"[harness] worker:    {WORKER}")
    print(f"[harness] python:    {ENV_PY}")
    print(f"[harness] output:    {out_wav}")
    print(f"[harness] cfg_weight={args.cfg_weight} text={args.text!r}")
    print()

    proc = subprocess.Popen(
        [str(ENV_PY), str(WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # Binary mode — matches the production binary-pipe protocol.
        # Worker writes UTF-8 bytes; we decode here.
        cwd=str(REPO),
    )

    def send(obj):
        line = json.dumps(obj)
        print(f"[harness] >>> {line}")
        proc.stdin.write((line + "\n").encode("utf-8"))
        proc.stdin.flush()

    def read_until(predicate, deadline):
        """Read JSON lines from stdout until predicate(msg) is True or deadline."""
        while True:
            if time.time() > deadline:
                return None
            if proc.poll() is not None:
                return None
            raw = proc.stdout.readline()
            if not raw:
                return None
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[worker stdout, non-JSON] {line}")
                continue
            print(f"[harness] <<< {msg}")
            if predicate(msg):
                return msg

    deadline = time.time() + args.timeout
    try:
        ready = read_until(lambda m: m.get("type") in ("ready", "error"), deadline)
        if ready is None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            print("FAIL: worker exited or timed out before sending ready.",
                  file=sys.stderr)
            if stderr:
                print(f"[worker stderr]\n{stderr}", file=sys.stderr)
            return 1
        if ready.get("type") == "error":
            print(f"FAIL: worker errored at startup: {ready.get('msg')}",
                  file=sys.stderr)
            return 1

        send({
            "cmd": "generate",
            "text": args.text,
            "output_path": str(out_wav),
            "audio_prompt_path": args.audio_prompt,
            "exaggeration": args.exaggeration,
            "cfg_weight": args.cfg_weight,
            "temperature": args.temperature,
        })

        result = read_until(lambda m: m.get("type") in ("done", "error"), deadline)
        if result is None:
            print("FAIL: worker exited or timed out before finishing generate.",
                  file=sys.stderr)
            return 1
        if result.get("type") == "error":
            print(f"FAIL: generate errored: {result.get('msg')}", file=sys.stderr)
            return 1

        if not out_wav.exists():
            print(f"FAIL: output wav was not written at {out_wav}", file=sys.stderr)
            return 1
        size = out_wav.stat().st_size
        print(f"\nPASS: wrote {size} bytes to {out_wav}")
        if not args.keep_output:
            out_wav.unlink()
        return 0
    finally:
        try:
            send({"cmd": "quit"})
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        if proc.stderr:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            if stderr.strip():
                print(f"\n[worker stderr tail]\n{stderr}")


if __name__ == "__main__":
    sys.exit(main())
