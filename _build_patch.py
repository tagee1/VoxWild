"""Build a patch zip for in-app updates.

Included files are the ones that typically change between releases.
Heavy dependencies (torch, numpy, onnxruntime, kokoro model, etc.)
are NOT included — they change rarely and are already installed.

If a major update adds/removes deps, users must use the full installer.
"""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: _build_patch.py <version>", file=sys.stderr)
    sys.exit(1)

VERSION = sys.argv[1]
ROOT = Path(__file__).parent
SRC  = ROOT / "dist" / "VoxWild"
DST  = ROOT / "installer_output" / f"VoxWild-Patch-{VERSION}.zip"

# Files relative to the install root that change between patch releases.
PATCH_FILES = [
    "VoxWild.exe",
    "_internal/chatterbox_worker.py",
    "_internal/enhance_worker.py",
    "_internal/icon.ico",
    "_internal/logo.png",
    "_internal/theme.json",
    "_internal/vcomp140.dll",
    "CREDITS.txt",
    "PRIVACY.txt",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not SRC.is_dir():
        print(f"error: build not found at {SRC}", file=sys.stderr)
        return 1

    DST.parent.mkdir(parents=True, exist_ok=True)
    if DST.exists():
        DST.unlink()

    manifest = {
        "version":  VERSION,
        "files":    {},
    }

    print(f"Building patch: {DST.name}")
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in PATCH_FILES:
            full = SRC / rel
            if not full.exists():
                print(f"  SKIP (missing): {rel}")
                continue
            arcname = rel.replace("\\", "/")
            zf.write(full, arcname)
            size = full.stat().st_size
            manifest["files"][arcname] = {
                "sha256": sha256_of(full),
                "size":   size,
            }
            print(f"  added: {arcname:48s} ({size / 1024:>8.1f} KB)")

        # Embed manifest in the zip
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    print()
    print(f"  -> {DST}")
    print(f"  -> {DST.stat().st_size / 1024 / 1024:.1f} MB  (uncompressed ~{sum(m['size'] for m in manifest['files'].values()) / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
