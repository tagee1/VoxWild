"""
clips_library.py — Clips Library persistence (save generated clips into folders).
Pure I/O functions (os / json / shutil / datetime) with no UI dependencies —
mirrors clone_library.py so it's fully testable off the GUI.

Data model (index JSON):
    {
      "folders": ["Podcast", "Ads"],           # user folders; "All Clips" is virtual
      "clips": [
        {"id","name","text","voice","duration","date","folder",
         "file","origin","trashed","deletedAt"}
      ]
    }
Virtual folders: "" / None => All Clips (all non-trashed). Recently Deleted => trashed.
"""
import json
import os
import shutil
import uuid
from datetime import datetime, timezone, timedelta


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _find(clips, clip_id):
    for c in clips:
        if c.get("id") == clip_id:
            return c
    return None


def load_library(library_dir, index_path):
    """Return {'folders': [...], 'clips': [...]}, dropping clips whose file is gone."""
    os.makedirs(library_dir, exist_ok=True)
    data = {"folders": [], "clips": []}
    if os.path.exists(index_path):
        try:
            with open(index_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                data["folders"] = [f for f in raw.get("folders", []) if isinstance(f, str)]
                data["clips"] = [c for c in raw.get("clips", [])
                                 if isinstance(c, dict) and c.get("file")
                                 and os.path.exists(c["file"])]
        except Exception as e:
            print(f"[clips_library] Failed to load {index_path}: {e}", flush=True)
    return data


def save_library(data, library_dir, index_path):
    os.makedirs(library_dir, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── folders ──────────────────────────────────────────────────────────────────
def add_folder(name, library_dir, index_path):
    name = (name or "").strip()
    if not name:
        return False
    data = load_library(library_dir, index_path)
    if name not in data["folders"]:
        data["folders"].append(name)
        save_library(data, library_dir, index_path)
    return True


def rename_folder(old_name, new_name, library_dir, index_path):
    new_name = (new_name or "").strip()
    data = load_library(library_dir, index_path)
    if not new_name or old_name not in data["folders"]:
        return False
    data["folders"] = [new_name if f == old_name else f for f in data["folders"]]
    for c in data["clips"]:
        if c.get("folder") == old_name:
            c["folder"] = new_name
    save_library(data, library_dir, index_path)
    return True


def delete_folder(name, library_dir, index_path):
    """Remove a folder; its clips fall back to All Clips (folder=''). Clips are NOT deleted."""
    data = load_library(library_dir, index_path)
    if name not in data["folders"]:
        return False
    data["folders"] = [f for f in data["folders"] if f != name]
    for c in data["clips"]:
        if c.get("folder") == name:
            c["folder"] = ""
    save_library(data, library_dir, index_path)
    return True


# ── clips ────────────────────────────────────────────────────────────────────
def add_clip(src_audio_path, meta, folder, library_dir, index_path):
    """Copy src_audio_path into library_dir, register a clip, return the new clip dict.
    meta: {name, text, voice, duration, origin?}. folder: "" for All Clips."""
    os.makedirs(library_dir, exist_ok=True)
    ext = os.path.splitext(src_audio_path)[1] or ".wav"
    dst = os.path.join(library_dir, f"{uuid.uuid4().hex}{ext}")
    shutil.copy2(src_audio_path, dst)
    data = load_library(library_dir, index_path)
    clip = {
        "id": uuid.uuid4().hex,
        "name": meta.get("name") or "Untitled",
        "text": meta.get("text", ""),
        "voice": meta.get("voice", ""),
        "duration": float(meta.get("duration", 0.0) or 0.0),
        "date": _now_iso(),
        "folder": folder or "",
        "file": dst,
        "origin": meta.get("origin", "library"),
        "trashed": False,
        "deletedAt": None,
    }
    data["clips"].append(clip)
    save_library(data, library_dir, index_path)
    return clip


def rename_clip(clip_id, new_name, library_dir, index_path):
    new_name = (new_name or "").strip()
    data = load_library(library_dir, index_path)
    c = _find(data["clips"], clip_id)
    if not c or not new_name:
        return False
    c["name"] = new_name
    save_library(data, library_dir, index_path)
    return True


def move_clip(clip_id, folder, library_dir, index_path):
    data = load_library(library_dir, index_path)
    c = _find(data["clips"], clip_id)
    if not c:
        return False
    c["folder"] = folder or ""
    save_library(data, library_dir, index_path)
    return True


def trash_clip(clip_id, library_dir, index_path):
    """Soft-delete: flag trashed + stamp deletedAt (shows in Recently Deleted)."""
    data = load_library(library_dir, index_path)
    c = _find(data["clips"], clip_id)
    if not c:
        return False
    c["trashed"] = True
    c["deletedAt"] = _now_iso()
    save_library(data, library_dir, index_path)
    return True


def restore_clip(clip_id, library_dir, index_path):
    """Un-trash. Returns the clip dict (UI uses 'origin' to pick restore target)."""
    data = load_library(library_dir, index_path)
    c = _find(data["clips"], clip_id)
    if not c:
        return None
    c["trashed"] = False
    c["deletedAt"] = None
    save_library(data, library_dir, index_path)
    return c


def delete_clip_forever(clip_id, library_dir, index_path):
    data = load_library(library_dir, index_path)
    c = _find(data["clips"], clip_id)
    if not c:
        return False
    try:
        if c.get("file") and os.path.exists(c["file"]):
            os.remove(c["file"])
    except Exception as e:
        print(f"[clips_library] Could not remove {c.get('file')}: {e}", flush=True)
    data["clips"] = [x for x in data["clips"] if x.get("id") != clip_id]
    save_library(data, library_dir, index_path)
    return True


def empty_trash(library_dir, index_path):
    data = load_library(library_dir, index_path)
    for c in [x for x in data["clips"] if x.get("trashed")]:
        delete_clip_forever(c["id"], library_dir, index_path)
    return True


def sweep_trash(library_dir, index_path, days=30):
    """Permanently drop trashed clips whose deletedAt is older than `days`. Returns count."""
    data = load_library(library_dir, index_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    doomed = []
    for c in data["clips"]:
        if not c.get("trashed") or not c.get("deletedAt"):
            continue
        try:
            when = datetime.fromisoformat(c["deletedAt"])
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if when < cutoff:
            doomed.append(c["id"])
    for cid in doomed:
        delete_clip_forever(cid, library_dir, index_path)
    return len(doomed)


# ── views (pure, no I/O) ─────────────────────────────────────────────────────
def clips_in_folder(data, folder):
    """Non-trashed clips in `folder`. folder=None/'' => All Clips (every non-trashed)."""
    out = [c for c in data["clips"] if not c.get("trashed")]
    if folder:
        out = [c for c in out if (c.get("folder") or "") == folder]
    return out


def trashed_clips(data):
    return [c for c in data["clips"] if c.get("trashed")]


def search_clips(data, query, folder=None):
    """Search non-trashed clips by name/text (case-insensitive). Optional folder scope."""
    q = (query or "").strip().lower()
    pool = clips_in_folder(data, folder)
    if not q:
        return pool
    return [c for c in pool
            if q in (c.get("name", "").lower()) or q in (c.get("text", "").lower())]
