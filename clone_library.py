"""
clone_library.py — Voice clone library persistence.
Pure I/O functions (os / json / shutil) with no UI dependencies.
"""
import json
import os
import shutil
import uuid


def load_clone_library(clone_dir, index_path):
    """Return list of valid clone entries, filtering out missing files."""
    os.makedirs(clone_dir, exist_ok=True)
    if os.path.exists(index_path):
        try:
            with open(index_path, encoding="utf-8") as f:
                entries = json.load(f)
            return [e for e in entries
                    if isinstance(e, dict) and "file" in e and os.path.exists(e["file"])]
        except Exception as e:
            print(f"[clone_library] Failed to load {index_path}: {e}", flush=True)
    return []


def save_clone_library(entries, clone_dir, index_path):
    """Persist clone entries list to index_path."""
    os.makedirs(clone_dir, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def rename_clone_in_library(old_name, new_name, clone_dir, index_path):
    """Rename a clone entry in the library. Returns True on success, False if old_name not found."""
    entries = load_clone_library(clone_dir, index_path)
    for entry in entries:
        if entry.get("name") == old_name:
            entry["name"] = new_name
            save_clone_library(entries, clone_dir, index_path)
            return True
    return False


def add_clone_to_library(name, src_wav_path, clone_dir, index_path):
    """Copy src_wav_path into clone_dir, register in library, return new entry dict."""
    os.makedirs(clone_dir, exist_ok=True)
    dst = os.path.join(clone_dir, f"{uuid.uuid4().hex}.wav")
    shutil.copy2(src_wav_path, dst)
    entries = load_clone_library(clone_dir, index_path)
    entry = {"name": name, "file": dst}
    entries.append(entry)
    save_clone_library(entries, clone_dir, index_path)
    return entry
