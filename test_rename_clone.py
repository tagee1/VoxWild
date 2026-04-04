"""
test_rename_clone.py — Tests for voice clone rename functionality.
"""
import os
import json
import tempfile
import shutil

from clone_library import (
    add_clone_to_library,
    load_clone_library,
    rename_clone_in_library,
    save_clone_library,
)


def _setup():
    """Create a temp dir with a clone index and a dummy wav file."""
    d = tempfile.mkdtemp()
    index = os.path.join(d, "clones.json")
    wav = os.path.join(d, "dummy.wav")
    open(wav, "wb").close()  # empty file is fine for these tests
    return d, index, wav


def test_rename_basic():
    d, index, wav = _setup()
    try:
        add_clone_to_library("Alice", wav, d, index)
        result = rename_clone_in_library("Alice", "Alice v2", d, index)
        assert result is True
        entries = load_clone_library(d, index)
        names = [e["name"] for e in entries]
        assert "Alice v2" in names
        assert "Alice" not in names
    finally:
        shutil.rmtree(d)


def test_rename_not_found_returns_false():
    d, index, wav = _setup()
    try:
        add_clone_to_library("Bob", wav, d, index)
        result = rename_clone_in_library("Nobody", "Someone", d, index)
        assert result is False
        # Bob should be untouched
        entries = load_clone_library(d, index)
        assert entries[0]["name"] == "Bob"
    finally:
        shutil.rmtree(d)


def test_rename_preserves_file_path():
    d, index, wav = _setup()
    try:
        entry = add_clone_to_library("Carol", wav, d, index)
        original_file = entry["file"]
        rename_clone_in_library("Carol", "Carol Renamed", d, index)
        entries = load_clone_library(d, index)
        renamed = next(e for e in entries if e["name"] == "Carol Renamed")
        assert renamed["file"] == original_file
    finally:
        shutil.rmtree(d)


def test_rename_multiple_clones_only_renames_target():
    d, index, wav = _setup()
    try:
        add_clone_to_library("Dave", wav, d, index)
        add_clone_to_library("Eve", wav, d, index)
        rename_clone_in_library("Dave", "David", d, index)
        entries = load_clone_library(d, index)
        names = [e["name"] for e in entries]
        assert "David" in names
        assert "Dave" not in names
        assert "Eve" in names  # untouched
    finally:
        shutil.rmtree(d)


def test_rename_to_same_name_is_noop():
    d, index, wav = _setup()
    try:
        add_clone_to_library("Frank", wav, d, index)
        result = rename_clone_in_library("Frank", "Frank", d, index)
        assert result is True
        entries = load_clone_library(d, index)
        assert entries[0]["name"] == "Frank"
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    tests = [
        test_rename_basic,
        test_rename_not_found_returns_false,
        test_rename_preserves_file_path,
        test_rename_multiple_clones_only_renames_target,
        test_rename_to_same_name_is_noop,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(failed)
