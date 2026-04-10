"""
test_error_messages.py — Verify fmt_err maps technical exceptions to plain English
and always appends a crash code in [EXXX] format.

Run with:
    python test_error_messages.py
"""
import re
from tts_utils import fmt_err


def test_fmt_err():
    cases = [
        # (exception, expected text substring, expected code)
        (MemoryError("CUDA out of memory"),                        "RAM",            "E001"),
        (OSError("paging file is too small"),                      "RAM",            "E001"),
        (Exception("WinError 1455"),                               "RAM",            "E001"),
        (RuntimeError("Not enough RAM to load model: "),           "RAM",            "E001"),
        (Exception("PortAudio: No Default Output Device"),         "Audio device",   "E002"),
        (PermissionError("Access is denied: 'C:\\foo.wav'"),       "Permission",     "E003"),
        (FileNotFoundError("No such file: 'voice.wav'"),           "not found",      "E005"),
        (Exception("ConnectionError: timed out"),                  "Network",        "E006"),
        (Exception("Sizes of tensors must match except in dim 1"), "audio prompt",   "E007"),
        (Exception("Voice clone recording is too quiet"),          "too quiet",      "E010"),
        (RuntimeError("Audio cleanup (pedalboard) failed: boom  [E013]"), "cleanup",  "E013"),
        (Exception("RuntimeError: something obscure happened"),    "something",      "E099"),
        (ValueError("bad value supplied"),                         "bad value",      "E099"),
    ]

    all_pass = True
    for exc, expected_text, expected_code in cases:
        result = fmt_err(exc)
        text_ok = expected_text.lower() in result.lower()
        code_match = re.search(r'\[(E\d{3})\]', result)
        code_ok = code_match is not None and code_match.group(1) == expected_code
        ok = text_ok and code_ok
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {type(exc).__name__}: '{str(exc)[:45]}' -> '{result}'")
        if not text_ok:
            print(f"         Expected text: '{expected_text}'")
        if not code_ok:
            got = code_match.group(1) if code_match else "no code"
            print(f"         Expected code: {expected_code}, got: {got}")

    print()
    if all_pass:
        print("PASS — all errors have plain-English messages and correct crash codes")
    else:
        import sys
        print("FAIL — see above")
        sys.exit(1)


if __name__ == "__main__":
    test_fmt_err()
