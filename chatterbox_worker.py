"""
Chatterbox TTS persistent worker process.
Runs under chatterbox_env Python (3.11).

Protocol (line-delimited JSON over stdin/stdout):
  Startup: emits {"type": "status", "msg": "..."} then {"type": "ready", "sr": <int>}
  Request: {"cmd": "generate", "text": "...", "output_path": "...",
            "audio_prompt_path": null, "exaggeration": 0.5,
            "cfg_weight": 0.5, "temperature": 0.8}
  Response: {"type": "status", "msg": "..."} ... {"type": "done"}
            or {"type": "error", "msg": "..."}
  Quit:    {"cmd": "quit"}
"""
import sys
import os
import json
import logging
import copy

# ── DLL search path fix (Windows) ─────────────────────────────────────────────
# Two separate mechanisms are needed because different loaders check different
# sources:
#   - LoadLibraryW (used by ctypes.CDLL)  → checks os.environ["PATH"]
#   - LoadLibraryExW with LOAD_LIBRARY_SEARCH_* flags (used by Python C
#     extension imports) → checks os.add_dll_directory() registrations
#
# torchaudio loads libtorchaudio.pyd via ctypes.CDLL, so its dependencies
# (torch DLLs) must be on PATH. We add all relevant DLL dirs to both.
#
# We use sysconfig.get_path("purelib") instead of hardcoding "Lib\site-packages"
# because the embeddable Python (python_embed) may place site-packages at a
# different path than a regular virtualenv (chatterbox_env).
if os.name == "nt":
    import sysconfig as _sc
    _py_dir      = os.path.dirname(os.path.abspath(sys.executable))
    _sp          = _sc.get_path("purelib")   # actual site-packages for this interpreter
    _torch_lib   = os.path.join(_sp, "torch",      "lib")
    _taudio_lib  = os.path.join(_sp, "torchaudio", "lib")
    _scripts_dir = os.path.join(_py_dir, "Scripts")

    _dll_dirs = [d for d in [_py_dir, _scripts_dir, _torch_lib, _taudio_lib]
                 if os.path.isdir(d)]

    # Keep a copy for diagnostic logging in main() — the rest of the temps are deleted below.
    _registered_dll_dirs = list(_dll_dirs)

    # 1. PATH — for ctypes.CDLL / LoadLibraryW dependency resolution
    os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")

    # 2. add_dll_directory — for Python C-extension / LoadLibraryExW resolution.
    # IMPORTANT: os.add_dll_directory() returns a handle. If the handle is
    # garbage collected the directory is removed from the search path. Store
    # handles in a module-level list so they live for the process lifetime.
    if hasattr(os, "add_dll_directory"):
        _dll_handles = [os.add_dll_directory(_d) for _d in _dll_dirs]

    del _sc, _py_dir, _sp, _torch_lib, _taudio_lib, _scripts_dir, _dll_dirs

    # 3. VCOMP140.DLL — the VC++ OpenMP runtime required by libtorchaudio.pyd.
    # On machines without the Visual C++ Redistributable, this DLL is absent from
    # System32. We bundle vcomp140.dll alongside the worker script in the
    # installer. Copy it to torchaudio\lib\ (the directory Windows searches first
    # when loading libtorchaudio.pyd) so it is always found at runtime.
    try:
        import shutil as _shutil, sysconfig as _sc2
        _worker_dir  = os.path.dirname(os.path.abspath(__file__))
        _vcomp_src   = os.path.join(_worker_dir, "vcomp140.dll")
        _sp2         = _sc2.get_path("purelib")
        _taudio_lib2 = os.path.join(_sp2, "torchaudio", "lib")
        _vcomp_dst   = os.path.join(_taudio_lib2, "vcomp140.dll")
        if (os.path.isfile(_vcomp_src)
                and os.path.isdir(_taudio_lib2)
                and not os.path.isfile(_vcomp_dst)):
            _shutil.copy2(_vcomp_src, _vcomp_dst)
        del _shutil, _sc2, _worker_dir, _vcomp_src, _sp2, _taudio_lib2, _vcomp_dst
    except Exception:
        pass
# ──────────────────────────────────────────────────────────────────────────────

# Save the real stdout for our JSON protocol BEFORE anything else touches it
_proto = sys.stdout

# Redirect stdout during imports so third-party print() spam doesn't corrupt
# the line-delimited JSON protocol.
sys.stdout = open(os.devnull, "w")

# Suppress noisy loggers
logging.disable(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"

def emit(obj):
    print(json.dumps(obj), file=_proto, flush=True)

def load_model_from_local(local_dir):
    """Load Chatterbox model step-by-step from local cache with progress messages."""
    from pathlib import Path
    from safetensors.torch import load_file
    import torch

    ckpt_dir = Path(local_dir)

    emit({"type": "status", "msg": "Loading Natural mode (1/4) — voice encoder..."})
    from chatterbox.models.voice_encoder import VoiceEncoder
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    ve.to("cpu").eval()

    emit({"type": "status", "msg": "Loading Natural mode (2/4) — speech model (2 GB, ~2 min)..."})
    from chatterbox.models.t3 import T3
    t3 = T3()
    t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
    if "model" in t3_state:
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.to("cpu").eval()

    emit({"type": "status", "msg": "Loading Natural mode (3/4) — audio decoder (1 GB)..."})
    from chatterbox.models.s3gen import S3Gen, S3GEN_SR
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen.to("cpu").eval()

    emit({"type": "status", "msg": "Loading Natural mode (4/4) — tokenizer & voice profile..."})
    from chatterbox.models.tokenizers import EnTokenizer
    from chatterbox.models.s3tokenizer import S3_SR
    tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

    from chatterbox.tts import ChatterboxTTS, Conditionals
    import torch

    conds = None
    if (ckpt_dir / "conds.pt").exists():
        try:
            conds = Conditionals.load(ckpt_dir / "conds.pt", map_location="cpu").to("cpu")
            # conds.pt is sometimes saved with batch_size=2 (CFG pair: [uncond, cond]).
            # Normalize to batch-1. Tensors live inside conds.t3 (T3Cond dataclass)
            # and conds.gen (dict) — NOT at the top-level conds.__dict__.
            for key, val in list(conds.t3.__dict__.items()):
                if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == 2:
                    setattr(conds.t3, key, val[:1])
            for key, val in list(conds.gen.items()):
                if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == 2:
                    conds.gen[key] = val[:1]
        except Exception:
            conds = None

    import perth
    watermarker = perth.PerthImplicitWatermarker()

    model = ChatterboxTTS(t3, s3gen, ve, tokenizer, "cpu", conds=conds)
    model.watermarker = watermarker
    return model

def _scan_missing_dlls(pyd_path):
    """
    Read the PE import table of a .pyd/.dll and check each dependency.
    Returns list of (dll_name, found) tuples, plus error info if parsing fails.

    PE optional header import directory offsets (from start of optional header):
      PE32  (x86, magic 0x10B): data directories start at offset 96  → import at 96+8  = 104
      PE32+ (x64, magic 0x20B): data directories start at offset 112 → import at 112+8 = 120
    """
    import struct, ctypes
    results = []
    try:
        with open(pyd_path, "rb") as f:
            d = f.read()

        # MZ header → PE offset
        if d[0:2] != b"MZ":
            return [("(not a valid PE file)", False)]
        pe = struct.unpack_from("<I", d, 0x3C)[0]
        if d[pe:pe+4] != b"PE\x00\x00":
            return [("(PE signature not found)", False)]

        machine = struct.unpack_from("<H", d, pe + 4)[0]
        ns      = struct.unpack_from("<H", d, pe + 6)[0]   # number of sections
        osz     = struct.unpack_from("<H", d, pe + 20)[0]  # optional header size
        opt     = pe + 24                                   # start of optional header
        magic   = struct.unpack_from("<H", d, opt)[0]

        # Correct import directory RVA offset per architecture
        if magic == 0x20B:   # PE32+ (64-bit)
            import_rva_off = opt + 120
        elif magic == 0x10B: # PE32  (32-bit)
            import_rva_off = opt + 104
        else:
            return [(f"(unknown PE magic: 0x{magic:X})", False)]

        irva = struct.unpack_from("<I", d, import_rva_off)[0]
        if irva == 0:
            return [("(no import directory)", True)]  # technically fine

        sects = opt + osz  # section table immediately follows optional header

        def r2o(rva):
            """Convert RVA to file offset using the section table."""
            for i in range(ns):
                s = sects + i * 40
                va = struct.unpack_from("<I", d, s + 12)[0]
                vs = struct.unpack_from("<I", d, s + 16)[0]
                ro = struct.unpack_from("<I", d, s + 20)[0]
                if va <= rva < va + vs:
                    return ro + (rva - va)
            return None

        off = r2o(irva)
        if off is None:
            return [(f"(import RVA 0x{irva:X} not in any section)", False)]

        k32 = ctypes.windll.kernel32
        while True:
            name_rva = struct.unpack_from("<I", d, off + 12)[0]
            if name_rva == 0:
                break
            name_off = r2o(name_rva)
            if name_off is None:
                break
            end = d.index(b"\x00", name_off)
            dll_name = d[name_off:end].decode("ascii", errors="replace")

            h = k32.LoadLibraryW(dll_name)
            if h:
                k32.FreeLibrary(h)
                results.append((dll_name, True))
            else:
                err = ctypes.GetLastError()
                results.append((dll_name, False))

            off += 20  # each import directory entry is 20 bytes

    except Exception as ex:
        results.append((f"(parser exception: {ex})", False))

    return results


def _preload_torch_dlls():
    """
    Pre-load every torch DLL in dependency order BEFORE importing torchaudio.

    Why: ctypes.CDLL(libtorchaudio.pyd) fails with "Could not find module …
    (or one of its dependencies)" because Windows can't resolve the torch DLLs
    that libtorchaudio depends on, even when those dirs are in PATH/add_dll_directory.

    The guaranteed fix: load each torch DLL explicitly first (leaf deps first).
    Once a DLL is loaded into the process, Windows returns the existing handle
    for any subsequent request — no search path needed.
    """
    if os.name != "nt":
        return
    import ctypes
    _py_dir    = os.path.dirname(os.path.abspath(sys.executable))
    _torch_lib = os.path.join(_py_dir, "Lib", "site-packages", "torch", "lib")
    if not os.path.isdir(_torch_lib):
        return
    # Load in dependency order: leaves first, dependents after.
    # Any DLL that fails to load is silently skipped — if it's truly
    # missing we'll get a proper error when torchaudio imports.
    _load_order = [
        "libiomp5md",       # Intel OpenMP — system deps only
        "libiompstubs5md",  # OpenMP stubs
        "uv",               # libuv — system deps only
        "asmjit",           # JIT assembler — system deps only
        "c10",              # depends on libiomp5md, uv
        "fbgemm",           # depends on c10, asmjit, libiomp5md
        "shm",              # shared memory — depends on c10
        "torch_cpu",        # depends on c10, fbgemm, libiomp5md, asmjit
        "torch_global_deps",
        "torch_python",     # depends on torch_cpu, c10, python3xx
        "torch",            # main lib
    ]
    for _name in _load_order:
        _path = os.path.join(_torch_lib, f"{_name}.dll")
        if os.path.isfile(_path):
            try:
                ctypes.CDLL(_path)
            except OSError:
                pass  # missing dep at this stage → try anyway; torchaudio will surface the real error


def main():
    emit({"type": "status", "msg": "Starting Natural mode — loading 3 GB model on CPU..."})
    _preload_torch_dlls()

    # Scan libtorchaudio.pyd's DLL imports BEFORE attempting to load it.
    # torch must be imported first so its DLLs are in the process table.
    try:
        import torch as _torch_for_scan
        _py_dir   = os.path.dirname(os.path.abspath(sys.executable))
        _pyd_path = os.path.join(_py_dir, "Lib", "site-packages",
                                 "torchaudio", "lib", "libtorchaudio.pyd")
        _scan     = _scan_missing_dlls(_pyd_path)
        _log_path = os.path.join(os.path.dirname(_py_dir), "dll_diagnostic.log")
        with open(_log_path, "w", encoding="utf-8") as _lf:
            _lf.write(f"libtorchaudio.pyd dependency scan:\n")
            for entry in _scan:
                dll, found = entry[0], entry[1]
                _lf.write(f"  {'OK     ' if found else 'MISSING'}  {dll}\n")
    except Exception as _se:
        pass

    try:
        import torchaudio
        from chatterbox.tts import ChatterboxTTS

        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download("ResembleAI/chatterbox", local_files_only=True)
            model = load_model_from_local(local_dir)
        except Exception:
            emit({"type": "status", "msg": "Local cache not found — downloading model (first run only)..."})
            model = ChatterboxTTS.from_pretrained("cpu")

    except MemoryError as e:
        emit({"type": "error", "msg": f"Not enough RAM to load model: {e}"})
        return
    except OSError as e:
        import traceback as _tb
        _winerr = getattr(e, "winerror", None)
        _full   = _tb.format_exc()
        # Write full diagnostics to log so the truncated UI message isn't the only info
        try:
            _log_path = os.path.join(
                os.environ.get("APPDATA", ""), "TTS Studio", "natural_mode_error.log"
            )
            with open(_log_path, "w", encoding="utf-8") as _lf:
                _lf.write(f"winerror: {_winerr}\n")
                _lf.write(f"str(e):   {e}\n\n")
                _lf.write(_full)
        except Exception:
            pass
        if _winerr == 1455 or "paging file" in str(e).lower():
            emit({"type": "error", "msg": f"Not enough RAM/virtual memory to load model: {e}"})
        else:
            emit({"type": "error", "msg": f"Model load failed (OS error {_winerr}): {e}  [E099]"})
        return
    except Exception as e:
        import traceback as _tb
        _full = _tb.format_exc()
        try:
            _log_path = os.path.join(
                os.environ.get("APPDATA", ""), "TTS Studio", "natural_mode_error.log"
            )
            with open(_log_path, "w", encoding="utf-8") as _lf:
                _lf.write(f"Exception type: {type(e).__name__}\n")
                _lf.write(f"str(e): {e}\n\n")
                _lf.write(_full)
        except Exception:
            pass
        emit({"type": "error", "msg": f"Model load failed: {e}"})
        return

    emit({"type": "ready", "sr": model.sr})

    # Save default voice conditioning so we can restore it when the user
    # switches back to Default voice. model.generate() mutates model.conds
    # in-place when given an audio prompt, so without this reset, subsequent
    # calls with audio_prompt_path=None would silently keep using the last
    # clone's voice.
    try:
        _default_conds = copy.deepcopy(model.conds)
    except Exception:
        _default_conds = None

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception:
            continue

        cmd = req.get("cmd")
        if cmd == "quit":
            break

        if cmd == "generate":
            try:
                import traceback as _tb
                text         = req.get("text", "")
                out_path     = req.get("output_path", "")
                if not text or not out_path:
                    emit({"type": "error", "msg": "Malformed generate request (missing text or output_path)."})
                    continue
                audio_prompt = req.get("audio_prompt_path") or None
                exaggeration = float(req.get("exaggeration", 0.5))
                cfg_weight   = float(req.get("cfg_weight", 0.5))
                temperature  = float(req.get("temperature", 0.8))

                # Preprocess voice clone audio:
                #   1. Convert stereo → mono (model requires 1 channel)
                #   2. Normalize amplitude — quiet recordings (e.g. phone mics, whispers)
                #      produce near-zero embeddings that cause the model to output
                #      garbage or NaN audio, which crashes PortAudio on playback.
                _tmp_prompt = None
                if audio_prompt:
                    try:
                        import tempfile
                        import torch as _torch
                        waveform, sr = torchaudio.load(audio_prompt)
                        changed = False

                        # Stereo → mono
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                            changed = True

                        # Minimum duration: VoiceEncoder needs at least 1.6s of speech
                        # (processed in 160ms windows). Reject anything under 2s to be safe.
                        duration_sec = waveform.shape[-1] / sr
                        if duration_sec < 2.0:
                            emit({"type": "error",
                                  "msg": f"Voice clone recording is too short ({duration_sec:.1f}s). "
                                         "Please record at least 6 seconds of clear speech.  [E011]"})
                            audio_prompt = None

                        # Amplitude normalization — target peak of 0.95
                        if audio_prompt is not None:
                            peak = waveform.abs().max().item()
                            if peak < 0.001:
                                emit({"type": "error",
                                      "msg": "Voice clone recording is too quiet to use "
                                             f"(peak level: {peak:.5f}). "
                                             "Please re-record in a louder environment.  [E010]"})
                                audio_prompt = None
                            elif peak < 0.5:
                                waveform = waveform * (0.95 / peak)
                                changed = True

                        if audio_prompt and changed:
                            _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            torchaudio.save(_tmp.name, waveform, sr)
                            _tmp.close()
                            _tmp_prompt = _tmp.name
                            audio_prompt = _tmp_prompt
                    except Exception:
                        pass  # if preprocessing fails, pass original and let model error naturally

                # Restore default voice conditioning when no clone is selected.
                # model.generate() mutates model.conds in-place when given a
                # voice prompt, so we must reset it to prevent bleed-through.
                if audio_prompt is None and _default_conds is not None:
                    try:
                        model.conds = copy.deepcopy(_default_conds)
                    except Exception as _ce:
                        emit({"type": "error",
                              "msg": f"Failed to reset voice conditioning: {_ce}  [E012]"})
                        continue

                emit({"type": "status", "msg": "Generating audio..."})
                wav = model.generate(
                    text,
                    audio_prompt_path=audio_prompt,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )

                emit({"type": "status", "msg": "Saving chunk..."})
                torchaudio.save(out_path, wav, model.sr)
                emit({"type": "done"})
            except Exception as e:
                import traceback as _tb
                print(_tb.format_exc(), file=sys.stderr, flush=True)
                emit({"type": "error", "msg": str(e)})
            finally:
                if _tmp_prompt:
                    try:
                        os.unlink(_tmp_prompt)
                    except OSError:
                        pass

main()
