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

def main():
    emit({"type": "status", "msg": "Starting Natural mode — loading 3 GB model on CPU..."})
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
        if getattr(e, "winerror", None) == 1455 or "paging file" in str(e).lower():
            emit({"type": "error", "msg": f"Not enough RAM/virtual memory to load model: {e}"})
        else:
            emit({"type": "error", "msg": f"Model load failed (OS error): {e}"})
        return
    except Exception as e:
        emit({"type": "error", "msg": f"Model load failed: {e}"})
        return

    emit({"type": "ready", "sr": model.sr})

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
                text         = req["text"]
                audio_prompt = req.get("audio_prompt_path") or None
                exaggeration = float(req.get("exaggeration", 0.5))
                cfg_weight   = float(req.get("cfg_weight", 0.5))
                temperature  = float(req.get("temperature", 0.8))
                out_path     = req["output_path"]

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
