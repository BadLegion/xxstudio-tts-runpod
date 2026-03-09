#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Serverless handler til Qwen3-TTS.

Håndterer 3 GPU-operationer:
  1. generate_voice_clone  — TTS med voice cloning
  2. create_clone_prompt   — Opret clone prompt fra reference audio
  3. design_voice          — Voice design (2-model operation)

Deployment:
  1. docker build -t yourname/xxstudio-tts:v1 .
  2. docker push yourname/xxstudio-tts:v1
  3. Opret RunPod Serverless Endpoint med dette image
  4. Kopiér Endpoint ID til XXXStudio indstillinger
"""

import base64
import io
import traceback

import torch
import numpy as np
import soundfile as sf
import runpod

# ── Global model-state (persisterer mellem kald på samme worker) ──
_models = {}

CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def _get_model(model_id):
    """Lazy-load model (cached mellem kald)."""
    if model_id not in _models:
        from qwen_tts import Qwen3TTSModel
        print(f"[handler] Loader model: {model_id}...")
        _models[model_id] = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        print(f"[handler] Model loadet: {model_id}")
    return _models[model_id]


def _serialize_audio(wav_np, sr):
    """Konvertér numpy audio til base64 WAV."""
    buf = io.BytesIO()
    sf.write(buf, wav_np, sr, format="WAV")
    return {
        "wav_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "sample_rate": sr,
    }


def _deserialize_audio(b64_str):
    """Konvertér base64 WAV til (numpy, sr)."""
    buf = io.BytesIO(base64.b64decode(b64_str))
    wav, sr = sf.read(buf, dtype="float32")
    return wav, sr


# ── Handler ─────────────────────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler — dispatcher til de 3 GPU-operationer."""
    inp = job["input"]
    action = inp.get("action")

    try:
        if action == "generate_voice_clone":
            return _handle_generate(inp)
        elif action == "create_clone_prompt":
            return _handle_clone_prompt(inp)
        elif action == "design_voice":
            return _handle_design_voice(inp)
        elif action == "health":
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
            return {"status": "ok", "gpu": gpu_name}
        else:
            return {"error": f"Ukendt action: {action}"}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def _handle_generate(inp):
    """TTS generation med voice cloning."""
    model = _get_model(CLONE_MODEL_ID)

    # Deserialisér clone prompt
    prompt_buf = io.BytesIO(base64.b64decode(inp["clone_prompt_b64"]))
    clone_prompt = torch.load(prompt_buf, weights_only=False)

    wavs, sr = model.generate_voice_clone(
        text=inp["text"],
        language=inp.get("language", "English"),
        voice_clone_prompt=clone_prompt,
        temperature=inp.get("temperature", 1.0),
        top_p=inp.get("top_p", 0.8),
        repetition_penalty=inp.get("repetition_penalty", 1.05),
    )

    return _serialize_audio(wavs[0], sr)


def _handle_clone_prompt(inp):
    """Opret clone prompt fra reference audio."""
    model = _get_model(CLONE_MODEL_ID)

    wav, sr = _deserialize_audio(inp["ref_audio_b64"])

    prompt = model.create_voice_clone_prompt(
        ref_audio=(wav, sr),
        ref_text=inp["ref_text"],
        x_vector_only_mode=True,
    )

    # Serialisér tensor til base64
    buf = io.BytesIO()
    torch.save(prompt, buf)
    return {
        "clone_prompt_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def _handle_design_voice(inp):
    """Voice design: VoiceDesign model → ref audio → Clone model → clone prompt.

    Begge modeller køres i ét handler-kald for at undgå 2 netværksroundtrips.
    """
    # Trin 1: Generér reference audio med VoiceDesign model
    design_model = _get_model(DESIGN_MODEL_ID)
    wavs, sr = design_model.generate_voice_design(
        text=inp["text"],
        language=inp.get("language", "English"),
        instruct=inp["instruct"],
    )
    wav = wavs[0]

    # Frigør VoiceDesign model VRAM
    if DESIGN_MODEL_ID in _models:
        del _models[DESIGN_MODEL_ID]
        torch.cuda.empty_cache()

    # Trin 2: Opret clone prompt fra den genererede stemme
    clone_model = _get_model(CLONE_MODEL_ID)
    clone_prompt = clone_model.create_voice_clone_prompt(
        ref_audio=(wav, sr),
        ref_text=inp["text"],
        x_vector_only_mode=True,
    )

    # Returnér både audio og clone prompt
    result = _serialize_audio(wav, sr)

    buf = io.BytesIO()
    torch.save(clone_prompt, buf)
    result["clone_prompt_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")

    return result


# ── RunPod entrypoint ──
runpod.serverless.start({"handler": handler})
