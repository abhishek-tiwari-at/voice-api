"""Voice biometrics demo API over voice-detect.cpp.

Interactive docs (upload files and try every endpoint): http://HOST:PORT/docs
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

import config
from engine import AudioError, EngineError, VoiceEngine, cosine_distance, check_duration, decode_audio
from registry import RegistryError, SpeakerRegistry

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup and reuse it for every request. This is the
    # whole point of the C API's context object; loading per request would add
    # ~25 ms and a lot of memory churn.
    engine = VoiceEngine(config.LIB_PATH, config.MODEL_PATH)
    model_id = os.path.basename(config.MODEL_PATH)
    STATE["engine"] = engine
    STATE["registry"] = SpeakerRegistry(config.REGISTRY_PATH, model_id)
    STATE["model_id"] = model_id
    try:
        yield
    finally:
        engine.close()


app = FastAPI(
    title="Voice Biometrics Demo API",
    description=(
        "Speaker enrolment, verification and identification over the "
        "voice-detect.cpp engine. Threshold defaults to a calibrated value, NOT "
        "the engine's shipped 0.25, which false-accepts a different speaker."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _engine():
    if "engine" not in STATE:
        raise HTTPException(503, "engine not loaded")
    return STATE["engine"]


def _embed_upload(upload: UploadFile):
    """Read an upload -> (embedding, duration_seconds). Maps errors to HTTP 400."""
    data = upload.file.read()
    try:
        pcm, sr = decode_audio(data, upload.filename or "")
        seconds = check_duration(pcm, sr)
        return _engine().embed(pcm, sr), seconds, sr
    except AudioError as e:
        raise HTTPException(400, f"{upload.filename or 'upload'}: {e}")
    except EngineError as e:
        raise HTTPException(500, f"engine error: {e}")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", summary="Engine status and active configuration")
def health():
    eng = _engine()
    return {
        "status": "ok",
        "model": STATE["model_id"],
        "abi_version": eng.abi_version,
        "embedding_dim": eng.dim,          # null until the first embed
        "threshold": config.THRESHOLD,
        "identify_margin": config.IDENTIFY_MARGIN,
        "min_seconds": config.MIN_SECONDS,
        "enrolled_speakers": len(STATE["registry"].list()),
        "library": eng.lib_path,
    }


@app.get("/speakers", summary="List enrolled speakers")
def list_speakers():
    return {"speakers": STATE["registry"].list()}


@app.post("/speakers/{speaker_id}/enroll", summary="Enrol or update a speaker")
def enroll(
    speaker_id: str,
    files: list[UploadFile] = File(..., description="Two or more clips, ideally recorded separately"),
    replace: bool = Form(False, description="Discard any existing voiceprint instead of merging"),
):
    embeddings, details = [], []
    for f in files:
        emb, seconds, sr = _embed_upload(f)
        embeddings.append(emb)
        details.append({"filename": f.filename, "seconds": round(seconds, 2), "sample_rate": sr})

    try:
        result = STATE["registry"].enroll(speaker_id, embeddings, replace=replace)
    except RegistryError as e:
        raise HTTPException(400, str(e))

    warning = None
    if result["n_utterances"] < config.MIN_ENROLL_UTTERANCES:
        warning = (f"only {result['n_utterances']} utterance(s) enrolled; "
                   f"{config.MIN_ENROLL_UTTERANCES}+ recorded on separate occasions "
                   f"gives a far more stable voiceprint")
    return {**result, "utterances": details, "warning": warning}


@app.post("/verify", summary="1:1 - does this voice match the claimed speaker?")
def verify(
    speaker_id: str = Form(..., description="The identity the caller claims"),
    file: UploadFile = File(...),
    threshold: float = Form(None, description=f"Defaults to {config.THRESHOLD}"),
):
    emb, seconds, _ = _embed_upload(file)
    thr = config.THRESHOLD if threshold is None else float(threshold)
    result = STATE["registry"].verify(emb, speaker_id, thr)
    if result["decision"] == "not_enrolled":
        raise HTTPException(404, f"speaker '{speaker_id}' is not enrolled")
    return {**result, "seconds": round(seconds, 2),
            "contact_attributes": {
                "voiceAuthDecision": result["decision"],
                "voiceAuthScore": f"{result['score']:.4f}",
                "voiceAuthThreshold": str(thr),
                "voiceAuthModel": STATE["model_id"],
            }}


@app.post("/identify", summary="1:N - who is this? Returns no_match if nobody fits")
def identify(
    file: UploadFile = File(...),
    threshold: float = Form(None),
    margin: float = Form(None),
):
    emb, seconds, _ = _embed_upload(file)
    thr = config.THRESHOLD if threshold is None else float(threshold)
    mar = config.IDENTIFY_MARGIN if margin is None else float(margin)
    return {**STATE["registry"].identify(emb, thr, mar), "seconds": round(seconds, 2)}


@app.post("/compare", summary="Compare two clips directly, without the registry")
def compare(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    threshold: float = Form(None),
):
    emb_a, sec_a, _ = _embed_upload(file_a)
    emb_b, sec_b, _ = _embed_upload(file_b)
    thr = config.THRESHOLD if threshold is None else float(threshold)
    d = cosine_distance(emb_a, emb_b)
    return {
        "distance": round(d, 6),
        "score": round(1.0 - d, 6),
        "same_speaker": bool(d <= thr),
        "threshold": thr,
        "at_engine_default_0_25": bool(d <= 0.25),   # shows the false accept
        "seconds": {"a": round(sec_a, 2), "b": round(sec_b, 2)},
    }


@app.delete("/speakers/{speaker_id}", summary="Delete a voiceprint")
def delete_speaker(speaker_id: str):
    if not STATE["registry"].delete(speaker_id):
        raise HTTPException(404, f"speaker '{speaker_id}' is not enrolled")
    return {"deleted": speaker_id}
