"""Thin, thread-safe wrapper around libvoicedetect + audio decoding."""
import ctypes
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import wave

import numpy as np

import config


class AudioError(ValueError):
    """Input audio was unusable (wrong format, too short, too long)."""


class EngineError(RuntimeError):
    """The engine refused a request."""


# --------------------------------------------------------------------------- audio
def _read_wav_bytes(data: bytes):
    """Decode 16-bit PCM WAV with the standard library. Returns (float32 mono, sr)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2:
            raise AudioError("WAV must be 16-bit PCM")
        sr, n_ch, n_frames = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n_frames)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1)
    return pcm, sr


def _ffmpeg_decode(data: bytes, suffix: str):
    """Convert anything ffmpeg understands (mp3, m4a, webm, ogg, float WAV...)
    into 16 kHz mono 16-bit PCM. Only used when the stdlib reader cannot cope."""
    if not shutil.which("ffmpeg"):
        raise AudioError(
            "Unsupported audio format and ffmpeg is not installed. Either upload "
            "16-bit PCM WAV, or install ffmpeg on the server."
        )
    src = dst = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as f:
            f.write(data)
            src = f.name
        dst = src + ".conv.wav"
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", src,
             "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", dst],
            capture_output=True, timeout=60,
        )
        if proc.returncode != 0 or not os.path.exists(dst):
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
            raise AudioError(f"ffmpeg could not decode this file. {' '.join(tail)}")
        with open(dst, "rb") as f:
            return _read_wav_bytes(f.read())
    finally:
        for p in (src, dst):
            if p and os.path.exists(p):
                os.unlink(p)


def decode_audio(data: bytes, filename: str = ""):
    """Bytes from an upload -> (float32 mono PCM, sample_rate). Raises AudioError."""
    if not data:
        raise AudioError("empty upload")
    try:
        return _read_wav_bytes(data)
    except AudioError:
        raise
    except Exception:
        return _ffmpeg_decode(data, os.path.splitext(filename)[1].lower())


def check_duration(pcm: np.ndarray, sr: int) -> float:
    """Enforce the crash floor and the practical floor. Returns duration in seconds."""
    seconds = len(pcm) / float(sr) if sr else 0.0
    if seconds < config.ABSOLUTE_MIN_SECONDS:
        raise AudioError(
            f"audio is {seconds:.3f}s; below {config.ABSOLUTE_MIN_SECONDS}s the "
            f"engine aborts the process, so it is refused here")
    if seconds < config.MIN_SECONDS:
        raise AudioError(
            f"audio is {seconds:.2f}s of speech; need at least "
            f"{config.MIN_SECONDS:.2f}s for a reliable decision")
    if seconds > config.MAX_SECONDS:
        raise AudioError(f"audio is {seconds:.1f}s; maximum is {config.MAX_SECONDS:.0f}s")
    return seconds


# -------------------------------------------------------------------------- engine
class VoiceEngine:
    """Loads the model once and serialises access to it.

    The engine's thread-safety under concurrent calls is not documented or
    verified, and its per-context error buffer is shared mutable state, so every
    call is taken under a lock. For a demo API that costs nothing; a production
    service would run several isolated worker processes instead.
    """

    def __init__(self, lib_path: str, model_path: str):
        if not lib_path or not os.path.exists(lib_path):
            raise EngineError(f"shared library not found: {lib_path or '(unset)'}")
        if not os.path.exists(model_path):
            raise EngineError(f"model not found: {model_path}")

        lib_path = os.path.abspath(lib_path)
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            # Windows has no RPATH; the ggml DLLs sit beside voicedetect.dll.
            os.add_dll_directory(os.path.dirname(lib_path))
        self._lib = ctypes.CDLL(lib_path)
        self._declare()

        self.abi_version = self._lib.voicedetect_capi_abi_version()
        self._ctx = self._lib.voicedetect_capi_load(model_path.encode())
        if not self._ctx:
            raise EngineError(f"engine failed to load model: {model_path}")

        self._lock = threading.Lock()
        self.model_path = model_path
        self.lib_path = lib_path
        self.dim = None  # filled after the first embed

    def _declare(self):
        L = self._lib
        L.voicedetect_capi_abi_version.restype = ctypes.c_int
        L.voicedetect_capi_load.restype = ctypes.c_void_p
        L.voicedetect_capi_load.argtypes = [ctypes.c_char_p]
        L.voicedetect_capi_free.argtypes = [ctypes.c_void_p]
        L.voicedetect_capi_last_error.restype = ctypes.c_char_p
        L.voicedetect_capi_last_error.argtypes = [ctypes.c_void_p]
        L.voicedetect_capi_free_vec.argtypes = [ctypes.POINTER(ctypes.c_float)]
        L.voicedetect_capi_embed_pcm.restype = ctypes.c_int
        L.voicedetect_capi_embed_pcm.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_int)]

    def embed(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        """Mono float PCM -> L2-normalised speaker embedding."""
        check_duration(pcm, sample_rate)          # crash guard, before the FFI call
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        vec = ctypes.POINTER(ctypes.c_float)()
        dim = ctypes.c_int(0)
        with self._lock:
            rc = self._lib.voicedetect_capi_embed_pcm(
                self._ctx, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(pcm), int(sample_rate), ctypes.byref(vec), ctypes.byref(dim))
            if rc != 0:
                err = self._lib.voicedetect_capi_last_error(self._ctx)
                raise EngineError(err.decode() if err else "embed failed")
            out = np.ctypeslib.as_array(vec, shape=(dim.value,)).copy()
            self._lib.voicedetect_capi_free_vec(vec)
        self.dim = int(dim.value)
        return out

    def close(self):
        if getattr(self, "_ctx", None):
            self._lib.voicedetect_capi_free(ctypes.c_void_p(self._ctx))
            self._ctx = None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. Re-normalises: an averaged centroid is not unit length."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(1.0 - float(np.dot(a, b)))
