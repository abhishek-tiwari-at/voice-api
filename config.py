"""Configuration, all overridable by environment variable.

Nothing here is hard-coded to one machine: the same files run on Linux,
Cloud Shell and Windows as long as VD_LIB and VD_MODEL point at the build.
"""
import os
import sys

# --- paths to the built engine -------------------------------------------------
# Defaults assume this folder sits NEXT TO the voice-detect.cpp checkout:
#     parent/
#       voice-detect.cpp/     <- built repo
#       voice-api/            <- this folder
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..", "voice-detect.cpp")

_LIB_NAMES = (["voicedetect.dll", "libvoicedetect.dll"] if sys.platform == "win32"
              else ["libvoicedetect.so"])
_LIB_DIRS = (["build/Release", "build/bin/Release", "build"] if sys.platform == "win32"
             else ["build", "build-shared"])


def _autodetect_lib():
    for d in _LIB_DIRS:
        for n in _LIB_NAMES:
            p = os.path.join(_REPO, d, n)
            if os.path.exists(p):
                return os.path.abspath(p)
    return ""


LIB_PATH   = os.environ.get("VD_LIB")   or _autodetect_lib()
MODEL_PATH = os.environ.get("VD_MODEL") or os.path.abspath(
    os.path.join(_REPO, "models", "ecapa-tdnn-voxceleb.gguf"))
REGISTRY_PATH = os.environ.get("VD_REGISTRY", os.path.join(_HERE, "registry.json"))

# --- decision thresholds -------------------------------------------------------
# NOT the engine's shipped default of 0.25. That value false-accepts a different
# speaker on the project's own labelled test clips. 0.13 separates that pair, but
# it is a PoC value derived from 3 clips: calibrate on real call audio before any
# production use.
THRESHOLD = float(os.environ.get("VD_THRESHOLD", "0.13"))

# Margin the best match must beat the runner-up by during 1:N identify, so two
# near-tied candidates return "ambiguous" rather than a coin-flip.
IDENTIFY_MARGIN = float(os.environ.get("VD_IDENTIFY_MARGIN", "0.02"))

# --- audio gating --------------------------------------------------------------
# HARD FLOOR: audio yielding fewer than 5 feature frames aborts the process with
# a ggml assertion (SIGABRT) that no try/except can catch. 5 frames = 400 + 4*160
# samples = 1040 @ 16 kHz = 65 ms. Never call the engine below this.
ABSOLUTE_MIN_SECONDS = 0.065
# PRACTICAL floor for a trustworthy decision. Short clips give unstable
# embeddings even when they do not crash.
MIN_SECONDS = float(os.environ.get("VD_MIN_SECONDS", "1.0"))
MAX_SECONDS = float(os.environ.get("VD_MAX_SECONDS", "30.0"))

MIN_ENROLL_UTTERANCES = int(os.environ.get("VD_MIN_ENROLL_UTTERANCES", "3"))
