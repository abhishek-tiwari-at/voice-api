# Voice biometrics API, self-contained: the image clones and builds the
# voice-detect.cpp engine and downloads the model, so nothing has to sit
# beside this folder on the host.
#
#   docker build -t voice-api .
#   docker run -d --name voice-api -p 8000:8000 -v voice-data:/data voice-api
#
# Voiceprints are written to /data/registry.json; keep /data on a volume or
# every enrolment is lost when the container is replaced.

# Both stages share one base so the engine is built against the same glibc and
# libstdc++ it runs with.
ARG BASE=python:3.12-slim-bookworm

# ---------------------------------------------------------------------------
# engine: clone, build libvoicedetect.so, fetch the model
# ---------------------------------------------------------------------------
FROM ${BASE} AS engine

# Pinned to the engine commit and model this API was tested against. Change
# them deliberately; a moved engine or model can shift the scores the 0.13
# threshold was chosen for.
ARG VD_REPO=https://github.com/localai-org/voice-detect.cpp.git
ARG VD_REF=1db1759572c90faef6f3a78c36b5941a096a9f89
ARG MODEL_URL=https://huggingface.co/mudler/voice-detect-gguf/resolve/main/ecapa-tdnn-voxceleb.gguf
ARG MODEL_SHA256=68046a1fdfb7843f460962db4739fbd381cc5c3ab93d1505e75e2f4c0dc19b8f

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone "${VD_REPO}" voice-detect.cpp \
    && cd voice-detect.cpp \
    && git checkout -q "${VD_REF}" \
    && git submodule update --init --recursive --depth 1

# GGML_NATIVE=OFF: the image must run on CPUs other than the build machine's.
RUN cd voice-detect.cpp \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DVOICEDETECT_SHARED=ON \
        -DVOICEDETECT_BUILD_CLI=OFF \
        -DVOICEDETECT_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)"

# libvoicedetect.so plus the ggml libraries it links against.
RUN mkdir -p /opt/voicedetect/lib /opt/voicedetect/models \
    && find voice-detect.cpp/build -name '*.so*' -exec cp -a {} /opt/voicedetect/lib/ \;

RUN curl -fL --retry 3 -o /opt/voicedetect/models/ecapa-tdnn-voxceleb.gguf "${MODEL_URL}" \
    && echo "${MODEL_SHA256}  /opt/voicedetect/models/ecapa-tdnn-voxceleb.gguf" | sha256sum -c -

# ---------------------------------------------------------------------------
# runtime: Python API + engine libraries + model
# ---------------------------------------------------------------------------
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# ffmpeg converts browser/phone audio (webm, m4a, mp3) to 16 kHz WAV.
# libgomp1 is the OpenMP runtime ggml's CPU backend links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=engine /opt/voicedetect /opt/voicedetect
RUN echo /opt/voicedetect/lib > /etc/ld.so.conf.d/voicedetect.conf && ldconfig

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py engine.py registry.py main.py demo.html ./

# Run unprivileged; /data is the only writable path the API needs.
RUN useradd --system --uid 10001 --home /app voiceapi \
    && mkdir -p /data && chown voiceapi /data
USER voiceapi

ENV VD_LIB=/opt/voicedetect/lib/libvoicedetect.so \
    VD_MODEL=/opt/voicedetect/models/ecapa-tdnn-voxceleb.gguf \
    VD_REGISTRY=/data/registry.json
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
