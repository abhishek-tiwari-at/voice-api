# Voice Biometrics Demo API

A small FastAPI service over the [voice-detect.cpp](https://github.com/localai-org/voice-detect.cpp)
speaker-recognition engine. It adds the layer the engine does not ship: a speaker
registry, 1:N identification, threshold control and audio validation.

Built for demonstration and testing. Not production code — see **Known limits**.

---

## 1. Prerequisites

You need the engine built and a model downloaded **before** this API will start.

```
parent-folder/
├── voice-detect.cpp/          <- cloned and built
│   ├── build/                 <- libvoicedetect.so / voicedetect.dll
│   └── models/
│       └── ecapa-tdnn-voxceleb.gguf
└── voice-api/                 <- this folder
```

With that layout the API finds everything automatically. Anywhere else, set
`VD_LIB` and `VD_MODEL` (see section 4).

**Build the engine** (Linux / WSL / Cloud Shell):

```
git clone --recursive https://github.com/localai-org/voice-detect.cpp
cd voice-detect.cpp
cmake -B build -DVOICEDETECT_BUILD_TESTS=ON -DVOICEDETECT_SHARED=ON -DGGML_NATIVE=OFF
cmake --build build -j
curl -fL --create-dirs -o models/ecapa-tdnn-voxceleb.gguf https://huggingface.co/mudler/voice-detect-gguf/resolve/main/ecapa-tdnn-voxceleb.gguf
```

On native Windows use `cmake --build build --config Release --parallel` instead;
the DLLs land in `build\Release\`.

**Optional but recommended: ffmpeg.** Without it the API accepts only 16-bit PCM
WAV. With it, any format (m4a, mp3, webm, ogg) is converted automatically.

---

## 2. Install and run

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
.venv/bin/python -m uvicorn main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** — interactive documentation where you can
upload audio files and call every endpoint from the browser. That is the easiest
way to demo this; no curl needed.

Check it came up correctly:

```
curl http://127.0.0.1:8000/health
```

---

## 3. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Engine status, model, active thresholds |
| `GET` | `/speakers` | List enrolled speakers |
| `POST` | `/speakers/{id}/enroll` | Enrol or update a voiceprint from one or more clips |
| `POST` | `/verify` | 1:1 — does this voice match the claimed speaker? |
| `POST` | `/identify` | 1:N — who is this? Returns `no_match` if nobody fits |
| `POST` | `/compare` | Compare two clips directly, no registry involved |
| `DELETE` | `/speakers/{id}` | Delete a voiceprint |

`/verify` also returns a `contact_attributes` block in the flat string form an
Amazon Connect contact flow consumes.

`/compare` returns `at_engine_default_0_25`, which shows whether the engine's own
shipped threshold would have accepted the pair. That is how you demonstrate the
false-accept finding live.

---

## 4. Configuration

All optional, all environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `VD_LIB` | auto-detected | Path to `libvoicedetect.so` / `voicedetect.dll` |
| `VD_MODEL` | auto-detected | Path to the `.gguf` model |
| `VD_REGISTRY` | `./registry.json` | Where voiceprints are stored |
| `VD_THRESHOLD` | `0.13` | Accept if cosine distance ≤ this |
| `VD_IDENTIFY_MARGIN` | `0.02` | Best match must beat runner-up by this |
| `VD_MIN_SECONDS` | `1.0` | Minimum speech for a decision |
| `VD_MAX_SECONDS` | `30.0` | Maximum accepted clip length |

**On the threshold.** The default here is `0.13`, **not** the engine's shipped
`0.25`. At `0.25` the engine accepts a different speaker on its own labelled test
clips — a false accept. `0.13` separates that pair, but it was derived from three
clips and is a demo value. Calibrate on real call recordings before trusting it.

---

## 5. Recording your own voice samples

**What to record.** For a demo that proves something, you need three sets:

1. **Enrolment — you, 3+ clips.** 5–10 seconds each, different sentences each
   time. Record them on separate occasions if you can; back-to-back clips share
   the same throat, mic position and room, which flatters the result.
2. **Test — you, 1–2 clips.** Recorded separately from enrolment, saying
   something different. This is your true-accept test.
3. **Impostor — someone else, 1+ clip.** A colleague or family member. This is
   the true-reject test, and **without it the demo proves nothing** — a system
   that accepts everyone passes a test that only uses your own voice.

**How to record on Windows.**

- *Voice Recorder* app → produces `.m4a` → the API converts it if ffmpeg is
  installed.
- *Audacity* → `File > Export > Export as WAV`, set **16-bit PCM**, mono. This
  path needs no ffmpeg.

**Convert with ffmpeg** if you prefer to prepare files yourself:

```
ffmpeg -i myvoice.m4a -ac 1 -ar 16000 -sample_fmt s16 myvoice.wav
```

**Prefer lossless WAV for enrolment.** Lossy compression shifts the embedding
slightly — the same clip measured 0.0165 as WAV and 0.0241 as MP3 against the
same voiceprint. Harmless here, but avoid baking compression artefacts into a
stored voiceprint.

**Practical tips.** Speak naturally and continuously; a clip that is mostly
silence fails the length check even if the file is long. Use the same microphone
for enrolment and testing where possible — channel mismatch is a real source of
false rejects. Keep the room quiet.

---

## 6. A demo that actually proves something

```
# 1. enrol yourself from three clips
curl -X POST http://127.0.0.1:8000/speakers/abhishek/enroll -F "files=@me1.wav" -F "files=@me2.wav" -F "files=@me3.wav"

# 2. verify with your own voice -> expect accept
curl -X POST http://127.0.0.1:8000/verify -F "speaker_id=abhishek" -F "file=@me_test.wav"

# 3. verify with someone else claiming to be you -> expect reject
curl -X POST http://127.0.0.1:8000/verify -F "speaker_id=abhishek" -F "file=@colleague.wav"

# 4. identify an unenrolled person -> expect no_match, not a wrong name
curl -X POST http://127.0.0.1:8000/identify -F "file=@colleague.wav"

# 5. show the engine's shipped default would have accepted the impostor
curl -X POST http://127.0.0.1:8000/compare -F "file_a=@me_test.wav" -F "file_b=@colleague.wav"
```

Step 5 is the interesting one. Look at `same_speaker` (false, at the calibrated
threshold) against `at_engine_default_0_25` — if that comes back `true`, you have
reproduced the false-accept finding with your own voice.

---

## 7. Known limits

Be upfront about these when demonstrating.

- **Calls are serialised.** The engine's thread-safety under concurrency is not
  documented or verified, so every call is taken under a lock. Throughput is one
  request at a time. A production service would run isolated worker processes.
- **Audio under 65 ms aborts the engine.** Fewer than 5 feature frames trips a
  ggml assertion that calls `abort()`, which no `try/except` can catch — it kills
  the whole process. The API refuses anything that short before calling the
  engine, but the underlying hazard is why input gating is mandatory, not
  cosmetic.
- **No liveness or anti-spoofing.** A recording of an enrolled speaker will be
  accepted. Speaker recognition is built to accept a matching voice; it is not a
  defence against replay or cloned audio.
- **The registry is a JSON file.** Fine for a demo, wrong for production, where
  it would be DynamoDB or similar.
- **Voiceprints are biometric personal data.** `registry.json` is gitignored for
  that reason. Use consenting volunteers for the demo and delete the data
  afterwards via `DELETE /speakers/{id}`.
- **Thresholds are uncalibrated.** See section 4.
