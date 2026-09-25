# 1:1 verification benchmark

Measures **EER**, **minDCF** and the calibrated operating threshold on a trial
list, and reports what the false-accept / false-reject rates actually are at any
threshold you care about.

Built for VoxCeleb1-O, but any trial list in the same format works — including,
eventually, one built from our own call recordings.

---

## Why run this

Three things it gives you that the three fixture clips cannot:

1. **A correctness check on the whole pipeline.** SpeechBrain publishes roughly
   **0.8–1.0% EER** for this ECAPA-TDNN model on VoxCeleb1-O. Land near that and
   the build, the model and the scoring are all verified end to end. Land at 15%
   and something is broken that you would otherwise never have noticed.
2. **A defensible threshold.** The EER threshold comes from 37,720 trials, not
   from the midpoint of two measurements.
3. **The telephony cost, measured properly.** Run the same trials at 16 kHz and
   at 8 kHz and the difference in EER quantifies the narrowband penalty on real
   data instead of on three clips.

---

## Getting the data

VoxCeleb1 is at <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/>. Registration
is required, and the terms are **research-only** — fine for evaluation, but flag
it for legal before anything ships.

You need two things:

- **`veri_test2.txt`** — the official cleaned trial list: 37,720 trials over 40
  held-out speakers, balanced target / nontarget.
- **The VoxCeleb1 test audio** — around 1 GB, extracting to `wav/id10270/...`.
  This is the *test* portion only. Do **not** download the ~39 GB dev set; you
  do not need it and it will not fit in Cloud Shell.

> **Cloud Shell storage.** `$HOME` is 5 GB and already holds the engine build
> and model (~150 MB). The zip plus its extracted copy will be tight, so delete
> the archive as soon as it extracts. If it does not fit, run this on a small
> GCE VM instead — the workload is CPU-only and finishes in well under an hour.

Sanity-check the layout before a long run:

```
head -2 veri_test2.txt
ls wav | head -3
```

The paths in the trial list must resolve relative to whatever you pass as
`--root`.

---

## Running it

```
pip install -r requirements.txt
```

**Always start with a subset.** A 2,000-trial run takes a few minutes and
catches path mistakes before you commit to the full list:

```
python eval_verification.py --trials veri_test2.txt --root wav --limit 2000
```

Then the two full conditions:

```
# native 16 kHz - the ceiling
python eval_verification.py --trials veri_test2.txt --root wav \
    --target-sr 16000 --out vox1_16k.json

# 8 kHz band-limited - the telephony proxy
python eval_verification.py --trials veri_test2.txt --root wav \
    --target-sr 8000 --out vox1_8k.json
```

Every unique utterance is embedded once and cached to `embeddings_<rate>.npz`,
checkpointed every 250 files. VoxCeleb1-O reuses 4,874 utterances across 37,720
trials, so this saves about 8x — and if a Cloud Shell session times out, rerun
the same command and it resumes from the cache.

Expect roughly **4,874 × your per-clip time**. At ~120 ms that is about 10
minutes per condition; on Cloud Shell's 2 shared vCPUs, budget longer.

---

## Reading the output

```
  EER                          1.032 %
  threshold at EER (distance)  0.2914   <- the calibrated value
  minDCF (p_target=0.01)       0.1183

  operating point                FAR %     FRR %
  our 0.13                       0.021     8.440
  engine default 0.25            0.415     1.902
```

- **EER** — the error rate where false accepts and false rejects are equal. One
  number for overall accuracy. Compare against SpeechBrain's published figure.
- **threshold at EER (distance)** — plug this into `VD_THRESHOLD` and you have a
  threshold with evidence behind it.
- **minDCF** — accuracy when genuine calls are rare (1% here) and both error
  types cost the same. EER treats the two errors as equally likely; minDCF does
  not, which is closer to a real contact centre.
- **The operating-point table** — what each threshold actually costs. This is
  the table the business needs in order to choose, because FAR and FRR trade
  directly against each other.

> The numbers above are an illustrative layout, not results. Run it and record
> your own.

---

## Caveats to carry into the report

- **VoxCeleb is not telephony, and not our callers.** 16 kHz YouTube interview
  audio. The 16 kHz EER is a **ceiling**, not a production forecast.
- **The model was trained on VoxCeleb-family data.** The test speakers are
  properly held out, so the number is honest — but it is in-domain, and real
  call audio is not.
- **The 8 kHz condition is a proxy, not a recording.** It band-limits clean
  audio; it does not add codec artefacts, packet loss or handset variation. Real
  Connect audio will be worse.
- **The EER threshold is not automatically the right threshold.** EER weights
  both errors equally. If a false accept is worse than a false reject for us,
  the operating point belongs somewhere else on the curve — a business decision,
  informed by the FAR/FRR table.
