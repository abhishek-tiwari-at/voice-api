"""Simulated phone call against the voice biometrics API, using the laptop mic.

Plays the part of the IVR: asks for a customer id (as if resolved from the
caller's number), records the caller, and sends each clip to POST /call, which
enrols a first-time caller or runs 1:1 verify + 1:N identify for a known one.

    python call_sim.py                         # interactive, records from the mic
    python call_sim.py --caller abhishek       # skip the id prompt
    python call_sim.py --caller abhishek --file me_test.wav   # replay a file instead

Audio is captured at 8 kHz mono by default, i.e. telephone bandwidth, so the
demo hears what a real call would. Use --rate 16000 for wideband.

Recording uses `arecord` (Linux/ALSA) when present, otherwise ffmpeg
(pulse on Linux, avfoundation on macOS, dshow on Windows - Windows needs
--device "audio=<name from: ffmpeg -list_devices true -f dshow -i dummy>").
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile

import requests

WORDS = ("river tiger copper harbour lantern meadow cobalt thunder velvet "
         "orchard falcon granite silver maple ember canyon").split()

ROUTE_TEXT = {
    "self_service":       "Caller authenticated -> straight to self-service, no security questions",
    "agent_with_kba":     "Route to agent -> agent asks knowledge-based security questions",
    "agent_fraud_review": "Route to fraud desk -> voice belongs to a different enrolled customer",
    "collect_more_audio": "Keep the caller talking -> more audio needed to finish enrolment",
}


def record(path, seconds, rate, device):
    if shutil.which("arecord") and not device:
        cmd = ["arecord", "-q", "-f", "S16_LE", "-c", "1", "-r", str(rate),
               "-d", str(seconds), path]
    elif shutil.which("ffmpeg"):
        if sys.platform == "darwin":
            src = ["-f", "avfoundation", "-i", device or ":0"]
        elif sys.platform == "win32":
            if not device:
                sys.exit('Windows needs --device "audio=<microphone name>"')
            src = ["-f", "dshow", "-i", device]
        else:
            src = ["-f", "pulse", "-i", device or "default"]
        cmd = ["ffmpeg", "-loglevel", "error", "-nostdin", "-y", *src, "-t", str(seconds),
               "-ac", "1", "-ar", str(rate), "-sample_fmt", "s16", path]
    else:
        sys.exit("Need arecord or ffmpeg to record from the microphone.")
    subprocess.run(cmd, check=True)


def send(api, caller, path):
    with open(path, "rb") as f:
        r = requests.post(f"{api}/call", data={"caller_id": caller},
                          files={"file": (os.path.basename(path), f)}, timeout=60)
    body = r.json()
    if r.status_code != 200:
        raise RuntimeError(body.get("detail", body))
    return body


def show(d):
    print()
    if d["mode"] == "enrolment":
        if d["outcome"] == "duplicate_voice":
            print(f"  !! This voice is already enrolled as '{d['matches_existing']}' "
                  f"(distance {d['distance']:.4f}). Clip NOT stored.")
        else:
            print(f"  ENROLMENT  {d['n_utterances']}/{d['needed']} clips stored "
                  f"({d['seconds']} s of speech)")
    else:
        v, i = d["verify"], d["identify"]
        print(f"  OUTCOME    {d['outcome'].upper()}")
        print(f"  1:1 verify   claimed '{d['caller_id']}': {v['decision']}  "
              f"distance {v['distance']:.4f}  (accept if <= {v['threshold']})")
        who = i["speaker_id"] or f"{i['decision']} (closest: {i.get('best_candidate')})"
        print(f"  1:N identify best match: {who}  distance {i['distance']:.4f}")
        for c in i["candidates"]:
            print(f"               - {c['speaker_id']:<20} {c['distance']:.4f}")
    print(f"  ROUTING    {ROUTE_TEXT.get(d['routing'], d['routing'])}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--caller", help="customer id (prompted if omitted)")
    ap.add_argument("--seconds", type=int, default=6, help="length of each recording")
    ap.add_argument("--rate", type=int, default=8000, help="8000 = telephone, 16000 = wideband")
    ap.add_argument("--device", help="ffmpeg input device (see module docstring)")
    ap.add_argument("--file", help="send this audio file instead of recording")
    a = ap.parse_args()

    try:
        h = requests.get(f"{a.api}/health", timeout=5).json()
    except requests.RequestException:
        sys.exit(f"API not reachable at {a.api}. Start it: python -m uvicorn main:app --port 8000")
    print(f"Connected: model {h['model']}, threshold {h['threshold']}, "
          f"{h['enrolled_speakers']} speaker(s) enrolled")

    caller = a.caller or input("\n[IVR] Incoming call. Customer id: ").strip()
    if not caller:
        sys.exit("No customer id.")
    known = {s["speaker_id"]: s["n_utterances"] for s in requests.get(f"{a.api}/speakers").json()["speakers"]}
    print(f"[IVR] {caller}: " + (f"voiceprint on file ({known[caller]} clips)" if caller in known
                                 else "no voiceprint on file -> enrolment"))

    tmp = tempfile.mkdtemp(prefix="callsim_")
    turn = 0
    while True:
        turn += 1
        if a.file:
            path = a.file
        else:
            phrase = "  ".join(random.sample(WORDS, 4))
            print(f"\n[IVR] Please say, then keep talking naturally for {a.seconds} s "
                  f"(name, why you're calling, ...):\n\n        \"{phrase}\"\n")
            input("      Press Enter to start speaking...")
            path = os.path.join(tmp, f"turn{turn}.wav")
            print("      Recording...", flush=True)
            record(path, a.seconds, a.rate, a.device)
        try:
            d = send(a.api, caller, path)
        except RuntimeError as e:
            print(f"\n  Clip rejected by the API: {e}\n  Try again, speaking for longer.")
            if a.file:
                return 1
            continue
        show(d)
        # A first-time caller keeps recording until the voiceprint is complete.
        if d["outcome"] != "enrolment_in_progress" or a.file:
            break

    print("[IVR] Call ended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
