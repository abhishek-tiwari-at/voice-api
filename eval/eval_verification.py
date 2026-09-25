#!/usr/bin/env python3
"""1:1 speaker-verification evaluation on a trial list (e.g. VoxCeleb1-O).

Trial-list format - one trial per line, whitespace separated, as VoxCeleb ships:

    1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav
    0 id10270/x6uYqmx31kE/00001.wav id10300/ize_eiCFEg0/00003.wav

    label 1 = target (same speaker), 0 = nontarget (impostor)

Each unique utterance is embedded ONCE and cached, because a trial list reuses
the same files many times over (VoxCeleb1-O: 37,720 trials, 4,874 utterances -
about 8x saved). Scoring the pairs afterwards is just dot products.

Examples
--------
    # quick sanity run on 2000 trials
    python eval_verification.py --trials veri_test2.txt --root wav --limit 2000

    # full run, native 16 kHz
    python eval_verification.py --trials veri_test2.txt --root wav --out vox1_16k.json

    # same trials band-limited to 8 kHz, the telephony proxy
    python eval_verification.py --trials veri_test2.txt --root wav \
        --target-sr 8000 --out vox1_8k.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                          # noqa: E402
from engine import VoiceEngine, decode_audio           # noqa: E402
from metrics import compute_eer, compute_min_dcf, rates_at_threshold  # noqa: E402


def parse_trials(path, limit=None):
    trials = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"{path}:{ln}: expected 'label pathA pathB', got {len(parts)} fields")
            label, a, b = parts
            if label not in ("0", "1"):
                raise ValueError(f"{path}:{ln}: label must be 0 or 1, got {label!r}")
            trials.append((int(label), a, b))
            if limit and len(trials) >= limit:
                break
    return trials


def band_limit(pcm, src_sr, target_sr):
    """Decimate to target_sr with an anti-alias filter, as a telephony codec
    would. The engine upsamples back to 16 kHz internally; the high frequencies
    are gone by then, which is the point of the experiment."""
    if target_sr == src_sr:
        return pcm, src_sr
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(target_sr), int(src_sr))
    return resample_poly(pcm, target_sr // g, src_sr // g).astype(np.float32), target_sr


def embed_all(engine, paths, root, target_sr, cache_path):
    """Embed every unique utterance once, caching to disk so a rerun is free."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as z:
            cache = {k: z[k] for k in z.files}
        print(f"  loaded {len(cache)} cached embeddings from {cache_path}")

    todo = [p for p in paths if p not in cache]
    print(f"  {len(paths)} unique utterances, {len(todo)} to embed")

    failures, t0 = {}, time.time()
    for i, rel in enumerate(todo, 1):
        full = os.path.join(root, rel)
        try:
            with open(full, "rb") as f:
                pcm, sr = decode_audio(f.read(), rel)
            pcm, sr = band_limit(pcm, sr, target_sr)
            cache[rel] = engine.embed(pcm, sr)
        except Exception as e:                      # short/corrupt files: skip, count
            failures[rel] = str(e)

        if i % 250 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / rate if rate else 0
            print(f"    {i}/{len(todo)}  {rate:.1f}/s  eta {eta/60:.1f} min"
                  f"{'  failures: ' + str(len(failures)) if failures else ''}", flush=True)
            if cache_path:
                np.savez(cache_path, **cache)       # checkpoint, so a timeout is not fatal

    if cache_path:
        np.savez(cache_path, **cache)
    return cache, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", required=True, help="trial list, e.g. veri_test2.txt")
    ap.add_argument("--root", required=True, help="folder the trial paths are relative to")
    ap.add_argument("--target-sr", type=int, default=16000,
                    help="band-limit every clip to this rate first (8000 = telephony proxy)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N trials")
    ap.add_argument("--cache", default=None, help="embedding cache .npz (default: auto-named)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--min-seconds", type=float, default=0.3,
                    help="reject clips shorter than this (crash floor is 0.065s)")
    args = ap.parse_args()

    # The API's 1.0s floor is a product decision; for benchmarking we keep only
    # the hard safety floor so short dataset clips are not silently dropped.
    config.MIN_SECONDS = max(args.min_seconds, config.ABSOLUTE_MIN_SECONDS)

    trials = parse_trials(args.trials, args.limit)
    n_tgt = sum(1 for t in trials if t[0] == 1)
    print(f"\ntrials: {len(trials)}  ({n_tgt} target, {len(trials)-n_tgt} nontarget)")
    print(f"condition: {args.target_sr} Hz")

    paths = sorted({p for _, a, b in trials for p in (a, b)})
    cache_path = args.cache or f"embeddings_{args.target_sr}.npz"

    engine = VoiceEngine(config.LIB_PATH, config.MODEL_PATH)
    print(f"model: {os.path.basename(config.MODEL_PATH)}")
    try:
        emb, failures = embed_all(engine, paths, args.root, args.target_sr, cache_path)
    finally:
        engine.close()

    scores, labels, skipped = [], [], 0
    for label, a, b in trials:
        if a not in emb or b not in emb:
            skipped += 1
            continue
        va, vb = emb[a], emb[b]
        scores.append(float(np.dot(va / np.linalg.norm(va), vb / np.linalg.norm(vb))))
        labels.append(label)

    if skipped:
        print(f"\n  WARNING: skipped {skipped} trials whose audio failed to embed")
    if not scores:
        sys.exit("no trials could be scored")

    eer, sim_thr, far_at, frr_at = compute_eer(scores, labels)
    mindcf, dcf_thr = compute_min_dcf(scores, labels, p_target=0.01)
    api_far, api_frr = rates_at_threshold(scores, labels, 1.0 - config.THRESHOLD)
    d_far, d_frr = rates_at_threshold(scores, labels, 1.0 - 0.25)   # engine default

    res = {
        "trials_scored": len(scores),
        "targets": int(sum(labels)),
        "nontargets": len(labels) - int(sum(labels)),
        "skipped_trials": skipped,
        "failed_utterances": len(failures),
        "condition_hz": args.target_sr,
        "model": os.path.basename(config.MODEL_PATH),
        "eer_percent": round(eer * 100, 4),
        "eer_threshold_similarity": round(sim_thr, 6),
        "eer_threshold_distance": round(1.0 - sim_thr, 6),
        "min_dcf_p01": round(mindcf, 4),
        "at_api_threshold_0_13": {"distance": config.THRESHOLD,
                                  "far_percent": round(api_far * 100, 4),
                                  "frr_percent": round(api_frr * 100, 4)},
        "at_engine_default_0_25": {"distance": 0.25,
                                   "far_percent": round(d_far * 100, 4),
                                   "frr_percent": round(d_frr * 100, 4)},
    }

    print("\n" + "=" * 62)
    print(f"RESULTS  ({args.target_sr} Hz, {len(scores)} trials)")
    print("=" * 62)
    print(f"  EER                          {res['eer_percent']:.3f} %")
    print(f"  threshold at EER (distance)  {res['eer_threshold_distance']:.4f}"
          "   <- the calibrated value")
    print(f"  minDCF (p_target=0.01)       {res['min_dcf_p01']:.4f}")
    print()
    print(f"  {'operating point':<26}{'FAR %':>10}{'FRR %':>10}")
    print(f"  {'our 0.13':<26}{res['at_api_threshold_0_13']['far_percent']:>10.3f}"
          f"{res['at_api_threshold_0_13']['frr_percent']:>10.3f}")
    print(f"  {'engine default 0.25':<26}{res['at_engine_default_0_25']['far_percent']:>10.3f}"
          f"{res['at_engine_default_0_25']['frr_percent']:>10.3f}")
    print("=" * 62)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")
    if failures:
        print(f"\n{len(failures)} utterances failed; first few:")
        for k, v in list(failures.items())[:5]:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
