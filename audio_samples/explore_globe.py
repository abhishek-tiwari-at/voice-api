#!/usr/bin/env python3
"""Explore the GLOBE_V2 dataset and export audio samples as one WAV per speaker.

The parquet files in data/ are Git LFS pointers; any shard that is still a
pointer is downloaded from the Hugging Face Hub (into ~/.cache/huggingface).

Examples:
    python explore_globe.py                                  # test shard 0, 10 speakers
    python explore_globe.py --shards test-00001-of-00002 --num-speakers 20
    python explore_globe.py --clips-per-speaker 5 --split    # also keep individual clips
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

REPO_ID = "MushanW/GLOBE_V2"
DATA_DIR = Path(__file__).parent / "data"
META_COLS = ["speaker_id", "transcript", "accent", "duration", "age", "gender"]


def resolve_shard(name):
    """Return a local path to a real parquet file, downloading if it is an LFS pointer."""
    local = DATA_DIR / f"{name}.parquet"
    if local.exists() and local.stat().st_size > 1024:
        return local
    from huggingface_hub import hf_hub_download

    print(f"Downloading {name}.parquet from the Hub (LFS pointer only locally)...")
    return Path(hf_hub_download(REPO_ID, f"data/{name}.parquet", repo_type="dataset"))


def explore(df):
    print("\n=== Overview ===")
    print(f"Utterances: {len(df):,}")
    print(f"Speakers:   {df.speaker_id.nunique():,}")
    print(f"Accents:    {df.accent.nunique():,}")
    print(f"Total audio: {df.duration.sum() / 3600:.2f} h")
    print("\nDuration (s):")
    print(df.duration.describe().round(2).to_string())
    for col in ["gender", "age", "accent"]:
        print(f"\n--- Utterances by {col} (top 15) ---")
        print(df[col].value_counts().head(15).to_string())
    per_spk = df.groupby("speaker_id").size()
    print("\nUtterances per speaker:")
    print(per_spk.describe().round(2).to_string())
    print("\nSample rows:")
    print(df[META_COLS].head(5).to_string())


def pick_speakers(df, n, seed):
    """Pick n speakers, preferring distinct accents so the samples are varied."""
    spk = df.groupby("speaker_id").agg(accent=("accent", "first")).reset_index()
    spk = spk.sample(frac=1, random_state=seed)
    chosen = spk.drop_duplicates("accent").head(n)
    if len(chosen) < n:
        rest = spk[~spk.speaker_id.isin(chosen.speaker_id)]
        chosen = pd.concat([chosen, rest.head(n - len(chosen))])
    return chosen.speaker_id.tolist()


def decode(audio):
    wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    return wav, sr


def export(path, df, speakers, clips_per_speaker, out_dir, gap_s, split, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = (
        df[df.speaker_id.isin(speakers)]
        .sample(frac=1, random_state=seed)
        .groupby("speaker_id")
        .head(clips_per_speaker)
    )
    # Only load the audio column for the rows we need.
    audio_col = pq.read_table(path, columns=["audio"]).column("audio")

    manifest = []
    for spk, rows in selected.groupby("speaker_id"):
        info = rows.iloc[0]
        chunks, sr = [], None
        for i, (idx, row) in enumerate(rows.iterrows()):
            wav, clip_sr = decode(audio_col[idx].as_py())
            if sr is None:
                sr = clip_sr
            elif clip_sr != sr:
                raise ValueError(f"{spk}: mixed sample rates {sr} and {clip_sr}")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if split:
                clip_path = out_dir / spk / f"{spk}_{i:02d}.wav"
                clip_path.parent.mkdir(exist_ok=True)
                sf.write(clip_path, wav, sr, subtype="PCM_16")
            chunks += [wav, np.zeros(int(gap_s * sr), dtype=np.float32)]
            manifest.append({**row[META_COLS].to_dict(), "clip_index": i})

        combined = np.concatenate(chunks[:-1])
        wav_path = out_dir / f"{spk}.wav"
        sf.write(wav_path, combined, sr, subtype="PCM_16")
        print(
            f"  {wav_path.name}: {len(rows)} clips, {len(combined) / sr:.1f}s @ {sr} Hz"
            f" | {info.gender}, {info.age}, {info.accent}"
        )

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(f"\nWrote {len(speakers)} speaker WAVs + manifest.csv to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", default=["test-00000-of-00002"],
                    help="shard names without .parquet (see data/)")
    ap.add_argument("--num-speakers", type=int, default=10)
    ap.add_argument("--clips-per-speaker", type=int, default=3)
    ap.add_argument("--gap", type=float, default=0.5, help="silence between clips (s)")
    ap.add_argument("--split", action="store_true",
                    help="also save each clip separately under <out>/<speaker_id>/")
    ap.add_argument("--out", type=Path, default=Path("samples"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for name in args.shards:
        path = resolve_shard(name)
        print(f"\n##### {name} ({path}) #####")
        df = pq.read_table(path, columns=META_COLS).to_pandas()
        explore(df)
        speakers = pick_speakers(df, args.num_speakers, args.seed)
        print(f"\n=== Exporting {len(speakers)} speakers ===")
        out = args.out / name if len(args.shards) > 1 else args.out
        export(path, df, speakers, args.clips_per_speaker, out, args.gap, args.split, args.seed)


if __name__ == "__main__":
    main()
