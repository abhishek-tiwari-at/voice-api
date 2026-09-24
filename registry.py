"""Persistent speaker registry - the layer voice-detect.cpp does not ship.

The engine's README advertises an `identify` operation against a registry, but
no such function exists in its source. That is fine: a registry is application
state. In production this becomes DynamoDB; here it is a JSON file.
"""
import json
import os
import threading
from datetime import datetime, timezone

import numpy as np

from engine import cosine_distance


class RegistryError(RuntimeError):
    pass


class SpeakerRegistry:
    def __init__(self, path: str, model_id: str):
        self.path = path
        self.model_id = model_id      # embeddings are NOT portable across models
        self._lock = threading.Lock()
        self._speakers = {}
        self._load()

    # ------------------------------------------------------------------ storage
    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            blob = json.load(f)
        stored = blob.get("model_id")
        if stored and stored != self.model_id:
            raise RegistryError(
                f"registry at {self.path} was built with model '{stored}' but the "
                f"engine is running '{self.model_id}'. Voiceprints are not portable "
                f"between models - delete the file and re-enrol.")
        for sid, rec in blob.get("speakers", {}).items():
            rec["embedding"] = np.array(rec["embedding"], dtype=np.float32)
            self._speakers[sid] = rec

    def _save(self):
        blob = {
            "model_id": self.model_id,
            "speakers": {
                sid: {**rec, "embedding": rec["embedding"].tolist()}
                for sid, rec in self._speakers.items()
            },
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(blob, f, indent=2)
        os.replace(tmp, self.path)     # atomic, so a crash cannot truncate it

    # ------------------------------------------------------------------ mutation
    def enroll(self, speaker_id: str, embeddings, replace: bool = False):
        """Average N utterance embeddings into one voiceprint, then RE-NORMALISE.

        Re-normalising matters: the mean of unit vectors is not a unit vector, so
        skipping it quietly biases every later cosine score."""
        if not embeddings:
            raise RegistryError("no usable utterances to enrol")
        with self._lock:
            existing = self._speakers.get(speaker_id)
            vectors = list(embeddings)
            n_prev = 0
            if existing is not None and not replace:
                # weight the stored centroid by how many utterances produced it
                n_prev = int(existing.get("n_utterances", 1))
                vectors = [existing["embedding"] * n_prev] + vectors

            centroid = np.sum(np.stack(vectors), axis=0) / (n_prev + len(embeddings))
            centroid = centroid / np.linalg.norm(centroid)

            rec = {
                "embedding": centroid.astype(np.float32),
                "n_utterances": n_prev + len(embeddings),
                "enrolled_at": (existing or {}).get(
                    "enrolled_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._speakers[speaker_id] = rec
            self._save()
            return {"speaker_id": speaker_id,
                    "n_utterances": rec["n_utterances"],
                    "enrolled_at": rec["enrolled_at"],
                    "updated_at": rec["updated_at"]}

    def delete(self, speaker_id: str) -> bool:
        """Deleting biometric data on request is a legal obligation, not a nicety."""
        with self._lock:
            if speaker_id not in self._speakers:
                return False
            del self._speakers[speaker_id]
            self._save()
            return True

    # ------------------------------------------------------------------- queries
    def list(self):
        with self._lock:
            return [{"speaker_id": s,
                     "n_utterances": r["n_utterances"],
                     "enrolled_at": r["enrolled_at"],
                     "updated_at": r["updated_at"]}
                    for s, r in sorted(self._speakers.items())]

    def has(self, speaker_id: str) -> bool:
        with self._lock:
            return speaker_id in self._speakers

    def verify(self, embedding, speaker_id: str, threshold: float):
        """1:1 - is this the customer the IVR thinks is calling?"""
        with self._lock:
            rec = self._speakers.get(speaker_id)
        if rec is None:
            return {"decision": "not_enrolled", "speaker_id": speaker_id,
                    "distance": None, "score": None, "threshold": threshold}
        d = cosine_distance(embedding, rec["embedding"])
        return {"decision": "accept" if d <= threshold else "reject",
                "speaker_id": speaker_id,
                "distance": round(d, 6),
                "score": round(1.0 - d, 6),
                "threshold": threshold}

    def identify(self, embedding, threshold: float, margin: float):
        """1:N - who is this? Two guards a plain argmax does not give you:
        an absolute threshold, so an unenrolled caller can match NOBODY, and a
        margin over the runner-up, so a near-tie is not a coin flip."""
        with self._lock:
            items = [(sid, rec["embedding"]) for sid, rec in self._speakers.items()]
        if not items:
            return {"decision": "empty_registry", "speaker_id": None,
                    "distance": None, "threshold": threshold, "candidates": []}

        ranked = sorted(((cosine_distance(embedding, c), sid) for sid, c in items))
        best_d, best_id = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None

        if best_d > threshold:
            decision = "no_match"
        elif runner_up is not None and (runner_up - best_d) < margin:
            decision = "ambiguous"
        else:
            decision = "match"

        return {"decision": decision,
                "speaker_id": best_id if decision == "match" else None,
                "best_candidate": best_id,
                "distance": round(best_d, 6),
                "runner_up_distance": round(runner_up, 6) if runner_up is not None else None,
                "threshold": threshold,
                "margin": margin,
                "candidates": [{"speaker_id": sid, "distance": round(d, 6)}
                               for d, sid in ranked[:5]]}
