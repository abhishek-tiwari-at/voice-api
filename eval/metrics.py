"""Speaker-verification metrics: EER, minDCF, and the operating threshold.

Scores are SIMILARITIES (higher = more likely the same speaker), i.e.
1 - cosine_distance. Labels are 1 for a target (same speaker) trial, 0 for a
nontarget (impostor) trial.
"""
import numpy as np


def _sweep(scores, labels):
    """Sort by score descending and build the FAR / FRR curves.

    Walking down the sorted list is the same as lowering the acceptance
    threshold: each step accepts one more trial."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    n_target = int(labels.sum())
    n_nontarget = len(labels) - n_target
    if n_target == 0 or n_nontarget == 0:
        raise ValueError("need at least one target and one nontarget trial")

    order = np.argsort(-scores, kind="mergesort")   # stable, highest score first
    s, y = scores[order], labels[order]

    accepted_targets    = np.cumsum(y)              # targets accepted so far
    accepted_nontargets = np.cumsum(1 - y)          # impostors wrongly accepted

    frr = 1.0 - accepted_targets / n_target         # genuine speakers turned away
    far = accepted_nontargets / n_nontarget         # impostors let in
    return s, far, frr, n_target, n_nontarget


def compute_eer(scores, labels):
    """Equal Error Rate: the point where FAR == FRR, plus the threshold there.

    The returned threshold is a SIMILARITY. Convert to the distance threshold the
    API uses with: distance_threshold = 1 - similarity_threshold."""
    s, far, frr, _, _ = _sweep(scores, labels)
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return eer, float(s[idx]), float(far[idx]), float(frr[idx])


def compute_min_dcf(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """Minimum normalised detection cost.

    EER weights both errors equally. minDCF asks a different question: if only
    1% of calls are the genuine customer (p_target=0.01) and a false accept
    costs as much as a false reject, how badly can the system do at its best
    operating point? It is the standard companion metric to EER."""
    s, far, frr, _, _ = _sweep(scores, labels)
    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    idx = int(np.argmin(dcf))
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(dcf[idx] / default_cost), float(s[idx])


def rates_at_threshold(scores, labels, sim_threshold):
    """FAR and FRR at one specific operating point. Use this to report what a
    chosen threshold actually costs."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    accept = scores >= sim_threshold
    n_target = int(labels.sum())
    n_nontarget = len(labels) - n_target
    far = float(np.sum(accept & (labels == 0)) / n_nontarget)
    frr = float(np.sum(~accept & (labels == 1)) / n_target)
    return far, frr
