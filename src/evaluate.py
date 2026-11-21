"""Ranking metrics and review costs for scores where higher means riskier."""

import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_precision(y_true, scores, min_precision: float = 0.80) -> float:
    """Return the best recall at a real score cutoff meeting the precision floor."""
    if not 0.0 <= min_precision <= 1.0:
        raise ValueError("min_precision must be between 0 and 1.")

    precision, recall, _ = precision_recall_curve(y_true, scores)
    # The final point is a plotting endpoint without a corresponding threshold.
    eligible_recall = recall[:-1][precision[:-1] >= min_precision]
    return float(eligible_recall.max(initial=0.0))


def summarize(y_true, scores, name: str = "model") -> dict[str, str | float]:
    """Summarize scores for a split containing both legitimate and fraud cases."""
    return {
        "model": name,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall@p80": recall_at_precision(y_true, scores, 0.80),
        "recall@p50": recall_at_precision(y_true, scores, 0.50),
    }


def _cost_inputs(y_true, scores, amounts, fp_cost):
    labels = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if labels.ndim != 1 or labels.size == 0 or labels.shape != scores.shape or labels.shape != amounts.shape:
        raise ValueError("Labels, scores and amounts must be aligned non-empty vectors.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (legitimate) or 1 (fraud).")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Original amounts must be finite and non-negative.")
    if not np.isfinite(fp_cost) or fp_cost < 0:
        raise ValueError("The cost per false alert must be finite and non-negative.")
    return labels, scores, amounts


def expected_cost(y_true, scores, amounts, threshold: float, fp_cost: float = 5.0) -> float:
    """Sum missed fraud amounts and the fixed cost per false alert."""
    labels, scores, amounts = _cost_inputs(y_true, scores, amounts, fp_cost)
    if np.isnan(threshold):
        raise ValueError("The threshold cannot be NaN.")
    flagged = scores >= threshold
    missed_amount = amounts[(labels == 1) & ~flagged].sum()
    false_alerts = np.count_nonzero((labels == 0) & flagged)
    return float(missed_amount + fp_cost * false_alerts)


def best_threshold(y_true, scores, amounts, fp_cost: float = 5.0):
    """Evaluate every distinct decision in O(n log n) and return cutoff, cost and curve.

    Cutoffs descend from infinity (no alerts) to the lowest score. Equal scores
    enter together. If costs tie, the higher cutoff sends fewer alerts.
    """
    labels, scores, amounts = _cost_inputs(y_true, scores, amounts, fp_cost)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    fraud = labels[order] == 1
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True]) + 1
    alert_counts = np.r_[0, group_ends]

    missed_amounts = np.where(fraud, amounts[order], 0.0)
    remaining_loss = np.r_[np.cumsum(missed_amounts[::-1])[::-1], 0.0]
    false_alert_counts = np.r_[0, np.cumsum(~fraud)]
    cutoffs = np.r_[np.inf, sorted_scores[group_ends - 1]]
    costs = remaining_loss[alert_counts] + fp_cost * false_alert_counts[alert_counts]
    best = int(costs.argmin())
    return float(cutoffs[best]), float(costs[best]), cutoffs, costs
