"""Ranking metrics for binary fraud labels and scores (higher means riskier)."""

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
