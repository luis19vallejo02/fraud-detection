"""Train the selected XGBoost configuration and report its held-out results."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import yaml
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.data import load_raw, time_split, xy
from src.evaluate import best_threshold, expected_cost, summarize
from src.features import Preprocessor


def main(config_path: str | Path = "config.yaml") -> None:
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    base_dir = config_path.parent
    model_path = (base_dir / config["outputs"]["model_path"]).resolve()
    metrics_path = (base_dir / config["outputs"]["metrics_path"]).resolve()
    if model_path == metrics_path:
        raise ValueError("The model and metrics must use distinct output files.")

    data = load_raw(base_dir / config["data_path"])
    train, validation, test = time_split(data, **config["split"])
    X_train_raw, y_train = xy(train)
    X_val_raw, y_val = xy(validation)
    X_test_raw, y_test = xy(test)
    for split_name, labels in (("Train", y_train), ("Validation", y_val), ("Test", y_test)):
        if set(labels.unique()) != {0, 1}:
            raise ValueError(f"{split_name} must contain both classes, labeled 0 and 1.")

    preprocessor = Preprocessor().fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    model = XGBClassifier(
        **config["xgboost"],
        scale_pos_weight=float(y_train.eq(0).sum() / y_train.eq(1).sum()),
        random_state=config["seed"],
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_scores = model.predict_proba(X_val)[:, 1]
    fp_cost = config["cost"]["fp_cost"]
    threshold, val_cost, _, _ = best_threshold(y_val, val_scores, X_val_raw["Amount"], fp_cost)
    test_scores = model.predict_proba(preprocessor.transform(X_test_raw))[:, 1]
    test_metrics = summarize(y_test, test_scores)
    metrics = {
        "threshold": threshold if np.isfinite(threshold) else None,
        "fp_cost": fp_cost,
        "best_iteration": model.best_iteration + 1,
        "val_pr_auc": float(average_precision_score(y_val, val_scores)),
        "val_cost": val_cost,
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_recall_at_p80": test_metrics["recall@p80"],
        "test_cost": expected_cost(y_test, test_scores, X_test_raw["Amount"], threshold, fp_cost),
        "test_cost_no_model": float(X_test_raw.loc[y_test.eq(1), "Amount"].sum()),
        "n_test": len(y_test),
        "n_test_frauds": int(y_test.sum()),
    }
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "threshold": threshold,
        "feature_columns": X_train.columns.to_list(),
    }

    report = json.dumps(metrics, indent=2, allow_nan=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(dir=model_path.parent) as model_staging,
        TemporaryDirectory(dir=metrics_path.parent) as metrics_staging,
    ):
        staged_model = Path(model_staging) / model_path.name
        staged_metrics = Path(metrics_staging) / metrics_path.name
        joblib.dump(artifact, staged_model, compress=3)
        staged_metrics.write_text(report + "\n", encoding="utf-8")
        staged_model.replace(model_path)
        staged_metrics.replace(metrics_path)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    main(parser.parse_args().config)
