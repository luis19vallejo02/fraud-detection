"""Export held-out predictions and relative-hour totals for the dashboard."""

import argparse
from pathlib import Path

import joblib
import yaml

from src.data import load_raw, time_split, xy


def main(config_path: str | Path = "config.yaml") -> None:
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    base_dir = config_path.parent
    model_path = base_dir / config["outputs"]["model_path"]
    bundle = joblib.load(model_path)
    _, _, test = time_split(load_raw(base_dir / config["data_path"]), **config["split"])
    features, labels = xy(test)
    transformed = bundle["preprocessor"].transform(features)
    transformed = transformed[bundle.get("feature_columns", transformed.columns)]
    scores = bundle["model"].predict_proba(transformed)[:, 1]
    alerts = scores >= bundle["threshold"]
    predictions = test[["Time", "Amount", "Class"]].assign(
        fraud_score=scores, alert=alerts.astype(int)
    )

    # Time is the elapsed second since the first transaction, rather than a wall-clock hour.
    fraud = labels.eq(1)
    report_hours = predictions.assign(
        relative_hour=predictions["Time"].floordiv(3600).astype(int),
        transactions=1,
        alerts=alerts.astype(int),
        true_alerts=(fraud & alerts).astype(int),
        detected_fraud_amount=predictions["Amount"].where(fraud & alerts, 0),
        missed_fraud_amount=predictions["Amount"].where(fraud & ~alerts, 0),
    ).groupby("relative_hour")[[
        "transactions", "alerts", "true_alerts", "detected_fraud_amount", "missed_fraud_amount"
    ]].sum()
    report_dir = (base_dir / config["outputs"]["metrics_path"]).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    report_hours.to_csv(report_dir / "dashboard_hourly.csv", float_format="%.2f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    main(parser.parse_args().config)
