import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data import load_raw, time_split, xy
from src.evaluate import best_threshold, expected_cost, recall_at_precision
from src.features import Preprocessor


def test_split_keeps_boundary_seconds_together_and_preserves_rows():
    data = pd.DataFrame({"Time": np.repeat(np.arange(10), 3)})
    train, validation, test = time_split(data, train=0.45, val=0.22)

    assert train["Time"].max() < validation["Time"].min()
    assert validation["Time"].max() < test["Time"].min()
    assert [len(train), len(validation), len(test)] == [12, 6, 12]
    pd.testing.assert_frame_equal(pd.concat([train, validation, test]), data)


def test_preprocessor_uses_training_statistics_for_future_rows():
    train = pd.DataFrame({"Time": [0, 1, 2], "V1": [1.0, 2.0, 3.0], "Amount": [0.0, 3.0, 15.0]})
    test = pd.DataFrame({"Time": [3, 4], "V1": [4.0, 5.0], "Amount": [63.0, 255.0]})
    preprocessor = Preprocessor().fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.columns.tolist() == ["V1", "Amount"]
    np.testing.assert_array_equal(transformed["V1"], test["V1"])
    np.testing.assert_allclose(transformed["Amount"], [2.0, 3.0])
    np.testing.assert_allclose(preprocessor.scaler.center_, [np.log(4)])
    np.testing.assert_allclose(preprocessor.scaler.scale_, [np.log(4)])


def test_recall_at_fixed_precision():
    labels = [0, 0, 1, 1]
    assert recall_at_precision(labels, [0.1, 0.2, 0.8, 0.9], 0.9) == 1.0
    assert recall_at_precision(labels, [0.8, 0.9, 0.1, 0.2], 0.9) == 0.0


def test_training_command_reproduces_cost_metrics_and_saves_a_usable_model(tmp_path):
    transaction = np.arange(300)
    fraud = transaction % 11 == 0
    data = pd.DataFrame({
        "Time": transaction * 120,
        "V1": fraud.astype(float),
        "Amount": transaction + 20,
        "Class": fraud.astype(int),
    })
    data_path = tmp_path / "toy.csv"
    data.to_csv(data_path, index=False)

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["data_path"] = "toy.csv"
    config["xgboost"].update(n_estimators=12, early_stopping_rounds=3, n_jobs=1)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    command = [sys.executable, "-m", "src.train", "--config", str(config_path)]
    metrics_path = tmp_path / config["outputs"]["metrics_path"]
    model_path = tmp_path / config["outputs"]["model_path"]

    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    first_report = metrics_path.read_bytes()
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    assert metrics_path.read_bytes() == first_report

    metrics = json.loads(first_report)
    bundle = joblib.load(model_path)
    _, validation, test = time_split(load_raw(data_path), **config["split"])
    val_features, val_labels = xy(validation)
    test_features, test_labels = xy(test)
    preprocessor = bundle["preprocessor"]
    assert bundle["feature_columns"] == ["V1", "Amount"]
    val_scores = bundle["model"].predict_proba(preprocessor.transform(val_features))[:, 1]
    test_scores = bundle["model"].predict_proba(preprocessor.transform(test_features))[:, 1]
    fee = config["cost"]["fp_cost"]
    threshold, _, _, _ = best_threshold(val_labels, val_scores, validation["Amount"], fee)

    assert bundle["threshold"] == metrics["threshold"] == threshold
    assert metrics["test_cost"] == expected_cost(test_labels, test_scores, test["Amount"], threshold, fee)
    assert metrics["test_cost_no_model"] == test.loc[test_labels.eq(1), "Amount"].sum()
    assert metrics["n_test"] == len(test)
    assert metrics["n_test_frauds"] == test_labels.sum()

    export_command = [sys.executable, "-m", "src.export_predictions", "--config", str(config_path)]
    subprocess.run(export_command, cwd=root, check=True, capture_output=True, text=True)
    predictions = pd.read_csv(metrics_path.parent / "test_predictions.csv")
    transaction_columns = ["Time", "Amount", "Class"]
    pd.testing.assert_frame_equal(
        predictions[transaction_columns], test[transaction_columns].reset_index(drop=True)
    )
    np.testing.assert_allclose(predictions["fraud_score"], test_scores)
    np.testing.assert_array_equal(predictions["alert"], test_scores >= threshold)

    hourly = pd.read_csv(metrics_path.parent / "dashboard_hourly.csv", index_col="relative_hour")
    hours = predictions["Time"] // 3600
    assert hourly.index.tolist() == sorted(hours.unique().tolist())
    for hour, group in predictions.groupby(hours):
        alerted = group["alert"].eq(1)
        confirmed_fraud = group["Class"].eq(1)
        assert hourly.loc[hour].to_dict() == {
            "transactions": len(group),
            "alerts": int(alerted.sum()),
            "true_alerts": int((confirmed_fraud & alerted).sum()),
            "detected_fraud_amount": float(group.loc[confirmed_fraud & alerted, "Amount"].sum()),
            "missed_fraud_amount": float(group.loc[confirmed_fraud & ~alerted, "Amount"].sum()),
        }
