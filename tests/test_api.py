import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api.main import FEATURE_COLUMNS, app
from src.features import Preprocessor


@pytest.fixture(scope="module")
def fitted_bundle():
    labels = [0, 1] * 16
    rows = [[*([float(label)] * 28), float(10 + label)] for label in labels]
    data = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    preprocessor = Preprocessor().fit(data)
    model = XGBClassifier(n_estimators=4, max_depth=2, n_jobs=1)
    model.fit(preprocessor.transform(data), labels)
    return model, preprocessor


@pytest.mark.parametrize("no_alerts", [False, True])
def test_health_and_prediction_use_saved_bundle(fitted_bundle, tmp_path, monkeypatch, no_alerts):
    model, preprocessor = fitted_bundle
    row = pd.DataFrame([[*([1.0] * 28), 11.0]], columns=FEATURE_COLUMNS)
    score = float(model.predict_proba(preprocessor.transform(row))[0, 1])
    threshold = None if no_alerts else score
    path = tmp_path / "model.joblib"
    joblib.dump({
        "model": model,
        "preprocessor": preprocessor,
        "threshold": float("inf") if no_alerts else score,
        "feature_columns": FEATURE_COLUMNS,
    }, path)
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(path))

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "threshold": threshold}
        assert client.post("/predict", json={"v": [1] * 28, "amount": 11}).json() == {
            "fraud_score": score,
            "is_fraud_alert": not no_alerts,
            "threshold": threshold,
        }
        for payload in ({"v": [1] * 27, "amount": 11}, {"v": [1] * 28, "amount": -1}):
            assert client.post("/predict", json=payload).status_code == 422
        response = client.post(
            "/predict", content='{"v": [' + '0,' * 27 + '1e999], "amount": 11}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


def test_startup_explains_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(RuntimeError, match="Run python -m src.train"):
        with TestClient(app):
            pass
