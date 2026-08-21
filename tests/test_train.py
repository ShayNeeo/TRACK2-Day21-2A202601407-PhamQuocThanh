import os
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.train import train
from src.serve import app


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi GCS hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    # 1. Tao mang X co kich thuoc (n, len(FEATURE_NAMES)) voi gia tri [0, 1)
    X = rng.random((n, len(FEATURE_NAMES)))

    # 2. Tao mang y gom n phan tu nguyen ngau nhien trong [0, 3)
    y = rng.integers(0, 3, size=n)

    # 3. Xay dung DataFrame, them cot "target"
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # 4. Luu 160 dong dau lam tap huan luyen, 40 dong cuoi lam tap danh gia
    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    # 5. Tra ve (train_path, eval_path)
    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "class_distribution" in metrics


def test_model_file_created(tmp_path):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")


def test_multiple_algorithms(tmp_path):
    """Bonus 2: Kiem tra huan luyen thanh cong voi GradientBoosting va LogisticRegression."""
    train_path, eval_path = _make_temp_data(tmp_path)

    # Gradient Boosting
    acc_gb = train(
        {"model_type": "gradient_boosting", "n_estimators": 10, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert 0.0 <= acc_gb <= 1.0

    # Logistic Regression
    acc_lr = train(
        {"model_type": "logistic_regression", "C": 1.0},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert 0.0 <= acc_lr <= 1.0


def test_report_file_created(tmp_path):
    """Bonus 3: Kiem tra file outputs/report.txt duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.txt")
    with open("outputs/report.txt") as f:
        content = f.read()
    assert "Confusion Matrix" in content
    assert "Classification Report" in content


def test_serve_api(tmp_path):
    """Kiem tra cac endpoint cua FastAPI: /health va /predict."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    client = TestClient(app)

    # Health endpoint
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    # Predict endpoint - valid features
    sample_features = [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]
    res_predict = client.post("/predict", json={"features": sample_features})
    assert res_predict.status_code == 200
    data = res_predict.json()
    assert "prediction" in data
    assert data["label"] in ["thap", "trung_binh", "cao"]

    # Predict endpoint - invalid features length (400 Bad Request)
    res_invalid = client.post("/predict", json={"features": [1.0, 2.0]})
    assert res_invalid.status_code == 400

