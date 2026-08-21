import os
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

EVAL_THRESHOLD = 0.70


def check_data_drift(y_train: pd.Series) -> dict:
    """
    Kiem tra phan phoi nhan trong tap huan luyen (Bonus 5).
    Neu bat ky lop nao chiem < 10% tong mau, in canh bao ro rang.
    """
    total = len(y_train)
    counts = y_train.value_counts().to_dict()
    proportions = {int(cls): float(count / total) for cls, count in counts.items()}

    print(f"[DATA QUALITY] Label distribution ({total} samples):")
    for cls in sorted(proportions.keys()):
        pct = proportions[cls] * 100
        print(f"  Class {cls}: {counts.get(cls, 0)}/{total} ({pct:.2f}%)")
        if proportions[cls] < 0.10:
            print(f"  ⚠️ WARNING: Class {cls} accounts for only {pct:.2f}% (< 10%) of the training set!")

    return proportions


def build_model(params: dict):
    """
    Khoi tao mo hinh dua tren sieu tham so va model_type (Bonus 2).
    Ho tro: random_forest, gradient_boosting, logistic_regression.
    """
    model_params = params.copy()
    model_type = model_params.pop("model_type", "random_forest")
    random_state = model_params.pop("random_state", 42)

    if model_type == "random_forest":
        return RandomForestClassifier(random_state=random_state, **model_params)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state, **model_params)
    elif model_type == "logistic_regression":
        return LogisticRegression(random_state=random_state, max_iter=1000, **model_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def save_performance_report(y_eval, preds, report_path: str = "outputs/report.txt"):
    """
    Tao bao cao hieu suat chi tiet: Confusion Matrix va Precision/Recall/F1 per class (Bonus 3).
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    target_names = ["thap (0)", "trung_binh (1)", "cao (2)"]
    labels = [0, 1, 2]

    cm = confusion_matrix(y_eval, preds, labels=labels)
    clf_report = classification_report(
        y_eval,
        preds,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    report_content = [
        "=" * 60,
        "MLOps Model Performance Report (Bonus 3)",
        "=" * 60,
        "\n--- Confusion Matrix ---",
        f"Row = Actual, Column = Predicted (Labels: {labels})",
        str(cm),
        "\n--- Classification Report (Per-Class Precision, Recall, F1) ---",
        clf_report,
        "=" * 60,
    ]

    full_text = "\n".join(report_content)
    with open(report_path, "w") as f:
        f.write(full_text)

    print("\n" + full_text)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """
    # 1. Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Kiem tra lech lac phan phoi du lieu (Bonus 5)
    class_proportions = check_data_drift(y_train)

    with mlflow.start_run():
        # 3. Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # 4. Khoi tao va huan luyen mo hinh
        model = build_model(params)
        model.fit(X_train, y_train)

        # 5. Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # 6. Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for cls, pct in class_proportions.items():
            mlflow.log_metric(f"class_pct_{cls}", pct)

        # Log artifact mo hinh vao MLflow
        mlflow.sklearn.log_model(model, "model")

        # 7. In ket qua ra man hinh
        print(f"\n[EVALUATION] Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 8. Tao bao cao hieu suat chi tiet (Bonus 3)
        os.makedirs("outputs", exist_ok=True)
        save_performance_report(y_eval, preds, "outputs/report.txt")

        # Luu metrics ra file outputs/metrics.json (Bonus 5: them class_distribution)
        metrics_payload = {
            "accuracy": acc,
            "f1_score": f1,
            "class_distribution": class_proportions,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)

        # 9. Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # 10. Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)

