from pathlib import Path
from datetime import datetime, timedelta
import logging

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed.cleveland.data"
MODEL_PATH = ROOT_DIR / "best_model.pkl"
LOG_PATH = ROOT_DIR / "inference.log"
DRIFT_REPORT_PATH = ROOT_DIR / "drift_report.csv"
METRIC_PLOT_PATH = ROOT_DIR / "metric_timeseries.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_VERSION = "RandomForest_GridSearch_v1"

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

continuous_cols = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def load_data():
    df = pd.read_csv(
        DATA_PATH,
        names=columns,
        na_values="?"
    )

    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "best_model.pkl not found. Run src/train.py first."
        )

    return joblib.load(MODEL_PATH)

# 추론 경로에 logging 기반 계측 추가
def log_inference(model, X_batch, y_true=None):
    predictions = model.predict(X_batch)

    log_message = {
        "model_version": MODEL_VERSION,
        "input_shape": X_batch.shape,
        "predictions": predictions.tolist()
    }

    if y_true is not None:
        log_message["true_labels"] = y_true.tolist()

    logging.info(log_message)

    return predictions

#테스트 셋 복사본에서 연속형 특성 최소 하나의 분포를 인위적으로 이동
def create_drifted_data(X_test):
    X_drifted = X_test.copy()

    rng = np.random.default_rng(RANDOM_STATE)

    X_drifted["chol"] = X_drifted["chol"] + 80

    X_drifted["chol"] = X_drifted["chol"] + rng.normal(
        loc=0,
        scale=40,
        size=len(X_drifted)
    )

    X_drifted["oldpeak"] = X_drifted["oldpeak"] + rng.normal(
        loc=1.0,
        scale=0.5,
        size=len(X_drifted)
    )

    return X_drifted

# 각 연속형 특성에 대해 학습 분포와 이동된 푼부로 ks_2samp 수행 및 p-value 보고, p<0.05인 특성 플래그
def run_ks_tests(X_train, X_drifted):
    drift_results = []

    for col in continuous_cols:
        statistic, p_value = ks_2samp(
            X_train[col].dropna(),
            X_drifted[col].dropna()
        )

        drift_results.append({
            "feature": col,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_flag": p_value < 0.05
        })

    drift_report = pd.DataFrame(drift_results)
    drift_report.to_csv(DRIFT_REPORT_PATH, index=False)

    return drift_report

# 원본 테스트셋과 트리프트 테스트셋의 balanced accuracy를 비교 보고, 입력 트리프트와 성능 저하의 연관성을 가시화 
def compare_performance(model, X_test, y_test, X_drifted):
    original_pred = model.predict(X_test)
    drifted_pred = model.predict(X_drifted)

    original_balanced_acc = balanced_accuracy_score(
        y_test,
        original_pred
    )

    drifted_balanced_acc = balanced_accuracy_score(
        y_test,
        drifted_pred
    )

    return original_balanced_acc, drifted_balanced_acc

# 시간에 따른 지표 변화를 보여주는 선택 지표의 시계열 그래프 포함
def save_metric_timeseries(original_score, drifted_score):
    timestamps = [
        datetime.now(),
        datetime.now() + timedelta(days=1)
    ]

    scores = [
        original_score,
        drifted_score
    ]

    plt.figure(figsize=(6, 4))
    plt.plot(timestamps, scores, marker="o")
    plt.title("Balanced Accuracy Over Time")
    plt.xlabel("Time")
    plt.ylabel("Balanced Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(METRIC_PLOT_PATH)
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = load_model()

    log_inference(
        model=model,
        X_batch=X_test,
        y_true=y_test
    )

    X_drifted = create_drifted_data(X_test)

    drift_report = run_ks_tests(
        X_train=X_train,
        X_drifted=X_drifted
    )

    original_score, drifted_score = compare_performance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_drifted=X_drifted
    )

    save_metric_timeseries(
        original_score=original_score,
        drifted_score=drifted_score
    )

    print("\nDrift Report")
    print(drift_report)

    print("\nPerformance Comparison")
    print("Original Balanced Accuracy:", original_score)
    print("Drifted Balanced Accuracy:", drifted_score)

    print("\nSaved Files")
    print("Inference log:", LOG_PATH)
    print("Drift report:", DRIFT_REPORT_PATH)
    print("Metric plot:", METRIC_PLOT_PATH)


if __name__ == "__main__":
    main()