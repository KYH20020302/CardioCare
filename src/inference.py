from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "best_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "best_model.pkl not found. Run src/train.py first."
        )

    return joblib.load(MODEL_PATH)


def create_sample_input():
    return pd.DataFrame([{
        "age": 55,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }])


def run_inference():
    model = load_model()
    sample = create_sample_input()

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    print("Input Sample")
    print(sample)

    print("\nPrediction")
    print("Heart Disease" if prediction == 1 else "Normal")

    print("\nPrediction Probability")
    print("Normal:", probability[0])
    print("Heart Disease:", probability[1])


if __name__ == "__main__":
    run_inference()