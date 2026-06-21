from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from preprocessing import create_preprocessor, create_feature_selector, create_model_pipeline

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed.cleveland.data"
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"

RANDOM_STATE = 42
TEST_SIZE = 0.2

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]


df = pd.read_csv(
    DATA_PATH,
    names=columns,
    na_values="?"
)

df["target"] = (df["target"] > 0).astype(int)

X = df.drop("target", axis=1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


preprocessor = create_preprocessor()
feature_selector = create_feature_selector()

feature_selection_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selector", feature_selector)
])

feature_selection_pipeline.fit(X_train, y_train)

selected_features = X_train.columns[
    feature_selection_pipeline.named_steps["feature_selector"].get_support()
]

print("\nSelected Features")
print(selected_features.tolist())


mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
mlflow.set_experiment("CardioCare_HeartDisease")


models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    "SVC": SVC(
        random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        random_state=RANDOM_STATE
    )
}

results = []

for model_name, model in models.items():

    pipeline = create_model_pipeline(model)

    with mlflow.start_run(run_name=model_name):

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "model": model_name,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        })

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("imputer_strategy", "median")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("feature_selector", "SelectFromModel_RandomForest")

        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("run_type", "baseline_model")

        mlflow.log_metric("balanced_accuracy", balanced_acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual_0", "Actual_1"],
            columns=["Pred_0", "Pred_1"]
        )

        cm_path = ROOT_DIR / f"confusion_matrix_{model_name}.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink(missing_ok=True)

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle"
        )

        print("\n================================")
        print("Model:", model_name)
        print("Balanced Accuracy:", balanced_acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("Confusion Matrix:")
        print(cm)


results_df = pd.DataFrame(results)

print("\nModel Comparison")
print(results_df[[
    "model",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1"
]])


candidate_pipeline = create_model_pipeline(
    RandomForestClassifier(random_state=RANDOM_STATE)
)

cv_scores = cross_val_score(
    candidate_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="balanced_accuracy"
)

print("\n5-Fold Cross Validation - RandomForest")
print("CV Scores:", cv_scores)
print("Mean Balanced Accuracy:", cv_scores.mean())
print("Std Balanced Accuracy:", cv_scores.std())


param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 3, 5, 10],
    "model__min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=candidate_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nGridSearchCV - RandomForest")
print("Best Params:", grid_search.best_params_)
print("Best CV Balanced Accuracy:", grid_search.best_score_)


best_model = grid_search.best_estimator_
MODEL_PATH = ROOT_DIR / "best_model.pkl"
joblib.dump(best_model, MODEL_PATH)

print("Saved final model to:", MODEL_PATH)
y_pred_best = best_model.predict(X_test)

best_balanced_acc = balanced_accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)
best_cm = confusion_matrix(y_test, y_pred_best)

print("\nTuned RandomForest Test Performance")
print("Balanced Accuracy:", best_balanced_acc)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1:", best_f1)
print("Confusion Matrix:")
print(best_cm)


with mlflow.start_run(run_name="RandomForest_GridSearch"):

    mlflow.set_tag("model_family", "RandomForest")
    mlflow.set_tag("run_type", "hyperparameter_tuning")

    mlflow.log_param("cv", 5)
    mlflow.log_param("scoring", "balanced_accuracy")
    mlflow.log_param("search_method", "GridSearchCV")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    mlflow.log_metric("cv_mean_balanced_accuracy", grid_search.best_score_)
    mlflow.log_metric("cv_std_balanced_accuracy", cv_scores.std())
    mlflow.log_metric("test_balanced_accuracy", best_balanced_acc)
    mlflow.log_metric("test_precision", best_precision)
    mlflow.log_metric("test_recall", best_recall)
    mlflow.log_metric("test_f1", best_f1)

    tuned_cm_df = pd.DataFrame(
        best_cm,
        index=["Actual_0", "Actual_1"],
        columns=["Pred_0", "Pred_1"]
    )

    tuned_cm_path = ROOT_DIR / "confusion_matrix_RandomForest_GridSearch.csv"
    tuned_cm_df.to_csv(tuned_cm_path)
    mlflow.log_artifact(str(tuned_cm_path))
    tuned_cm_path.unlink(missing_ok=True)

    mlflow.sklearn.log_model(
        best_model,
        name="model",
        serialization_format="cloudpickle"
    )


print("\nFinal Model Selection")
print(
    "RandomForest was selected as the main candidate because it showed "
    "the highest balanced accuracy and F1-score among the baseline models. "
    "However, Logistic Regression showed the highest recall, which is clinically "
    "important because false negatives mean predicting a heart disease patient "
    "as normal. Therefore, the final decision should consider both overall "
    "performance and recall."
)