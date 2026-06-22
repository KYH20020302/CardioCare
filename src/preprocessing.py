from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42

#모델에 맞는 스케일링 적용
def create_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def create_feature_selector():
    return SelectFromModel(
        RandomForestClassifier(
            random_state=RANDOM_STATE
        )
    )


def create_model_pipeline(model):
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("feature_selector", create_feature_selector()),
        ("model", model)
    ])