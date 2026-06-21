import unittest
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import create_model_pipeline


class TestCardioCarePipeline(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame([
            {
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
            },
            {
                "age": 60,
                "sex": 0,
                "cp": 3,
                "trestbps": 130,
                "chol": 220,
                "fbs": 0,
                "restecg": 0,
                "thalach": 160,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 3
            },
            {
                "age": 67,
                "sex": 1,
                "cp": 4,
                "trestbps": 160,
                "chol": 286,
                "fbs": 0,
                "restecg": 2,
                "thalach": 108,
                "exang": 1,
                "oldpeak": 1.5,
                "slope": 2,
                "ca": 3,
                "thal": 3
            },
            {
                "age": 41,
                "sex": 0,
                "cp": 2,
                "trestbps": 130,
                "chol": 204,
                "fbs": 0,
                "restecg": 2,
                "thalach": 172,
                "exang": 0,
                "oldpeak": 1.4,
                "slope": 1,
                "ca": 0,
                "thal": 3
            }
        ])

        self.y = pd.Series([1, 0, 1, 0])

        self.model = create_model_pipeline(
            RandomForestClassifier(
                random_state=42,
                n_estimators=10
            )
        )

        self.model.fit(self.X, self.y)

    def test_prediction_shape_matches_input_shape(self):
        preds = self.model.predict(self.X)

        self.assertEqual(
            preds.shape[0],
            self.X.shape[0]
        )

    def test_prediction_probability_range_and_sum(self):
        probs = self.model.predict_proba(self.X)

        self.assertTrue((probs >= 0).all())
        self.assertTrue((probs <= 1).all())

        row_sums = probs.sum(axis=1)

        for value in row_sums:
            self.assertAlmostEqual(value, 1.0, places=5)

    def test_chol_input_range(self):
        chol_values = self.X["chol"]

        self.assertTrue(
            ((chol_values >= 0) & (chol_values <= 600)).all()
        )

    def test_pipeline_deterministic_with_fixed_seed(self):
        preds_first = self.model.predict(self.X)
        preds_second = self.model.predict(self.X)

        self.assertTrue(
            (preds_first == preds_second).all()
        )


if __name__ == "__main__":
    unittest.main()