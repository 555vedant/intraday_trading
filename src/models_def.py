# src/models_def.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def get_models():

    models = {}

    models["logistic_regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=150,          
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    models["gradient_boosting"] = GradientBoostingClassifier(
        n_estimators=80,           
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    models["svm"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.5, gamma="scale", class_weight="balanced")),
        ]
    )

    return models
