# src/models_def.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def get_models():
    
    models = {}

    # 1. logistic regression i added scaling too
    models["logistic_regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]
    )

    # 2. random forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # 3. gradient boosting
    models["gradient_boosting"] = GradientBoostingClassifier(
        random_state=42
    )

    # 4. SVM + scaling 
    models["svm"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=False, random_state=42))
        ]
    )

    return models
