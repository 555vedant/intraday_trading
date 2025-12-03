# src/models_def.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_models():

    models = {}

    # 1. Logistic Regression (balanced)
    models["logistic"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    # 2. Random Forest (balanced + tuned)
    models["random_forest"] = RandomForestClassifier(
    n_estimators=300,     # more trees
    max_depth=10,        # deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)


    # 3. Gradient Boosting
    models["gradient_boosting"] = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    return models
