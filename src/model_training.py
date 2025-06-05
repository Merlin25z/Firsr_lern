# src/model_training.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from config import PROCESSED_DATA_PATH, MANUAL_MODELS_DIR, TARGET_COLUMN, RANDOM_STATE
import os

def train_models():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "log_reg": (LogisticRegression(max_iter=1000), {
            "C": [0.01, 0.1, 1, 10]
        }),
        "decision_tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), {
            "max_depth": [3, 5, 10]
        }),
        "random_forest": (RandomForestClassifier(random_state=RANDOM_STATE), {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        }),
        "svc": (SVC(), {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }),
        "knn": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 7]
        }),
        "xgboost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE), {
            "n_estimators": [50, 100],
            "max_depth": [3, 5]
        })
    }

    os.makedirs(MANUAL_MODELS_DIR, exist_ok=True)
    results = {}

    for name, (model, params) in models.items():
        print(f"\n[INFO] Обучение модели: {name}")
        grid = GridSearchCV(model, params, cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"[RESULT] {name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # Сохраняем модель
        model_path = MANUAL_MODELS_DIR / f"{name}.joblib"
        joblib.dump(best_model, model_path)
        print(f"[INFO] Модель сохранена: {model_path}")

        results[name] = {
            "accuracy": acc,
            "best_params": grid.best_params_
        }

    return results
