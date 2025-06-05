import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from config import PROCESSED_DATA_PATH, TARGET_COLUMN, AUTOML_MODELS_DIR
from sklearn.metrics import accuracy_score

def run_automl():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Разделим до запуска AutoML
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Объявим задачу
    task = Task('binary')

    # AutoML объект (без random_state)
    automl = TabularAutoML(task=task, timeout=300)

    print("[INFO] Запуск AutoML...")

    # Обучение — здесь target должен быть внутри train_data
    oof_preds = automl.fit_predict(train_data, roles={"target": TARGET_COLUMN})

    # Предсказания
    test_preds = automl.predict(test_data)

    acc = accuracy_score(test_data[TARGET_COLUMN], (test_preds.data > 0.5).astype(int))
    print(f"[RESULT] LightAutoML Accuracy: {acc:.4f}")

    os.makedirs(AUTOML_MODELS_DIR, exist_ok=True)
    model_path = AUTOML_MODELS_DIR / "automl_model.joblib"
    joblib.dump(automl, model_path)
    print(f"[INFO] AutoML модель сохранена: {model_path}")

    return acc
