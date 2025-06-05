# src/main.py

from data_loader import load_data
from preprocessing import preprocess_data
from model_training import train_models
from evaluation import evaluate_models
from automl_runner import run_automl

def main():
    print("[STEP 1] Загрузка данных")
    df = load_data()

    print("\n[STEP 2] Предобработка данных")
    processed_df = preprocess_data(df)

    print("\n[STEP 3] Обучение моделей (ручной подход)")
    manual_results = train_models()

    print("\n[STEP 4] Оценка моделей")
    evaluate_models(manual_results)

    print("\n[STEP 5] AutoML (LightAutoML)")
    automl_accuracy = run_automl()
    print(f"[SUMMARY] LightAutoML Accuracy: {automl_accuracy:.4f}")

if __name__ == "__main__":
    main()
