# src/evaluation.py

import matplotlib.pyplot as plt
import pandas as pd


def evaluate_models(results: dict):
    print("\n[INFO] Сравнение моделей:\n")

    # Создаем DataFrame для отображения
    summary = pd.DataFrame([
        {"model": model, "accuracy": round(info["accuracy"], 4), "best_params": info["best_params"]}
        for model, info in results.items()
    ]).sort_values(by="accuracy", ascending=False)

    print(summary.to_string(index=False))

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.bar(summary["model"], summary["accuracy"], color="skyblue")
    plt.title("Сравнение моделей по Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
