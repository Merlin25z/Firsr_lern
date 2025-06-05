# src/config.py

from pathlib import Path

# Базовая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Пути к данным
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "heart.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "heart_cleaned.csv"

# Папки для моделей
MANUAL_MODELS_DIR = BASE_DIR / "models" / "saved_manual"
AUTOML_MODELS_DIR = BASE_DIR / "models" / "saved_automl"

# Целевой столбец
TARGET_COLUMN = "condition"


# RANDOM_SEED для повторяемости
RANDOM_STATE = 42
