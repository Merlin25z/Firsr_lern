# src/data_loader.py

import pandas as pd
from config import RAW_DATA_PATH

def load_data():
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"[INFO] Данные успешно загружены. Форма: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Файл {RAW_DATA_PATH} не найден. Пожалуйста, скачай его и положи в data/raw/")
        raise
