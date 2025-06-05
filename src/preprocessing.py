# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import PROCESSED_DATA_PATH, TARGET_COLUMN

def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Обработка пропусков — удалим строки с NaN
    df = df.dropna()

    # Категориальные признаки — автоматическое определение
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Масштабируем числовые признаки (кроме целевой переменной)
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled[TARGET_COLUMN] = target.reset_index(drop=True)

    # Сохраняем
    df_scaled.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[INFO] Очищенные данные сохранены в {PROCESSED_DATA_PATH}")

    return df_scaled
