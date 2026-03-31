"""
构建时序训练数据集
"""
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import FEATURES, TARGETS
from data_loader import process_full


def build_dataset(period: str = "1min", type: str = "train", time_step: int = 60) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    df = process_full(period, type)

    X_raw = df[FEATURES].values
    y_raw = df[TARGETS].values

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    X, y = [], []
    for i in range(time_step, len(X_scaled)):
        X.append(X_scaled[i - time_step: i])
        y.append(y_scaled[i])

    X = np.array(X)
    y = np.array(y)

    # 保存归一化器
    with open(f"./scaler/scaler_x_{period}.pkl", "wb") as f:
        pickle.dump(scaler_x, f)
    with open(f"./scaler/scaler_y_{period}.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    return X, y, scaler_x, scaler_y


if __name__ == "__main__":
    X_data, y_data, _, _ = build_dataset()
    # print("X shape:", X_data.shape)
    # print("y shape:", y_data.shape)