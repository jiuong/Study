
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from args import TrainArgs

SCALER_PATH = "./scaler"

# 创造归一化函数
def create_scaler() -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))

    return scaler

def get_scaler(args: TrainArgs, key: str) -> MinMaxScaler:
    period = args.period
    path = f"{SCALER_PATH}/scaler_{key}_{period}.pkl"

    with open(path, "rb") as f:
        scaler = pickle.load(f)

    return scaler

def save_scaler(args: TrainArgs, key: str, scaler: MinMaxScaler):
    path = f"{SCALER_PATH}/scaler_{key}_{args.period}.pkl"
    with open(path, "wb") as f:
      pickle.dump(scaler, f)

    print(f"====归一化函数({key})已保存至：{path}")

    