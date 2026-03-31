"""
预测下一根K线 OCHL
"""
import argparse
import pickle
import numpy as np
from keras.models import load_model
from config import FEATURES
from data_loader import process_full
from args import get_random_train_args

def get_args():
  parser = argparse.ArgumentParser(description="LSTM K线预测训练工具")

  # 周期
  parser.add_argument(
    "--period",
    type=str,
    default="15min",
    choices=["1min", "15min", "1h"],
    help="K线周期"
  )

  return parser.parse_args()

def predict_next(period: str = "15min") -> tuple[float, float, float, float]:
    df = process_full(period, "predict")
    predict_args = get_random_train_args()
    model = load_model(f"./model/lstm_ohlc_{period}.keras")

    with open(f"./scaler/scaler_x_{period}.pkl", "rb") as f:
        scaler_x = pickle.load(f)
    with open(f"./scaler/scaler_y_{period}.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    data = df[FEATURES].values
    scaled = scaler_x.transform(data)
    last_seq = scaled[-predict_args.time_step:].reshape(1, predict_args.time_step, -1)

    pred_scaled = model.predict(last_seq, verbose=0)[0]
    o, h, l, c = scaler_y.inverse_transform([pred_scaled])[0]

    current = df["close"].iloc[-1]
    print("===== 下一根K线预测 =====")
    print(f"开盘: {o:.2f}")
    print(f"最高: {h:.2f}")
    print(f"最低: {l:.2f}")
    print(f"收盘: {c:.2f}")
    print(f"当前收盘价: {current:.2f}")

    return o, h, l, c


if __name__ == "__main__":
    args = get_args()
    predict_next(args.period)