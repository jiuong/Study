"""
模型训练（CPU 专用）
"""
import os
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import FEATURES
from data_loader import process_full
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制CPU
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset_builder import build_dataset
from args import get_random_train_args, TrainArgs

def print_train_args(args: TrainArgs):
    print(f"时间步：         {args.time_step}")
    print(f"训练神经元数：    {args.hidden_size}")
    print(f"学习率：         {args.learning_rate * 100}%")
    print(f"dropout:        {args.dropout:.2f}")
    print(f"batch_size:     {args.batch_size}")
    print(f"epchos:         {args.epchos}")

def build_model(train_args: TrainArgs, features ) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(train_args.time_step, features)))
    model.add(LSTM(train_args.hidden_size, return_sequences=False))
    model.add(Dropout(train_args.dropout))
    model.add(Dense(16, activation='relu'))
    # model.add(LSTM(train_args.hidden_size, return_sequences=False))
    model.add(Dropout(train_args.dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=train_args.learning_rate), loss="binary_crossentropy", metrics=['accuracy'])
    return model


def train_model(period: str = "15min") -> tuple[TrainArgs, Sequential]:
    train_args = get_random_train_args()
    print_train_args(train_args)
    # 1. 加载数据 & 构造 X, y
    df = process_full(period, "train")
    data = df[FEATURES].values

    # 2. 归一化
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaled_x = scaler_x.fit_transform(data)
    # y 只预测下一根K线 涨跌
    scaled_y = scaler_y.fit_transform(df[["prob"]].values)

    # 3. 构造时序样本
    X, y = [], []
    for i in range(train_args.time_step, len(scaled_x)):
        X.append(scaled_x[i - train_args.time_step : i])
        y.append(scaled_y[i])

    X = np.array(X)
    y = np.array(y)

    # 4. 早停+保存最优模型
    early_stop = EarlyStopping(
        monitor = "val_loss",
        patience = 5,
        restore_best_weights = True,
        verbose = 1
    )

    checkpoint = ModelCheckpoint(
        f"./model/lstm_ohlc_{period}_{train_args.time_step}.keras",
        monitor = "val_loss",
        save_best_only = True,
        verbose = 1
    )
    print(f"features: {X.shape[2]}")
    # 5. 训练
    model = build_model(train_args, X.shape[2])
    model.fit(
        X, y,
        batch_size = train_args.batch_size,
        epochs = train_args.epchos,
        validation_split = 0.1,
        callbacks = [early_stop, checkpoint],
        # verbose = 1
    )

    # 6. 保存 scaler
    with open(f"./scaler/scaler_x_{period}.pkl", "wb") as f:
        pickle.dump(scaler_x, f)
    with open(f"./scaler/scaler_y_{period}.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    print("训练完成，模型已保存")
    return [train_args, model]


if __name__ == "__main__":
    train_model()