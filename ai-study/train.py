"""
模型训练（CPU 专用）
"""
import os

import numpy as np

from config import FEATURES
from data_loader import process_full
from scaler import create_scaler, save_scaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from args import TrainArgs

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制CPU

def get_model_url(args: TrainArgs) -> str:
    return f"./model/lstm_ohlc_{args.period}_{args.time_step}.keras"

def build_model(args: TrainArgs, features ) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(args.time_step, features)))
    model.add(LSTM(args.hidden_size, return_sequences=False))
    model.add(Dropout(args.dropout))
    model.add(Dense(16, activation='relu'))
    # model.add(LSTM(args.hidden_size, return_sequences=False))
    model.add(Dropout(args.dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="binary_crossentropy", metrics=['accuracy'])
    return model


def train_model(args: TrainArgs) -> Sequential:
    period = args.period
    time_step = args.time_step
    model_path = get_model_url(args)
    # 1. 加载数据 & 构造 X, y
    df = process_full(period, "train")
    data = df[FEATURES].values

    # 2. 归一化
    scaler_x = create_scaler()

    scaled_x = scaler_x.fit_transform(data)
    save_scaler(args, 'x', scaler_x)
    # y 只预测下一根K线 涨跌
    scaled_y = df[["prob"]].values

    # 3. 构造时序样本
    X, y = [], []
    for i in range(time_step, len(scaled_x)):
        X.append(scaled_x[i - time_step : i])
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
        model_path,
        monitor = "val_loss",
        save_best_only = True,
        verbose = 1
    )
    # 5. 训练
    model = build_model(args, X.shape[2])
    model.fit(
        X, y,
        batch_size = args.batch_size,
        epochs = args.epoch,
        validation_split = 0.1,
        callbacks = [early_stop, checkpoint],
        # verbose = 1
    )

    print(f"训练完成，模型已保存至：{model_path}")
    return model


if __name__ == "__main__":
    train_model()