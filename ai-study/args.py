import argparse
import random
from types import SimpleNamespace
from config import (
  TIME_STEP_LIST, HIDDEN_SIZE_LIST, BATCH_SIZE_LIST,
  LR_LIST, DROPOUT_LIST, EPOCHS_LIST
)

class TrainArgs:
  def __init__(
      self,
      time_step: int,
      hidden_size: int,
      learning_rate: float,
      dropout: float,
      batch_size: int,
      epchos: int
  ):
    self.time_step = time_step
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    self.dropout = dropout
    self.batch_size = batch_size
    self.epchos = epchos


def get_random_train_args() -> TrainArgs:
  args = TrainArgs(
    time_step = random.choice(TIME_STEP_LIST),
    hidden_size = random.choice(HIDDEN_SIZE_LIST),
    learning_rate = random.choice(LR_LIST),
    dropout = random.choice(DROPOUT_LIST),
    batch_size = random.choice(BATCH_SIZE_LIST),
    epchos = random.choice(EPOCHS_LIST)
  )

  return args

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

  # K线数
  parser.add_argument(
    "--step",
    type=int,
    default=60
  )

  # 训练轮数
  parser.add_argument(
      "--epoch", 
      type=int, 
      default=12,
      help="训练轮数"
  )

  # 批次
  parser.add_argument(
      "--batch", 
      type=int, 
      default=128,
      help="批次大小"
  )

  # 是否只预测不训练
  parser.add_argument(
      "--action", 
      type=str,
      default="train",
      choices=["train", "predict"],
      help="train（训练）还是predict（预测）"
  )

  return parser.parse_args()

