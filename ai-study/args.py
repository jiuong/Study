import argparse
import random
from config import (
  TIME_STEP_LIST, HIDDEN_SIZE_LIST, BATCH_SIZE_LIST,
  LR_LIST, DROPOUT_LIST, EPOCHS_LIST
)

class TrainArgs:
  def __init__(
      self,
      period: str,
      time_step: int,
      hidden_size: int = 0,
      learning_rate: float = 0,
      dropout: float = 0,
      batch_size: int = 0,
      epoch: int = 0,
  ):
    self.period = period
    self.time_step = time_step
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    self.dropout = dropout
    self.batch_size = batch_size
    self.epoch = epoch


def get_random_train_args(cmd_args = {}) -> TrainArgs:
  args = TrainArgs(
    period = cmd_args.period,
    time_step = cmd_args.time_step or random.choice(TIME_STEP_LIST),
    hidden_size = random.choice(HIDDEN_SIZE_LIST),
    learning_rate = random.choice(LR_LIST),
    dropout = random.choice(DROPOUT_LIST),
    batch_size = random.choice(BATCH_SIZE_LIST),
    epoch = random.choice(EPOCHS_LIST)
  )

  # print_train_args(args)
  return args

def print_train_args(args: TrainArgs):
    print(f"时间步：         {args.time_step}")
    print(f"训练神经元数：    {args.hidden_size}")
    print(f"学习率：         {args.learning_rate * 100}%")
    print(f"dropout:        {args.dropout:.2f}")
    print(f"batch_size:     {args.batch_size}")
    print(f"epoch:         {args.epoch}")

def get_args() -> TrainArgs:
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
    "--time_step",
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

  cmd_args = parser.parse_args()


  return TrainArgs(
    period = cmd_args.period,
    time_step = cmd_args.time_step,
    batch_size = cmd_args.batch,
    epoch = cmd_args.epoch,
  )

