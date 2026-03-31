import argparse
from train import train_model
from predict import predict_next
from backtest import run_backtest
from args import get_args

MAX_TRAIN_COUNTER = 20


def train_modal_until_pass(period: str, train_counter: int = 0):
  profit = -1
  max_consec_loss = 30
  no_pass_max_consec_loss = max_consec_loss >= 30

  while profit <= 0 and True and train_counter < MAX_TRAIN_COUNTER:
    if train_counter != 0:
      print("模型背测不通过，重新构建训练参数训练")

    train_counter += 1
    train_args, modal = train_model(period)
    bt_result = run_backtest(period, train_args.time_step, modal)
    max_consec_loss = bt_result.max_consec_loss
    profit = bt_result.profit
  
  print("模型训练结束")


if __name__ == "__main__":
  args = get_args()
  print(args.period)
  if args.action == "train":
    train_modal_until_pass(args.period, 0)
  else:
    result = predict_next(args.period)
