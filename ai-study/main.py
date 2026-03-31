from trade_stregety.args import print_trade_args
from train import train_model
from backtest import find_best_trade_stregety, print_backtest_result, run_backtest
from args import get_args, get_random_train_args

MAX_TRAIN_COUNTER = 20


def train_modal_until_pass(cmd_args, train_counter: int = 0):
  train_args = get_random_train_args(cmd_args)
  profit = -1
  max_consec_loss = 30
  no_pass_max_consec_loss = max_consec_loss >= 30

  while profit <= 0 and no_pass_max_consec_loss and train_counter < MAX_TRAIN_COUNTER:
    if train_counter != 0:
      print("模型背测不通过，重新构建训练参数训练")

    train_counter += 1
    modal = train_model(train_args)
    bt_result, trade_args = find_best_trade_stregety(train_args, modal)
    max_consec_loss = bt_result.max_consec_loss
    profit = bt_result.profit

  print("模型训练结束")
  print('===============最终结果===============')
  print_backtest_result(bt_result, train_args)
  print_trade_args(trade_args)


if __name__ == "__main__":
  args = get_args()
  train_modal_until_pass(args, 0)
  # else:
  #   result = predict_next(args.period)
