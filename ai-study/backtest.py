"""
历史回测复盘
"""
import numpy as np
import pandas as pd
from keras.models import load_model, Sequential
from config import (
  SPREAD, SLIPPAGE, FEATURES
)
from data_loader import process_full
from args import TrainArgs, get_args
from tqdm import tqdm

from scaler import get_scaler
from trade_stregety.args import TradeArgs, get_trade_args, print_trade_args
from train import get_model_url

class BacktestResult:
    def __init__(
        self,
        total: int = 0,
        profit: float = 0,
        win_rate: float = 0,
        max_consec_loss: int = 0,
        max_drawdown: float = 0,
        max_loss_count: int = 0,
        real_profit_loss_ratio: float = 0,
    ):
        self.total = total
        self.profit = profit
        self.win_rate = win_rate
        self.max_consec_loss = max_consec_loss
        self.max_drawdown = max_drawdown
        self.max_loss_count = max_loss_count
        self.real_profit_loss_ratio = real_profit_loss_ratio

    def is_better_than(self, a: 'BacktestResult') -> bool:
        is_profit_better = self.profit > a.profit
        is_max_drawdown_better = self.max_drawdown < a.max_drawdown
        is_profit_ratio_better = self.real_profit_loss_ratio > a.real_profit_loss_ratio
        
        if is_profit_better:
            return True
        elif self.profit == a.profit:
            if is_max_drawdown_better:
                return True
            elif self.max_drawdown == a.max_drawdown:
                if is_profit_ratio_better:
                    return True
        else:
            return False
        


def print_backtest_result(result: BacktestResult, args: TrainArgs):
    print('=========模型参数==========')
    print(f"时间步：         {args.time_step}")
    print(f"训练神经元数：    {args.hidden_size}")
    print(f"学习率：         {args.learning_rate * 100}%")
    print(f"dropout:        {args.dropout:.2f}")
    print(f"batch_size:     {args.batch_size}")
    print(f"epoch:         {args.epoch}")
    print('=========回测结果==========')
    print(f"总交易次数：{result.total}")
    print(f"胜率：{result.win_rate:.2f}%")
    print(f"最大连续亏损：{result.max_consec_loss} 笔")
    print(f"触及最大亏损限制次数：{result.max_loss_count}")
    print(f"最大回撤：{result.max_drawdown:.2f}")
    print(f"总收益：{result.profit:.2f}")
    print(f"实际平均盈亏比：{result.real_profit_loss_ratio:.2f}")

def getInput(args: TrainArgs, df: pd.DataFrame):
    # 用于预测的数据，与训练数据格式一致
    data = df[FEATURES].values
    scaler_x = get_scaler(args, 'x')
    scaled = scaler_x.transform(data)
    step = args.time_step
    start = step
    end = len(scaled) - 2
    indices = range(start, end)
    X_all = []
    for i in indices:
        seq = scaled[i - step: i]
        X_all.append(seq)

    X_all = np.array(X_all)  # shape: (n_samples, step, n_features)

    return (X_all, indices)

def run_backtest(args: TrainArgs, trade_args: TradeArgs, model: Sequential = None) -> BacktestResult:
    print('===============================================')
    print('===================开始回测=====================')
    print('===============================================')
    print_trade_args(trade_args)
    # 回测数据
    df = process_full(args.period, "backtest")
    # 模型
    if not model:
        model = load_model(get_model_url(args))

    trades = []
    # 记录盈利，用于计算回撤
    balance_history = []
    current_balance = 0
    # 记录最大连续亏损
    consec_loss = 0
    max_consec_loss = 0
    # 记录最大亏损次数
    max_loss_count = 0

    # ===================== 批量构造所有序列 =====================
    X_all, indices = getInput(args, df)

    # ===================== 批量一次性预测 =====================
    preds = model.predict(X_all, batch_size=128, verbose=1)

    # ===================== 遍历结果执行回测 =====================
    for idx, i in tqdm(enumerate(indices), total=len(indices), desc="回测中"):
        # 涨跌预测
        prob = preds[idx]
        row = df.iloc[i]
        atr = row['atr']
        ma20 = row['ma20']
        close = row['close']
        next_c = row['next_c']
        next_h = row['next_h']
        next_l = row['next_l']
        side = ''
        profit = 0

        if prob > trade_args.long and close > ma20:
            entry = close + SPREAD + SLIPPAGE
            tp = entry + atr * trade_args.tp_atr
            sl = entry - atr * trade_args.sl_atr
            max_loss = atr * trade_args.max_sl_atr
            # 触发止盈 -> 止盈
            if next_h >= tp:
                profit = tp - entry
            # 没止盈：拿到收盘，以收盘价来结算，有最大亏损限制
            else:
                profit = next_c - entry
                if profit < -max_loss:
                    profit = -max_loss
                    max_loss_count += 1

            side = 'long'
        
        elif prob < trade_args.short and close < ma20:
            entry = close - SPREAD - SLIPPAGE
            tp = entry - atr * trade_args.tp_atr
            sl = entry + atr * trade_args.sl_atr
            max_loss = atr * trade_args.max_sl_atr

            # 触发止盈 -> 止盈
            if next_l <= tp:
                profit = entry - tp
            else:
                profit = entry - next_c
                if profit < -max_loss:
                    profit = -max_loss
                    max_loss_count += 1

            side = 'short'

        # 统计最大连续亏损
        if side != '':
            trades.append({
                "type": side,
                "profit": profit,
                "tp": tp,
                "sl": sl,
            })
            if profit < 0:
                consec_loss += 1
                if consec_loss > max_consec_loss:
                    max_consec_loss = consec_loss
            else:
                consec_loss = 0
    
        # 每笔都记余额，用于回撤
        current_balance += profit
        balance_history.append(current_balance)

    if not trades:
        print("无交易信号")
        return BacktestResult(
            total = 0,
            profit = 0,
            win_rate = 0,
            max_consec_loss = 0,
            max_drawdown = 0,
            max_loss_count=0,
            real_profit_loss_ratio = 0,
        )
    
    # 计算最大回撤
    balance_arr = np.array(balance_history)
    peak = np.maximum.accumulate(balance_arr)
    drawdown = (peak - balance_arr)
    max_drawdown = drawdown.max()

    tdf = pd.DataFrame(trades)
    # 交易次数
    total_trades = len(tdf)
    # 盈利次数
    win_trades = (tdf['profit'] > 0).sum()
    # 胜率
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    # 总盈利
    total_profit = tdf['profit'].sum()
    # 平均赢率
    avg_win = tdf.loc[tdf['profit'] > 0, 'profit'].mean()
    # 平均亏损
    avg_loss = -tdf.loc[tdf['profit'] < 0, 'profit'].mean()
    # 实际盈亏比
    real_profit_loss_ratio = avg_win / avg_loss

    result = BacktestResult(
        total = total_trades,
        profit = total_profit,
        win_rate = win_rate,
        max_consec_loss = max_consec_loss,
        max_loss_count=max_loss_count,
        max_drawdown = max_drawdown,
        real_profit_loss_ratio = real_profit_loss_ratio,
    )

    print_backtest_result(result, args)
    

    return result

def find_best_trade_stregety(train_args: TrainArgs, model: Sequential = None):
    best_result = BacktestResult()
    best_trade_args = TradeArgs()
    counter = 0
    finished = False
    while not finished:
        trade_args = get_trade_args(counter)
        counter += 1

        # 若交易参数组合已都遍历完毕，则结束测试循环
        if trade_args is None:
            finished = True
            break

        result: BacktestResult = run_backtest(train_args, trade_args, model)

        # 测试结果更好，保存更优结果及参数
        if (result.is_better_than(best_result)):
            best_result = result
            best_trade_args = trade_args

    return (best_result, best_trade_args)
        


if __name__ == "__main__":
    args = get_args()
    find_best_trade_stregety(args)