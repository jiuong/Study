"""
历史回测复盘
"""
import pickle
from types import SimpleNamespace
import numpy as np
import pandas as pd
from keras.models import load_model, Sequential
from config import SPREAD, SLIPPAGE, FEATURES
from data_loader import process_full
from args import TrainArgs, get_args
from tqdm import tqdm

class BacktestResult:
    def __init__(
        self,
        total: int,
        profit: float,
        win_rate: float,
        max_consec_loss: int,
        max_drawdown: float,
        real_profit_loss_ratio: float
    ):
        self.total = total
        self.profit = profit
        self.win_rate = win_rate
        self.max_consec_loss = max_consec_loss,
        self.max_drawdown = max_drawdown,
        self.real_profit_loss_ratio = real_profit_loss_ratio


def run_backtest(period: str, time_step: int, model: Sequential = None) -> BacktestResult:
    # 回测数据
    df = process_full(period, "backtest")
    # 模型
    if not model:
        model = load_model(f"./model/lstm_ohlc_{period}_{time_step}.keras")
    # 归一化器
    with open(f"./scaler/scaler_x_{period}.pkl", "rb") as f:
        scaler_x = pickle.load(f)
    with open(f"./scaler/scaler_y_{period}.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    # 用于预测的数据，与训练数据格式一致
    data = df[FEATURES].values
    scaled = scaler_x.transform(data)
    trades = []
    # 做多信号
    prob_long = 0.52
    prob_short = 1 - prob_long
    # 记录盈利，用于计算回撤
    balance_history = []
    current_balance = 0
    # 记录最大连续亏损
    consec_loss = 0
    max_consec_loss = 0
    # 单日最大亏损
    max_loss_atr = 1
    # 止盈atr比例
    tp_atr = 1.8
    # 止损atr比例
    sl_atr = 0.9



    # ===================== 批量构造所有序列 =====================
    start = time_step
    end = len(scaled) - 2
    indices = range(start, end)
    X_all = []
    for i in indices:
        seq = scaled[i - time_step: i]
        X_all.append(seq)

    X_all = np.array(X_all)  # shape: (n_samples, time_step, n_features)

    # ===================== 批量一次性预测 =====================
    preds_scaled = model.predict(X_all, batch_size=128, verbose=1)
    preds = scaler_y.inverse_transform(preds_scaled)

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
    
        is_trade = False
        profit = 0

        
        if prob > prob_long and close > ma20:
            entry = close + SPREAD + SLIPPAGE
            tp = entry + atr * tp_atr
            sl = entry - atr * sl_atr
            max_loss = atr * max_loss_atr
            # 触发止盈 -> 止盈
            if next_h >= tp:
                profit = tp - entry
            # 没止盈：拿到收盘，以收盘价来结算，有最大亏损限制
            else:
                profit = max(next_c - entry, -max_loss)

            is_trade = True
            trades.append({
                "type": "long",
                "profit": profit,
                "tp": tp,
                "sl": sl,
            })
        
        elif prob < prob_short and close < ma20:
            entry = close - SPREAD - SLIPPAGE
            tp = entry - atr * tp_atr
            sl = entry + atr * sl_atr
            max_loss = atr * max_loss_atr

            # 触发止盈 -> 止盈
            if next_l <= tp:
                profit = entry - tp
            else:
                profit = max(entry - next_c, -max_loss)

            is_trade = True
            trades.append({
                "type": "short",
                "profit": profit,
                "tp": tp,
                "sl": sl,
            })

        # 统计最大连续亏损
        if is_trade:
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
            real_profit_loss_ratio = 0
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
    # 亏损次数
    lose_trades = (tdf['profit'] < 0).sum()
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


    print("===== 回测结果 =====")
    print(f"总交易次数(盈利/亏损)：{total_trades}({win_trades}/{lose_trades})")
    print(f"胜率：{win_rate:.2f}%")
    print(f"最大连续亏损：{max_consec_loss} 笔")
    print(f"最大回撤：{max_drawdown:.2f}")
    print(f"总收益：{total_profit:.2f}")
    print(f"实际平均盈亏比：{real_profit_loss_ratio:.2f}")
    print(f"理论盈亏比: {tp_atr}/{sl_atr}={tp_atr/sl_atr:.2f}")
    print(f'最大亏损限制：{max_loss_atr}倍atr')

    return BacktestResult(
        total = total_trades,
        profit = total_profit,
        win_rate = win_rate,
        max_consec_loss = max_consec_loss,
        max_drawdown = max_drawdown,
        real_profit_loss_ratio = real_profit_loss_ratio
    )


if __name__ == "__main__":
    args = get_args()
    run_backtest(args.period, args.step)