import random

from config import (
  PROB_PAIRS,
  TP_SL_ATR_PAIRS, MAX_SL_ATR_LIST,
  FIBO_POSITION_LIST,
)
from tools import get_base3_digits

class TradeArgs:
    def __init__(
        self,
        long: float = 0,
        short: float = 0,
        tp_atr: float = 0,
        sl_atr: float = 0,
        max_sl_atr: float = 0,
        fibo_pos: tuple[float, float] = FIBO_POSITION_LIST[0],
    ):
        self.long = long
        self.short = short
        self.tp_atr = tp_atr
        # 默认2倍盈亏比
        self.sl_atr = sl_atr
        self.max_sl_atr = max_sl_atr or 2 * sl_atr
        self.fibo_pos = fibo_pos

def print_trade_args(args: TradeArgs):
    print('=========交易策略==========')
    print(f'做多入场阈值：{args.long:.2f}')
    print(f'做空入场阈值：{args.short:.2f}')
    # print(f'斐波那契入场区间：[{args.fibo_pos[0]},{args.fibo_pos[1]}]')
    print(f'理论盈亏比：({args.tp_atr}/{args.sl_atr}){args.tp_atr/args.sl_atr:.2f}')
    print(f'最大亏损限制：{args.max_sl_atr}倍atr')


def get_trade_args(directive: int = 0) -> TradeArgs:
    if directive > 3 ** 3:
        return None
    max_sl_atr_idx, tp_sl_atr_idx, prob_idx = get_base3_digits(directive, 3)
    long, short = PROB_PAIRS[prob_idx]
    tp_atr, sl_atr = TP_SL_ATR_PAIRS[tp_sl_atr_idx]
    max_sl_atr = MAX_SL_ATR_LIST[max_sl_atr_idx]
    # fibo_pos = FIBO_POSITION_LIST[idx % 3]
    args = TradeArgs(
        long = long,
        short = short,
        tp_atr = tp_atr,
        sl_atr = sl_atr,
        max_sl_atr = max_sl_atr,
        # fibo_pos = fibo_pos
    )

    return args