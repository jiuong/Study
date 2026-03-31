import pandas as pd
import numpy as np

def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high = df['high']
    low = df['low']

    # ========== 5根K线峰谷判断 ==========
    pivot_high = np.zeros(len(df))
    pivot_low  = np.zeros(len(df))

    # 从第3根到倒数第3根（左右各留2根）
    for i in range(2, len(df)-2):
        # 峰（高点）
        if (high[i] > high[i-1] and high[i] > high[i-2] and
            high[i] > high[i+1] and high[i] > high[i+2]):
            pivot_high[i] = 1

        # 谷（低点）
        if (low[i] < low[i-1] and low[i] < low[i-2] and
            low[i] < low[i+1] and low[i] < low[i+2]):
            pivot_low[i] = 1

    df['is_pivot_high'] = pivot_high
    df['is_pivot_low']  = pivot_low

    # ========== 提取最近的波段高低点 ==========
    # 最近有效高点
    df['recent_pivot_high'] = np.nan
    last_high = np.nan
    # 最近有效低点
    df['recent_pivot_low']  = np.nan
    last_low  = np.nan

    for i in range(len(df)):
        if df['is_pivot_high'].iloc[i] == 1:
            last_high = df['high'].iloc[i]
        if df['is_pivot_low'].iloc[i] == 1:
            last_low  = df['low'].iloc[i]

        df['recent_pivot_high'].iloc[i] = last_high
        df['recent_pivot_low'].iloc[i]  = last_low

    return df


# 基于【波段高低点】计算斐波那契
def add_fib_by_pivots(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = add_pivot_points(df_copy)
    h = df_copy['recent_pivot_high']
    l = df_copy['recent_pivot_low']
    r = h - l

    df['fib_618'] = h - 0.618 * r
    df['fib_500'] = h - 0.500 * r
    df['fib_382'] = h - 0.382 * r

    return df
