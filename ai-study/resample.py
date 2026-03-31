"""
1分钟K线合成 15分钟 / 1小时周期
"""
import argparse
import pandas as pd
from args import get_args


def resample(period: str, type: str) -> None:
    df = pd.read_csv(f"./raw/xauusd_1m_raw_{type}.csv")
    df.columns = ["datetime", "open", "high", "low", "close", "volume", "spread"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_target = df.set_index("datetime")

    # if period != '1min':
    df_target = df_target.resample(period).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    df_target.reset_index().to_csv(f"./{type}/data/xauusd_{period}.csv", index=False)
    print(f"已生成 {period} 数据: ./{type}/data_/xauusd_{period}.csv")



if __name__ == "__main__":
    period = get_args().period
    resample(period, 'train')
    resample(period, 'backtest')
    resample(period, 'predict')