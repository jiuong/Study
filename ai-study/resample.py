"""
1分钟K线合成 15分钟 / 1小时周期
"""
import argparse

import pandas as pd

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

    # 重采样数据类型
    parser.add_argument(
        "--type",
        type=str,
        default="train",
        choices=["train", "predict", "backtest"],
        help="数据类型"
    )

    return parser.parse_args()


def resample(period: str, type: str) -> None:
    df = pd.read_csv(f"./raw/xauusd_1m_raw_{type}.csv")
    df.columns = ["datetime", "open", "high", "low", "close", "volume", "spread"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_target = df.set_index("datetime")

    if period != '1min':
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
    print('=================脚本开始运行===================')
    args = get_args()
    resample(args.period, args.type)