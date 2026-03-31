"""
数据加载、清洗、指标计算
"""
import pandas as pd


def load_data(period: str, type: str) -> pd.DataFrame:
    df = pd.read_csv(f"./{type}/data/xauusd_{period}.csv")
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    # 下根涨 = 1，不涨 = 0
    df['prob'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df

import pandas as pd

def clean_gap_data(df, period) -> pd.DataFrame:
    """
    清洗K线数据的时间断层
    df: 必须包含 datetime, open, high, low, close, volume
    period: 周期
    返回清洗后的连续 df
    """
    # 1. 确保时间列是 datetime 并排序
    df = df.copy()
    print(df.columns)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 2. 获取周期对应的时间频率字符串
    freq_map = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d"
    }
    freq = freq_map[period]

    # 3. 生成完整连续的时间索引
    start = df["datetime"].min()
    end   = df["datetime"].max()
    full_index = pd.date_range(start=start, end=end, freq=freq)

    # 4. 把原数据重索引 → 缺失时间行会变成 NaN
    df = df.set_index("datetime").reindex(full_index)
    df.index.name = "datetime"

    # 5. 标记断层（可选，用于观察）
    df["is_gap"] = df["open"].isna()

    # 6. 填充断层（K线标准做法：用前一根close填充价格）
    # 6. 填充断层（K线标准做法：用前一根close填充价格）
    df["close"] = df["close"].ffill()
    df["open"]  = df["open"].fillna(df["close"])
    df["high"]  = df["high"].fillna(df["close"])
    df["low"]   = df["low"].fillna(df["close"])

    # 7. 断层交易量填0
    df["volume"] = df["volume"].fillna(0)

    # 8. 去掉最开头无法填充的 NaN
    df = df.dropna()

    # 9. 恢复列顺序
    df = df.reset_index()
    return df



def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # MA20
    df["ma20"] = c.rolling(20).mean()

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # ATR(14)
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # 成交量
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20']

    df = df.dropna().reset_index(drop=True)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df["next_o"] = df["open"].shift(-1)
    df["next_h"] = df["high"].shift(-1)
    df["next_l"] = df["low"].shift(-1)
    df["next_c"] = df["close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

def process_full(period: str, type: str) -> pd.DataFrame:
    data = load_data(period, type)
    data = clean_gap_data(data, period)
    data = compute_indicators(data)
    return build_target(data)


if __name__ == "__main__":
    data = process_full("15min", "train")
    print(f"数据处理完成，共 {len(data)} 行")