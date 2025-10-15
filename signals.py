import pandas as pd

def calculate_moving_averages(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    data = df.copy()
    data['ma_short'] = data['close'].rolling(window=int(short_window), min_periods=1).mean()
    data['ma_long']  = data['close'].rolling(window=int(long_window),  min_periods=1).mean()
    return data

def generate_signals(df: pd.DataFrame, cooldown_bars: int = 0) -> pd.DataFrame:
    data = df.copy()
    data['signal'] = 0
    last_trade_i = None

    for i in range(1, len(data)):
        cur, prev = data.index[i], data.index[i-1]
        cross_up   = (data.loc[cur,'ma_short'] > data.loc[cur,'ma_long']) and (data.loc[prev,'ma_short'] <= data.loc[prev,'ma_long'])
        cross_down = (data.loc[cur,'ma_short'] < data.loc[cur,'ma_long']) and (data.loc[prev,'ma_short'] >= data.loc[prev,'ma_long'])

        if cooldown_bars > 0 and last_trade_i is not None:
            if i - last_trade_i < cooldown_bars:
                continue

        if cross_up:
            data.loc[cur,'signal'] = 1
            last_trade_i = i
        elif cross_down:
            data.loc[cur,'signal'] = -1
            last_trade_i = i

    return data