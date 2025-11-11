import os
import pandas as pd
import yfinance as yf


def fetch_stock_data(symbol, period='3y',
                     save_local=True, data_dir='data', use_adjusted=True) -> pd.DataFrame:
    if save_local and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    filename = f"{symbol}_{period}.csv"
    filepath = os.path.join(data_dir, filename)

    if save_local and os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
        return df

    raw = yf.download(symbol, period = period, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No data found for {symbol}.")
    raw = raw.sort_index()
    
    if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

    close_col = 'Adj Close' if use_adjusted and 'Adj Close' in raw.columns else 'Close'
    df = raw.rename(columns={'Open':'open','High':'high','Low':'low', close_col:'close','Volume':'volume'})[['open','high','low','close','volume']].copy()
    df.index.name = 'date'

    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close'], inplace=True)

    if save_local:
        df.reset_index().to_csv(filepath, index=False)

    return df