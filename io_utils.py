import os
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_backtest_results(backtest_results, trades_df, metrics, symbol, short_window, long_window, root='results'):
    param_dir = f"{symbol}_short{short_window}_long{long_window}"
    full_dir = os.path.join(root, param_dir)
    ensure_dir(full_dir)

    backtest_path = os.path.join(full_dir, f"{param_dir}_daily_results.csv")
    backtest_results.to_csv(backtest_path, encoding='utf-8')

    if not trades_df.empty:
        trades_path = os.path.join(full_dir, f"{param_dir}_trades.csv")
        trades_df.to_csv(trades_path, encoding='utf-8')

    metrics_df = pd.DataFrame([metrics])
    if 'symbol' not in metrics_df.columns:
        metrics_df.insert(0, 'symbol', symbol)
    if 'short_window' not in metrics_df.columns:
        metrics_df.insert(1, 'short_window', short_window)
    if 'long_window' not in metrics_df.columns:
        metrics_df.insert(2, 'long_window', long_window)

    metrics_path = os.path.join(full_dir, f"{param_dir}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')

    print(f" Backtest results saved at: {backtest_path}")
    if not trades_df.empty:
        print(f" Trading records saved at: {trades_path}")
    print(f" Metrics saved at: {metrics_path}")