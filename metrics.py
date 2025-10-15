# dma_strategy/metrics.py
import numpy as np

def calculate_performance_metrics(backtest_results, trades_df):
    import pandas as pd

    start_date = backtest_results.index[0]
    end_date   = backtest_results.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25 if days > 0 else 0

    initial_val = backtest_results['total'].iloc[0]
    final_val   = backtest_results['total'].iloc[-1]
    total_return = (final_val - initial_val) / initial_val * 100 if initial_val != 0 else 0
    annualized_return = (((1 + total_return/100) ** (1/years)) - 1) * 100 if years > 0 else 0

    bench_ret = (backtest_results['benchmark_cum_returns'].iloc[-1] - 1) * 100
    bench_ann = (((1 + bench_ret/100) ** (1/years)) - 1) * 100 if years > 0 else 0

    daily_returns = backtest_results['returns'].dropna()
    sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0

    rolling_max = backtest_results['total'].cummax()
    max_drawdown = (backtest_results['total'] / rolling_max - 1.0).min() * 100

    num_trades = len(trades_df)
    win_rate = 0
    avg_trade_return = 0
    if num_trades >= 2:
        wins = 0
        total_trade_ret = 0
        for i in range(0, len(trades_df), 2):
            if i + 1 >= len(trades_df): break
            buy = trades_df.iloc[i]
            sell = trades_df.iloc[i+1]
            if sell.get('revenue', 0) > buy.get('cost', 0): wins += 1
            tr = (sell.get('revenue', 0) - buy.get('cost', 0)) / max(buy.get('cost', 1e-9), 1e-9) * 100
            total_trade_ret += tr
        win_rate = (wins / (num_trades / 2)) * 100 if num_trades > 0 else 0
        avg_trade_return = total_trade_ret / (num_trades / 2) if num_trades > 0 else 0

    # 补充：平均持仓比例（time-in-market）
    time_in_market = ((backtest_results['shares'] * backtest_results['close']) / backtest_results['total']).fillna(0).mean() * 100

    return {
        'start_date': start_date.date(),
        'end_date': end_date.date(),
        'years_traded': round(years, 2),
        'total_return': round(total_return, 2),
        'annualized_return': round(annualized_return, 2),
        'benchmark_return': round(bench_ret, 2),
        'benchmark_annualized': round(bench_ann, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown': round(max_drawdown, 2),
        'num_trades': int(num_trades),
        'win_rate': round(win_rate, 2),
        'avg_trade_return': round(avg_trade_return, 2),
        'time_in_market': round(time_in_market, 2),
    }