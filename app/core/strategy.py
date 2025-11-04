import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from .io_utils import save_backtest_results


def fetch_stock_data(symbol, start_date=None, end_date=None, period='3y', save_local=True, data_dir='data', use_adjusted=True) -> pd.DataFrame:
    if save_local and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    filename = f"{symbol}_{start_date}_{end_date}.csv" if (start_date and end_date) else f"{symbol}_{period}.csv"
    filepath = os.path.join(data_dir, filename)

    if save_local and os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
        return df

    raw = yf.download(symbol, start=start_date, end=end_date, period=None if start_date else period, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No data found for {symbol}.")
    raw = raw.sort_index()

    close_col = 'Adj Close' if use_adjusted and 'Adj Close' in raw.columns else 'Close'
    df = raw.rename(columns={'Open':'open','High':'high','Low':'low', close_col:'close','Volume':'volume'})[['open','high','low','close','volume']].copy()
    df.index.name = 'date'

    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open','high','low','close'], inplace=True)

    if save_local:
        df.reset_index().to_csv(filepath, index=False)

    return df

def calculate_moving_averages(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    data = df.copy()
    data['ma_short'] = data['close'].rolling(window=int(short_window), min_periods=1).mean()
    data['ma_long'] = data['close'].rolling(window=int(long_window), min_periods=1).mean()
    return data

def generate_signals(df: pd.DataFrame, cooldown_bars: int = 0) -> pd.DataFrame:
    data = df.copy()
    data['signal'] = 0
    last_trade_i = None

    for i in range(1, len(data)):
        cur, prev = data.index[i], data.index[i-1]
        cross_up = (data.loc[cur,'ma_short'] > data.loc[cur,'ma_long']) and (data.loc[prev,'ma_short'] <= data.loc[prev,'ma_long'])
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

def backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    buy_fraction: float = 1.0,
    sell_fraction: float = 1.0,
    allow_fractional: bool = True,
    exec_price: str = 'open',
    exec_delay: int = 1,
    exclusive: bool = False
):
    buy_fraction: float = float(max(0.0, min(1.0, buy_fraction)))
    sell_fraction = float(max(0.0, min(1.0, sell_fraction)))
    required = {'close','signal'} | ({'open'} if exec_price=='open' else set())
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame must contain columns: {required}")

    data = df.copy()
    for col in ['cash','shares','total','returns','strategy_cum_returns','benchmark_cum_returns']:
        data[col] = np.nan
    data['strategy_cum_returns'] = 1.0
    data['benchmark_cum_returns'] = 1.0
    cash = initial_capital
    shares = 0.0
    prev_close = data['close'].iloc[0]
    bench_cum = 1.0
    trades = []
    
    for i in range(len(data)):
        date = data.index[i]

        if i - exec_delay >= 0:
            sig = data['signal'].iloc[i - exec_delay]
            price = data[exec_price].iloc[i]

            if sig == 1:
                if (not exclusive) or (exclusive and shares == 0):
                    cash_to_use = float(cash) * float(buy_fraction)
                    eff_price = price * (1 + transaction_cost)
                    quantity = cash_to_use / eff_price if allow_fractional else int(cash_to_use // eff_price)

                    if quantity > 0:
                        cost = quantity * price * (1 + transaction_cost)
                        cash -= cost
                        shares += quantity
                        trades.append({'date': date,'type':'BUY','price':price,'shares':quantity,'cost':cost,'cash_after':cash,'shares_after':shares})

            elif sig == -1 and shares > 0:
                quantity = shares * sell_fraction

                if not allow_fractional:
                    quantity = int(quantity)
                quantity = max(0, min(quantity, shares))

                if quantity > 0:
                    revenue = quantity * price * (1 - transaction_cost)
                    cash += revenue
                    shares -= quantity
                    trades.append({'date': date,'type':'SELL','price':price,'shares':quantity,'revenue':revenue,'cash_after':cash,'shares_after':shares})

        close_price = data['close'].iloc[i]
        daily_return = (close_price / prev_close) - 1.0
        bench_cum *= (1 + daily_return)
        data.at[date,'benchmark_cum_returns'] = bench_cum
        prev_close = close_price
        total = cash + shares * close_price
        data.at[date,'cash'] = cash
        data.at[date,'shares'] = shares
        data.at[date,'total'] = total

        if i > 0:
            prev_total = data['total'].iloc[i-1]
            return_rate = (total / prev_total - 1.0) if prev_total != 0 else 0.0
            data.at[date,'returns'] = return_rate
            data.at[date,'strategy_cum_returns'] = data['strategy_cum_returns'].iloc[i-1] * (1 + return_rate)

    trades_df = pd.DataFrame(trades).set_index('date') if trades else pd.DataFrame()

    return data, trades_df

def calculate_performance_metrics(backtest_results, trades_df):
    start_date = backtest_results.index[0]
    end_date = backtest_results.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25 if days > 0 else 0
    initial_val = backtest_results['total'].iloc[0]
    final_val = backtest_results['total'].iloc[-1]
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
            if i + 1 >= len(trades_df):
                break
            buy = trades_df.iloc[i]
            sell = trades_df.iloc[i+1]
            if sell.get('revenue', 0) > buy.get('cost', 0):
                wins += 1
            tr = (sell.get('revenue', 0) - buy.get('cost', 0)) / max(buy.get('cost', 1e-9), 1e-9) * 100
            total_trade_ret += tr
        win_rate = (wins / (num_trades / 2)) * 100 if num_trades > 0 else 0
        avg_trade_return = total_trade_ret / (num_trades / 2) if num_trades > 0 else 0

    time_in_market = ((backtest_results['shares'] * backtest_results['close']) / backtest_results['total']).fillna(0).mean() * 100

    return {
        'start_date': start_date.date(),
        'end_date': end_date.date(),
        'years_traded': round(years, 2),
        'total_return_rate': round(total_return, 2),
        'annualized_return_rate': round(annualized_return, 2),
        'benchmark_return': round(bench_ret, 2),
        'benchmark_annualized': round(bench_ann, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown': round(max_drawdown, 2),
        'num_trades': int(num_trades),
        'win_rate': round(win_rate, 2),
        'avg_trade_return': round(avg_trade_return, 2),
        'time_in_market': round(time_in_market, 2),
    }

def analyze_grid(
    stock_data,
    symbol: str,
    short_windows,
    long_windows,
    buy_fractions,
    sell_fractions,
    exclusives,
    cooldown_bars: int,
    initial_capital: float,
    transaction_cost: float,
    exec_price: str,
    exec_delay: int,
    allow_fractional: bool,
    last_n_days=None,
    save_comparison=True,
    save_individual=False,
    top_k: int = 5,
    ranking_metric: str = 'annualized_return',
    results_root: str = 'results'
):
    all_metrics = []
    best_metrics = None
    best_bt_res = None
    best_df_sig = None
    best_trades_df = None
    best_params = None

    for s in short_windows:
        for l in long_windows:
            if s >= l:
                continue
            df_ma = calculate_moving_averages(stock_data, s, l)
            df_sig = generate_signals(df_ma, cooldown_bars=cooldown_bars)
            for bf in buy_fractions:
                for sf in sell_fractions:
                    for ex in exclusives:
                        bt_res, trades_df = backtest(
                            df=df_sig,
                            initial_capital=initial_capital,
                            transaction_cost=transaction_cost,
                            buy_fraction=bf,
                            sell_fraction=sf,
                            allow_fractional=allow_fractional,
                            exec_price=exec_price,
                            exec_delay=exec_delay,
                            exclusive=ex
                        )
                        metrics = calculate_performance_metrics(bt_res, trades_df)
                        metrics.update({
                            'symbol': symbol,
                            'short_window': s, 'long_window': l,
                            'buy_fraction': bf, 'sell_fraction': sf, 'exclusive': ex,
                            'cooldown_bars': cooldown_bars,
                            'exec_price': exec_price, 'exec_delay': exec_delay,
                            'allow_fractional': allow_fractional,
                            'transaction_cost': transaction_cost,
                            'initial_capital': initial_capital,
                        })
                        all_metrics.append(metrics)
                        if save_individual:
                            save_backtest_results(bt_res, trades_df, metrics, symbol, s, l, root=results_root)
                        if (best_metrics is None) or (metrics.get(ranking_metric, float('-inf')) > best_metrics.get(ranking_metric, float('-inf'))):
                            best_metrics = metrics
                            best_bt_res = bt_res
                            best_df_sig = df_sig
                            best_trades_df = trades_df
                            best_params = (s, l, bf, sf, ex)
    metrics_comparison = pd.DataFrame(all_metrics)
    key_cols = [
        'symbol','short_window','long_window','buy_fraction','sell_fraction','exclusive',
        'cooldown_bars','exec_price','exec_delay','allow_fractional',
        'annualized_return','total_return','max_drawdown','sharpe_ratio',
        'win_rate','num_trades','benchmark_annualized','benchmark_return','time_in_market'
    ]
    metrics_comparison = metrics_comparison[key_cols + [c for c in metrics_comparison.columns if c not in key_cols]]
    if save_comparison:
        comp_dir = os.path.join(results_root, 'parameter_comparison')
        os.makedirs(comp_dir, exist_ok=True)
        comp_path = os.path.join(comp_dir, f"{symbol}_grid_comparison.csv")
        metrics_comparison.to_csv(comp_path, index=False, encoding='utf-8')
    return metrics_comparison, metrics_comparison.sort_values(ranking_metric, ascending=False).head(top_k), best_metrics, best_bt_res, best_df_sig

def plot_strategy(
    df,
    symbol,
    short_window,
    long_window,
    last_n_days=None,
    save_path=None,
    trades_df=None,
    show_signals=True,
    show_trades=True,
    annotate=True,
    show_trade_vlines=False
):
    import datetime as dt
    import pandas as pd
    data = df.copy()
    tdf = trades_df.copy() if isinstance(trades_df, pd.DataFrame) and not trades_df.empty else None
    if last_n_days:
        cutoff = data.index.max() - dt.timedelta(days=last_n_days)
        data = data[data.index >= cutoff]
        if tdf is not None:
            tdf = tdf[tdf.index >= cutoff]
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(data.index, data['close'], label='Close', linewidth=2, color='blue', alpha=0.7)
    ax.plot(data.index, data['ma_short'], label=f'{short_window}-Day MA', linewidth=2, color='orange')
    ax.plot(data.index, data['ma_long'], label=f'{long_window}-Day MA', linewidth=2, color='green')
    if show_signals and 'signal' in data.columns:
        buy_sig = data[data['signal'] == 1]
        sell_sig = data[data['signal'] == -1]
        if not buy_sig.empty:
            ax.scatter(buy_sig.index, buy_sig['close'], label='Signal BUY', marker='^', color='lime', s=80, zorder=4, alpha=0.8)
        if not sell_sig.empty:
            ax.scatter(sell_sig.index, sell_sig['close'], label='Signal SELL', marker='v', color='tomato', s=80, zorder=4, alpha=0.8)
    if show_trades and tdf is not None and not tdf.empty:
        buy_tr = tdf[tdf['type'] == 'BUY'] if 'type' in tdf.columns else tdf.iloc[0:0]
        sell_tr= tdf[tdf['type'] == 'SELL'] if 'type' in tdf.columns else tdf.iloc[0:0]
        if not buy_tr.empty:
            ax.scatter(buy_tr.index, buy_tr['price'], label='Trade BUY', marker='^', color='green', s=120, zorder=6)
        if not sell_tr.empty:
            ax.scatter(sell_tr.index, sell_tr['price'], label='Trade SELL', marker='v', color='red', s=120, zorder=6)
        if annotate:
            for idx, row in buy_tr.iterrows():
                txt = f"BUY\n{idx:%Y-%m-%d}\n{row.get('shares', 0):.2f}@{row.get('price', 0):.2f}"
                ax.annotate(txt, xy=(idx, row.get('price', 0)), xytext=(0, 12), textcoords='offset points',
                            fontsize=8, color='green', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='green', alpha=0.8))
            for idx, row in sell_tr.iterrows():
                txt = f"SELL\n{idx:%Y-%m-%d}\n{row.get('shares', 0):.2f}@{row.get('price', 0):.2f}"
                ax.annotate(txt, xy=(idx, row.get('price', 0)), xytext=(0, -30), textcoords='offset points',
                            fontsize=8, color='red', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='red', alpha=0.8))
        if show_trade_vlines:
            for dt_ in tdf.index.unique():
                ax.axvline(x=dt_, color='gray', linestyle=':', alpha=0.25, linewidth=1)
    ax.set_title(f'{symbol} DMA Strategy (Signals vs Executed Trades)', fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=14); ax.set_ylabel('Price', fontsize=14)
    ax.legend(fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_performance(backtest_results, symbol, save_path=None):
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(backtest_results.index, backtest_results['strategy_cum_returns'], label='Strategy', linewidth=2, color='blue')
    ax.plot(backtest_results.index, backtest_results['benchmark_cum_returns'], label='Buy & Hold', linewidth=2, color='gray', linestyle='--')
    ax.set_title(f'{symbol} Strategy Performance vs Benchmark', fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=14); ax.set_ylabel('Cumulative Returns', fontsize=14)
    ax.legend(fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
