import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

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

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(data.index, data['close'], label='Close', linewidth=2, color='blue', alpha=0.7)
    ax.plot(data.index, data['ma_short'], label=f'{short_window}-Day MA', linewidth=2, color='orange')
    ax.plot(data.index, data['ma_long'],  label=f'{long_window}-Day MA', linewidth=2, color='green')

    if show_signals and 'signal' in data.columns:
        buy_sig = data[data['signal'] == 1]
        sell_sig = data[data['signal'] == -1]
        if not buy_sig.empty:
            ax.scatter(buy_sig.index, buy_sig['close'], label='Signal BUY', marker='^', color='lime', s=80, zorder=4, alpha=0.85)
        if not sell_sig.empty:
            ax.scatter(sell_sig.index, sell_sig['close'], label='Signal SELL', marker='v', color='tomato', s=80, zorder=4, alpha=0.85)

    if show_trades and tdf is not None and not tdf.empty:
        buy_tr = tdf[tdf['type'] == 'BUY'] if 'type' in tdf.columns else tdf.iloc[0:0]
        sell_tr = tdf[tdf['type'] == 'SELL'] if 'type' in tdf.columns else tdf.iloc[0:0]
        if not buy_tr.empty:
            ax.scatter(buy_tr.index, buy_tr['price'], label='Trade BUY', marker='^', color='green', s=120, zorder=6)
        if not sell_tr.empty:
            ax.scatter(sell_tr.index, sell_tr['price'], label='Trade SELL', marker='v', color='red', s=120, zorder=6)

        if annotate:
            for idx, row in buy_tr.iterrows():
                txt = f"BUY{idx:%Y-%m-%d}{row.get('shares', 0):.2f}@{row.get('price', 0):.2f}"
                ax.annotate(
                    txt,
                    xy=(idx, row.get('price', 0)),
                    xytext=(0, 12), textcoords='offset points',
                    fontsize=8, color='green',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='green', alpha=0.85)
                )
            for idx, row in sell_tr.iterrows():
                txt = f"SELL{idx:%Y-%m-%d}{row.get('shares', 0):.2f}@{row.get('price', 0):.2f}"
                ax.annotate(
                    txt,
                    xy=(idx, row.get('price', 0)),
                    xytext=(0, -30), textcoords='offset points',
                    fontsize=8, color='red',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='red', alpha=0.85)
                )

        if show_trade_vlines:
            for dt_ in tdf.index.unique():
                ax.axvline(x=dt_, color='gray', linestyle=':', alpha=0.25, linewidth=1)

    ax.set_title(f'{symbol} DMA Strategy (Signals vs Executed Trades)', fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend(fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

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