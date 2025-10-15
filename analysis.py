# dma_strategy/analysis.py
import os

import pandas as pd

from backtest import backtest
from metrics import calculate_performance_metrics
from plot import plot_strategy, plot_performance
from signals import calculate_moving_averages, generate_signals


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
    exec_delay_bars: int,
    allow_fractional: bool,
    last_n_days=None,
    save_comparison=True,
    save_individual=False,
    top_k: int = 5,
    ranking_metric: str = 'annualized_return',
    results_root: str = 'results'
):
    print(f"=== Unified grid analysis (Symbol={symbol}) ===")
    print(f"Short={short_windows}, Long={long_windows}")
    print(f"Buy={buy_fractions}, Sell={sell_fractions}, Exclusives={exclusives}")
    print(f"Cooldown={cooldown_bars}, Exec={exec_price}@delay={exec_delay_bars}, Fractional={allow_fractional}")

    all_metrics = []
    best_metrics = None
    best_bt_res = None
    best_df_sig = None
    best_trades_df = None
    best_params = None

    for s in short_windows:
        for l in long_windows:
            if s >= l:
                print(f" Skip invalid windows (short {s} ≥ long {l})")
                continue

            df_ma  = calculate_moving_averages(stock_data, s, l)
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
                            exec_delay_bars=exec_delay_bars,
                            exclusive=ex
                        )

                        metrics = calculate_performance_metrics(bt_res, trades_df)
                        metrics.update({
                            'symbol': symbol,
                            'short_window': s, 'long_window': l,
                            'buy_fraction': bf, 'sell_fraction': sf, 'exclusive': ex,
                            'cooldown_bars': cooldown_bars,
                            'exec_price': exec_price, 'exec_delay_bars': exec_delay_bars,
                            'allow_fractional': allow_fractional,
                            'transaction_cost': transaction_cost,
                            'initial_capital': initial_capital,
                        })
                        all_metrics.append(metrics)

                        if save_individual:
                            from .io_utils import save_backtest_results
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
        'cooldown_bars','exec_price','exec_delay_bars','allow_fractional',
        'annualized_return','total_return','max_drawdown','sharpe_ratio',
        'win_rate','num_trades','benchmark_annualized','benchmark_return','time_in_market'
    ]
    metrics_comparison = metrics_comparison[key_cols + [c for c in metrics_comparison.columns if c not in key_cols]]

    if save_comparison:
        comp_dir = os.path.join(results_root, 'parameter_comparison')
        os.makedirs(comp_dir, exist_ok=True)
        comp_path = os.path.join(comp_dir, f"{symbol}_grid_comparison.csv")
        metrics_comparison.to_csv(comp_path, index=False, encoding='utf-8')
        print(f"All unified-grid comparisons saved at: {comp_path}")

    topK = metrics_comparison.sort_values(ranking_metric, ascending=False).head(top_k)
    print(f"=== Top{top_k} by {ranking_metric} ===")
    print(topK[['short_window','long_window','buy_fraction','sell_fraction','exclusive',
                'annualized_return','max_drawdown','sharpe_ratio','win_rate','num_trades','time_in_market']])

    if best_metrics is not None:
        s, l, bf, sf, ex = best_params
        save_root = os.path.join(results_root, 'plots', symbol)
        os.makedirs(save_root, exist_ok=True)
        suffix = f"short{s}_long{l}_bf{bf}_sf{sf}_ex{ex}_cd{cooldown_bars}_exec{exec_price}"
        strat_path = os.path.join(save_root, f"{symbol}_strategy_{suffix}.png")
        perf_path  = os.path.join(save_root, f"{symbol}_performance_{suffix}.png")

        print(trades_df)
        plot_strategy(best_df_sig, symbol, s, l,
                      last_n_days=last_n_days,
                      save_path=strat_path,
                      trades_df=best_trades_df,
                      show_signals=True,
                      show_trades=True,
                      annotate=True,
                      show_trade_vlines=True)
        plot_performance(best_bt_res, symbol, save_path=perf_path)

        print("=== Best combo (Top1) ===")
        print(best_metrics)
        print(f"Strategy plot saved:   {strat_path}")
        print(f"Performance plot saved:{perf_path}")

    print("=== Unified grid analysis finished ===")
    return metrics_comparison, topK, best_metrics, best_bt_res, best_df_sig
