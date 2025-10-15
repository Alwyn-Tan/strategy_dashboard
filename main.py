from config import DataConfig, SignalConfig, ExecConfig, TradeConfig, OutputConfig
from data import fetch_stock_data
from analysis import analyze_grid

def run():
    data_cfg  = DataConfig(symbol="AAPL", period="3y", use_adjusted=True)
    sig_cfg   = SignalConfig(short_windows=[5,10,15,20], long_windows=[30,50,60,80,100], cooldown_bars=5)
    exec_cfg  = ExecConfig(exec_price='open', exec_delay_bars=1)
    trade_cfg = TradeConfig(initial_capital=10000.0, transaction_cost=0.001,
                            allow_fractional=True,
                            buy_fractions=[0.25,0.5,0.75], sell_fractions=[0.25,0.5,0.75],
                            exclusives=[False, True])
    out_cfg   = OutputConfig(results_dir='results', last_n_days=500, save_individual=False,
                             save_comparison=True, top_k=5, ranking_metric='annualized_return')

    print(f"Fetching data for {data_cfg.symbol}...")
    stock = fetch_stock_data(data_cfg.symbol, period=data_cfg.period, use_adjusted=data_cfg.use_adjusted, data_dir=data_cfg.data_dir)

    analyze_grid(
        stock_data=stock,
        symbol=data_cfg.symbol,
        short_windows=sig_cfg.short_windows,
        long_windows=sig_cfg.long_windows,
        buy_fractions=trade_cfg.buy_fractions,
        sell_fractions=trade_cfg.sell_fractions,
        exclusives=trade_cfg.exclusives,
        cooldown_bars=sig_cfg.cooldown_bars,
        initial_capital=trade_cfg.initial_capital,
        transaction_cost=trade_cfg.transaction_cost,
        exec_price=exec_cfg.exec_price,
        exec_delay_bars=exec_cfg.exec_delay_bars,
        allow_fractional=trade_cfg.allow_fractional,
        last_n_days=out_cfg.last_n_days,
        save_comparison=out_cfg.save_comparison,
        save_individual=out_cfg.save_individual,
        top_k=out_cfg.top_k,
        ranking_metric=out_cfg.ranking_metric,
        results_root=out_cfg.results_dir
    )

if __name__ == "__main__":
    run()