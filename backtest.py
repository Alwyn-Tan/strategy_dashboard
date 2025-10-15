import numpy as np
import pandas as pd

def backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    buy_fraction: float = 1.0,
    sell_fraction: float = 1.0,
    allow_fractional: bool = True,
    exec_price: str = 'open',
    exec_delay_bars: int = 1,
    exclusive: bool = False
):
    # 夹取比例边界
    buy_fraction  = float(max(0.0, min(1.0, buy_fraction)))
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

        # 执行延迟后的信号（次日开盘等）
        if i - exec_delay_bars >= 0:
            sig = data['signal'].iloc[i - exec_delay_bars]
            px  = data[exec_price].iloc[i]

            if sig == 1:
                if (not exclusive) or (exclusive and shares == 0):
                    cash_to_use = cash * buy_fraction
                    eff_px = px * (1 + transaction_cost)
                    if allow_fractional:
                        qty = cash_to_use / eff_px
                    else:
                        qty = int(cash_to_use // eff_px)
                    if qty > 0:
                        cost = qty * px * (1 + transaction_cost)
                        cash -= cost
                        shares += qty
                        trades.append({'date': date,'type':'BUY','price':px,'shares':qty,'cost':cost,'cash_after':cash,'shares_after':shares})

            elif sig == -1 and shares > 0:
                qty = shares * sell_fraction
                if not allow_fractional:
                    qty = int(qty)
                qty = max(0, min(qty, shares))
                if qty > 0:
                    revenue = qty * px * (1 - transaction_cost)
                    cash += revenue
                    shares -= qty
                    trades.append({'date': date,'type':'SELL','price':px,'shares':qty,'revenue':revenue,'cash_after':cash,'shares_after':shares})

        # 基准（收盘对收盘）
        close_px = data['close'].iloc[i]
        daily_ret = (close_px / prev_close) - 1.0
        bench_cum *= (1 + daily_ret)
        data.at[date,'benchmark_cum_returns'] = bench_cum
        prev_close = close_px

        # 盘后估值与策略收益
        total = cash + shares * close_px
        data.at[date,'cash'] = cash
        data.at[date,'shares'] = shares
        data.at[date,'total'] = total

        if i > 0:
            prev_val = data['total'].iloc[i-1]
            ret = (total / prev_val - 1.0) if prev_val != 0 else 0.0
            data.at[date,'returns'] = ret
            data.at[date,'strategy_cum_returns'] = data['strategy_cum_returns'].iloc[i-1] * (1 + ret)

    trades_df = pd.DataFrame(trades).set_index('date') if trades else pd.DataFrame()
    return data, trades_df