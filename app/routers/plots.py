import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core.strategy import (
    fetch_stock_data,
    calculate_moving_averages,
    generate_signals,
    backtest,
    plot_strategy,
    plot_performance,
)
from ..schemas import PlotRequest

router = APIRouter()

@router.post('/strategy')
def strategy_plot(req: PlotRequest):
    try:
        data = fetch_stock_data(req.symbol, period=req.period)
        df_ma = calculate_moving_averages(data, req.short_window, req.long_window)
        df_sig = generate_signals(df_ma, cooldown_bars=req.cooldown_bars)
        bt_res, trades_df = backtest(
            df=df_sig,
            initial_capital=10000.0,
            transaction_cost=0.001,
            buy_fraction=req.buy_fraction,
            sell_fraction=req.sell_fraction,
            allow_fractional=req.allow_fractional,
            exec_price=req.exec_price,
            exec_delay_bars=req.exec_delay_bars,
            exclusive=req.exclusive
        )
        os.makedirs(os.path.join(req.results_root, 'plots', req.symbol), exist_ok=True)
        save_path = os.path.join(req.results_root, 'plots', req.symbol, 'strategy.png')
        plot_strategy(df_sig, req.symbol, req.short_window, req.long_window,
                      last_n_days=req.last_n_days, save_path=save_path, trades_df=trades_df)
        return FileResponse(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/performance')
def performance_plot(req: PlotRequest):
    try:
        data = fetch_stock_data(req.symbol, period=req.period)
        df_ma = calculate_moving_averages(data, req.short_window, req.long_window)
        df_sig = generate_signals(df_ma, cooldown_bars=req.cooldown_bars)
        bt_res, trades_df = backtest(
            df=df_sig,
            initial_capital=10000.0,
            transaction_cost=0.001,
            buy_fraction=req.buy_fraction,
            sell_fraction=req.sell_fraction,
            allow_fractional=req.allow_fractional,
            exec_price=req.exec_price,
            exec_delay_bars=req.exec_delay_bars,
            exclusive=req.exclusive
        )
        os.makedirs(os.path.join(req.results_root, 'plots', req.symbol), exist_ok=True)
        save_path = os.path.join(req.results_root, 'plots', req.symbol, 'performance.png')
        plot_performance(bt_res, req.symbol, save_path=save_path)
        return FileResponse(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
