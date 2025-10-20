from fastapi import APIRouter, HTTPException

from ..core.strategy import (
    fetch_stock_data,
    calculate_moving_averages,
    generate_signals,
    backtest,
    calculate_performance_metrics,
    analyze_grid,
)
from ..schemas import SingleBacktestRequest, GridBacktestRequest

router = APIRouter()

@router.post('/single')
def run_single(req: SingleBacktestRequest):
    try:
        data = fetch_stock_data(req.symbol, period=req.period, use_adjusted=req.use_adjusted)
        df_ma = calculate_moving_averages(data, req.short_window, req.long_window)
        df_sig = generate_signals(df_ma, cooldown_bars=req.cooldown_bars)
        bt_res, trades_df = backtest(
            df=df_sig,
            initial_capital=req.initial_capital,
            transaction_cost=req.transaction_cost,
            buy_fraction=req.buy_fraction,
            sell_fraction=req.sell_fraction,
            allow_fractional=req.allow_fractional,
            exec_price=req.exec_price,
            exec_delay_bars=req.exec_delay_bars,
            exclusive=req.exclusive
        )
        metrics = calculate_performance_metrics(bt_res, trades_df)
        response = {
            "metrics": metrics,
            "trades": trades_df.reset_index().to_dict(orient="records"),
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/grid')
def run_grid(req: GridBacktestRequest):
    try:
        data = fetch_stock_data(req.symbol, period=req.period, use_adjusted=req.use_adjusted)
        metrics_comparison, topK, best_metrics, best_bt_res, best_df_sig = analyze_grid(
            stock_data=data,
            symbol=req.symbol,
            short_windows=req.short_windows,
            long_windows=req.long_windows,
            buy_fractions=req.buy_fractions,
            sell_fractions=req.sell_fractions,
            exclusives=req.exclusives,
            cooldown_bars=req.cooldown_bars,
            initial_capital=req.initial_capital,
            transaction_cost=req.transaction_cost,
            exec_price=req.exec_price,
            exec_delay_bars=req.exec_delay_bars,
            allow_fractional=req.allow_fractional,
            last_n_days=req.last_n_days,
            save_comparison=req.save_comparison,
            save_individual=req.save_individual,
            top_k=req.top_k,
            ranking_metric=req.ranking_metric,
            results_root=req.results_root
        )
        return {
            "comparison": metrics_comparison.to_dict(orient="records"),
            "top": topK.to_dict(orient="records"),
            "best": best_metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
