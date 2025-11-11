from typing import List, Optional

from pydantic import BaseModel, Field


class FetchRequest(BaseModel):
    symbol: str = Field("AAPL", description="股票代码，例如 AAPL/DIA")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = Field("3y", description="若未提供日期范围，使用 period，例如 1y/3y/5y")
    save_local: bool = True
    data_dir: Optional[str] = "data"
    use_adjusted: bool = True
    return_data: bool = False
    max_rows: int = 1000

class SingleBacktestRequest(BaseModel):
    symbol: str = "AAPL"
    period: str = "3y"
    use_adjusted: bool = True
    cooldown_bars: int = 0
    short_window: int = 10
    long_window: int = 50
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001
    buy_fraction: float = 1.0
    sell_fraction: float = 1.0
    allow_fractional: bool = True
    exec_price: str = "open"  # open or close
    exec_delay: int = 1
    exclusive: bool = False
    last_n_days: Optional[int] = 500
    save_results: bool = False
    results_root: str = "results"

class GridBacktestRequest(BaseModel):
    symbol: str = "AAPL"
    period: str = "3y"
    use_adjusted: bool = True
    short_windows: List[int]
    long_windows: List[int]
    buy_fractions: List[float]
    sell_fractions: List[float]
    exclusives: List[bool]
    cooldown_bars: int = 0
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001
    exec_price: str = "open"
    exec_delay: int = 1
    allow_fractional: bool = True
    last_n_days: Optional[int] = 500
    save_comparison: bool = True
    save_individual: bool = False
    top_k: int = 5
    ranking_metric: str = "annualized_return"
    results_root: str = "results"

class PlotRequest(BaseModel):
    symbol: str
    short_window: int
    long_window: int
    buy_fraction: float = 1.0
    sell_fraction: float = 1.0
    exclusive: bool = False
    cooldown_bars: int = 0
    period: str = "3y"
    last_n_days: Optional[int] = 500
    exec_price: str = "open"
    exec_delay: int = 1
    allow_fractional: bool = True
    results_root: str = "results"
