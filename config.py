# dma_strategy/config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    symbol: str = "AAPL"
    period: str = "3y"     # 或 start/end
    use_adjusted: bool = True
    data_dir: str = "data"

@dataclass
class SignalConfig:
    short_windows: List[int] = (5, 10, 15)
    long_windows: List[int]  = (30, 50, 60)
    cooldown_bars: int = 5   # 你已验证更优

@dataclass
class ExecConfig:
    exec_price: str = "open"     # 次日开盘
    exec_delay_bars: int = 1

@dataclass
class TradeConfig:
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001
    allow_fractional: bool = True
    buy_fractions: List[float] = (0.25, 0.5, 0.75)
    sell_fractions: List[float] = (0.25, 0.5, 0.75)
    exclusives: List[bool] = (False, True)  # False=部分仓位；True=全进全出

@dataclass
class OutputConfig:
    results_dir: str = "results"
    last_n_days: Optional[int] = 500
    save_individual: bool = False
    save_comparison: bool = True
    top_k: int = 5
    ranking_metric: str = "annualized_return"