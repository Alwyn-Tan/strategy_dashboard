import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
from data import fetch_stock_data
from analysis import analyze_grid

SYMBOL = "AAPL"
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001
COOLDOWN_BARS = 0
TEST_SHORT_WINDOWS = [5, 10, 15, 20]
TEST_LONG_WINDOWS  = [30, 50, 60, 80, 100]
BUY_GRID  = [1.0]
SELL_GRID = [0.0]
EXCLUSIVES = [False]
LAST_N_DAYS = 500

stock_data=fetch_stock_data(SYMBOL)