# app.py
# -------------------------------
# Minimal Streamlit dashboard for SMA strategy backtest
# Python 3.12 compatible
# -------------------------------

import io
import json
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------
# Page config
# ---------------------------------------------
st.set_page_config(
    page_title="SMA Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📈 双均线策略最小仪表盘（SMA）")

# ---------------------------------------------
# Utils: data loader (supports DIA_3y.csv custom header)
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def load_price_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Load OHLCV CSV.
    Supports two formats:
    1) Standard header: Date, open, high, low, close, volume
    2) Custom multi-row header like DIA_3y.csv:
       Row0: Price,open,high,low,close,volume
       Row1: Ticker, ...
       Row2: Date,,,,,
       Data from Row3
    Returns dataframe indexed by Date (datetime), with columns: open, high, low, close, volume
    """
    # Try read as standard
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        # Heuristics: standard if it contains 'Date' and 'close'
        if {"Date", "close"}.issubset(set(df.columns)):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            cols = ["open", "high", "low", "close", "volume"]
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Date", "close"]).sort_values("Date").set_index("Date")
            # Keep only needed columns (if present)
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[keep]
    except Exception:
        pass

    # Fallback: DIA custom header (no header, data starts from row 3)
    raw = pd.read_csv(io.BytesIO(file_bytes), header=None)
    # Expect first three rows to be header decorations
    if raw.shape[0] < 4:
        raise ValueError("CSV 行数不足，无法解析。")

    cols = ['Date', 'open', 'high', 'low', 'close', 'volume']
    df = raw.iloc[3:].copy()
    df.columns = cols
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "close"]).sort_values("Date").set_index("Date")
    return df[["open", "high", "low", "close", "volume"]]

# ---------------------------------------------
# Indicators & signal
# ---------------------------------------------
def add_sma_signals(df: pd.DataFrame, short_w: int, long_w: int) -> pd.DataFrame:
    d = df.copy()
    d["sma_s"] = d["close"].rolling(short_w, min_periods=short_w).mean()
    d["sma_l"] = d["close"].rolling(long_w,  min_periods=long_w).mean()
    # Signal created on close; position during day t = signal at t-1
    d["signal"] = (d["sma_s"] > d["sma_l"]).astype(int)
    d["pos_day"] = d["signal"].shift(1).fillna(0)
    return d

# ---------------------------------------------
# Backtest core (T+1 execute at next day's open)
# ---------------------------------------------
def backtest_full_switch(
    df: pd.DataFrame, short_w: int, long_w: int,
    initial_cap: float = 10_000.0, tcost_bps: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full switch: pos ∈ {0, 1}. Trade at next-day open.
    tcost_bps: cost per side in basis points (e.g. 10 bps = 0.1%)
    Returns:
      equity_df (index=Date, columns=['strategy_total','strategy_cum','bench_total','bench_cum'])
      trades_df (columns=[date,side,price,shares])
    """
    tc = tcost_bps / 10_000
    d = add_sma_signals(df, short_w, long_w)
    start_idx = max(short_w, long_w)

    cash = initial_cap
    shares = 0
    totals = []
    dates = []
    trades = []

    bench_open = None
    bench_shares = 0
    bench_cash = 0.0
    bench_vals = []

    idx_list = list(d.index)

    for i, dt in enumerate(idx_list):
        row = d.iloc[i]
        if i < start_idx:
            # Pre-tradable window: idle
            totals.append(cash + shares * row["close"])
            dates.append(dt)
            continue

        pos_t = int(row["pos_day"])
        pos_y = int(d.iloc[i-1]["pos_day"]) if i > 0 else 0
        open_px = row["open"]

        # Init benchmark at first tradable day open
        if bench_open is None:
            bench_open = open_px
            bench_shares = int(initial_cap / bench_open)
            bench_cash = initial_cap - bench_shares * bench_open

        # Trade at today's open if position changed
        if pos_t != pos_y:
            if pos_t == 1 and shares == 0:
                max_sh = int(cash / (open_px * (1 + tc)))
                if max_sh > 0:
                    cost = max_sh * open_px * (1 + tc)
                    cash -= cost
                    shares += max_sh
                    trades.append((dt, "BUY", float(open_px), int(max_sh)))
            elif pos_t == 0 and shares > 0:
                revenue = shares * open_px * (1 - tc)
                cash += revenue
                trades.append((dt, "SELL", float(open_px), int(shares)))
                shares = 0

        # Mark to market at close
        close_px = row["close"]
        totals.append(cash + shares * close_px)
        dates.append(dt)

        # Benchmark value (buy&hold)
        bench_vals.append(bench_cash + bench_shares * close_px)

    equity = pd.DataFrame(index=pd.DatetimeIndex(dates))
    equity["strategy_total"] = totals
    equity["strategy_cum"] = equity["strategy_total"] / initial_cap

    # Align benchmark
    bench_series = pd.Series(initial_cap, index=equity.index, dtype=float)
    bench_series.iloc[start_idx:] = bench_vals
    equity["bench_total"] = bench_series.values
    equity["bench_cum"] = equity["bench_total"] / initial_cap

    trades_df = pd.DataFrame(trades, columns=["date", "side", "price", "shares"])
    return equity, trades_df

def backtest_overlay_base(
    df: pd.DataFrame, short_w: int, long_w: int, base_weight: float = 0.3,
    initial_cap: float = 10_000.0, tcost_bps: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Overlay base position: keep base_weight at all times.
    When signal=1 => target 100% ; signal=0 => target base_weight
    Trade at next-day open.
    """
    assert 0.0 <= base_weight <= 1.0
    tc = tcost_bps / 10_000
    d = add_sma_signals(df, short_w, long_w)
    start_idx = max(short_w, long_w)

    cash = initial_cap
    shares = 0
    totals, dates, trades = [], [], []

    idx_list = list(d.index)

    for i, dt in enumerate(idx_list):
        row = d.iloc[i]
        if i < start_idx:
            totals.append(cash + shares * row["close"])
            dates.append(dt)
            continue

        pos_t = int(row["pos_day"])
        w_target = base_weight + (1 - base_weight) * pos_t  # 1.0 or base_weight
        open_px = row["open"]

        # Pre-trade portfolio value at open
        pre_total = cash + shares * open_px
        target_val = w_target * pre_total
        cur_val = shares * open_px

        if target_val > cur_val + 1e-8:
            # Need to buy
            need_val = target_val - cur_val
            buy_sh = int(need_val / (open_px * (1 + tc)))
            if buy_sh > 0:
                cost = buy_sh * open_px * (1 + tc)
                if cost > cash:
                    buy_sh = int(cash / (open_px * (1 + tc)))
                    cost = buy_sh * open_px * (1 + tc)
                if buy_sh > 0:
                    cash -= cost
                    shares += buy_sh
                    trades.append((dt, "BUY", float(open_px), int(buy_sh)))
        elif cur_val > target_val + 1e-8:
            # Need to sell (partial)
            target_sh = int(target_val / open_px)
            sell_sh = shares - target_sh
            if sell_sh > 0:
                revenue = sell_sh * open_px * (1 - tc)
                cash += revenue
                shares -= sell_sh
                trades.append((dt, "SELL", float(open_px), int(sell_sh)))

        close_px = row["close"]
        totals.append(cash + shares * close_px)
        dates.append(dt)

    equity = pd.DataFrame(index=pd.DatetimeIndex(dates))
    equity["strategy_total"] = totals
    equity["strategy_cum"] = equity["strategy_total"] / initial_cap
    return equity, pd.DataFrame(trades, columns=["date", "side", "price", "shares"])

# ---------------------------------------------
# Metrics
# ---------------------------------------------
def compute_metrics(equity: pd.DataFrame, initial_cap: float = 10_000.0) -> dict:
    eq = equity["strategy_cum"]
    total_return = float(eq.iloc[-1] - 1)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan

    rets = equity["strategy_total"].pct_change().dropna()
    vol_ann = float(rets.std(ddof=0) * np.sqrt(252)) if len(rets) > 1 else np.nan
    roll_max = eq.cummax()
    dd = eq / roll_max - 1
    mdd = float(dd.min())
    sharpe = float((rets.mean() / rets.std(ddof=0)) * np.sqrt(252)) if rets.std(ddof=0) > 0 else np.nan

    return {
        "total_return": total_return,
        "CAGR": cagr,
        "vol_annual": vol_ann,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "initial_capital": initial_cap,
    }

# ---------------------------------------------
# Sidebar: Inputs
# ---------------------------------------------
st.sidebar.header("参数设置")

# 数据源：上传或选默认文件
file_source = st.sidebar.radio("数据来源", ["上传 CSV", "读取 ./data/DIA_3y.csv"], horizontal=True)

uploaded = None
if file_source == "上传 CSV":
    uploaded = st.sidebar.file_uploader("上传价格数据 CSV（支持你提供的 DIA_3y 格式或标准列名）", type=["csv"])
else:
    # 读取本地 data/DIA_3y.csv
    try:
        with open("data/DIA_3y.csv", "rb") as f:
            uploaded = io.BytesIO(f.read())
        st.sidebar.info("已读取 ./data/DIA_3y.csv")
    except FileNotFoundError:
        st.sidebar.error("未找到 ./data/DIA_3y.csv，请改为上传 CSV 或把文件放到 data/ 下。")

col1, col2 = st.sidebar.columns(2)
short_w = col1.number_input("短均线 (SMA)", min_value=3, max_value=200, value=10, step=1)
long_w  = col2.number_input("长均线 (SMA)", min_value=10, max_value=400, value=40, step=1)

tcost_bps = st.sidebar.slider("单边交易成本（bps）", min_value=0, max_value=50, value=10, step=1)
initial_cap = st.sidebar.number_input("初始资金（USD）", min_value=1000.0, max_value=1_000_000.0, value=10_000.0, step=1000.0)

use_overlay = st.sidebar.checkbox("启用保留底仓模式", value=True)
base_weight = 0.3
if use_overlay:
    base_weight = st.sidebar.slider("底仓比例", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

run_btn = st.sidebar.button("▶ 运行回测", use_container_width=True)

# ---------------------------------------------
# Main: Run & Display
# ---------------------------------------------
if uploaded is None:
    st.warning("请在左侧上传 CSV 或放置 ./data/DIA_3y.csv 后点击『运行回测』。")
    st.stop()

if run_btn:
    try:
        df = load_price_csv(uploaded.read() if hasattr(uploaded, "read") else uploaded.getvalue())
    except Exception as e:
        st.error(f"读取 CSV 失败：{e}")
        st.stop()

    # 基础检查
    if long_w <= short_w:
        st.error("长均线窗口应大于短均线窗口。")
        st.stop()

    # 回测一：全仓切换
    eq_full, trades_full = backtest_full_switch(df, short_w, long_w, initial_cap, tcost_bps)

    # 回测二：保留底仓（可选）
    eq_overlay, trades_overlay = None, pd.DataFrame()
    if use_overlay:
        eq_overlay, trades_overlay = backtest_overlay_base(df, short_w, long_w, base_weight, initial_cap, tcost_bps)

    # 指标（策略与基准）
    m_full = compute_metrics(eq_full, initial_cap)
    bench_metrics = compute_metrics(
        eq_full.rename(columns={"bench_total": "strategy_total", "bench_cum": "strategy_cum"})[["strategy_total", "strategy_cum"]],
        initial_cap
    )
    m_overlay = compute_metrics(eq_overlay, initial_cap) if use_overlay else None

    # 顶部 KPI
    st.subheader("关键指标")
    kpi_cols = st.columns(3 if use_overlay else 2)
    # 全仓切换
    with kpi_cols[0]:
        st.markdown("**全仓切换（10/40）**")
        st.metric("总收益", f"{m_full['total_return']*100:.2f}%")
        st.metric("年化收益", f"{m_full['CAGR']*100:.2f}%")
        st.metric("最大回撤", f"{m_full['max_drawdown']*100:.2f}%")
        st.caption(f"Sharpe: {m_full['sharpe']:.2f} | 波动: {m_full['vol_annual']*100:.2f}%")

    # 保留底仓
    if use_overlay and m_overlay:
        with kpi_cols[1]:
            st.markdown(f"**保留底仓（底仓={base_weight:.0%}）**")
            st.metric("总收益", f"{m_overlay['total_return']*100:.2f}%")
            st.metric("年化收益", f"{m_overlay['CAGR']*100:.2f}%")
            st.metric("最大回撤", f"{m_overlay['max_drawdown']*100:.2f}%")
            st.caption(f"Sharpe: {m_overlay['sharpe']:.2f} | 波动: {m_overlay['vol_annual']*100:.2f}%")

    # 基准
    with kpi_cols[-1]:
        st.markdown("**买入并持有（DIA）**")
        st.metric("总收益", f"{bench_metrics['total_return']*100:.2f}%")
        st.metric("年化收益", f"{bench_metrics['CAGR']*100:.2f}%")
        st.metric("最大回撤", f"{bench_metrics['max_drawdown']*100:.2f}%")
        st.caption(f"Sharpe: {bench_metrics['sharpe']:.2f} | 波动: {bench_metrics['vol_annual']*100:.2f}%")

    # 曲线对比
    st.subheader("累计收益曲线（起始=1.0）")
    plot_df = pd.DataFrame({
        "全仓切换": eq_full["strategy_cum"],
        "买入并持有": eq_full["bench_cum"]
    })
    if use_overlay and eq_overlay is not None:
        plot_df["保留底仓"] = eq_overlay["strategy_cum"]

    fig = px.line(plot_df, labels={"value": "累计收益", "index": "日期"})
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # 回撤对比
    st.subheader("回撤曲线")
    dd_full = eq_full["strategy_cum"] / eq_full["strategy_cum"].cummax() - 1
    dd_bench = eq_full["bench_cum"] / eq_full["bench_cum"].cummax() - 1
    dd_df = pd.DataFrame({"全仓切换": dd_full, "买入并持有": dd_bench})
    if use_overlay and eq_overlay is not None:
        dd_overlay = eq_overlay["strategy_cum"] / eq_overlay["strategy_cum"].cummax() - 1
        dd_df["保留底仓"] = dd_overlay
    fig2 = px.line(dd_df, labels={"value": "回撤", "index": "日期"})
    st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

    # 交易明细与下载
    st.subheader("交易明细（开盘成交，T+1 执行）")
    tabs = st.tabs(["全仓切换", "保留底仓（如启用）"])
    with tabs[0]:
        st.dataframe(trades_full, use_container_width=True, height=300)
        # 导出
        eq_csv = eq_full.rename_axis("Date").to_csv().encode("utf-8")
        st.download_button("下载全仓切换净值 CSV", eq_csv, file_name="equity_full_switch.csv", use_container_width=True)
        st.download_button("下载全仓切换交易 CSV", trades_full.to_csv(index=False).encode("utf-8"),
                           file_name="trades_full_switch.csv", use_container_width=True)

    with tabs[1]:
        if use_overlay and not trades_overlay.empty:
            st.dataframe(trades_overlay, use_container_width=True, height=300)
            st.download_button("下载保留底仓净值 CSV",
                               eq_overlay.rename_axis("Date").to_csv().encode("utf-8"),
                               file_name="equity_overlay.csv", use_container_width=True)
            st.download_button("下载保留底仓交易 CSV",
                               trades_overlay.to_csv(index=False).encode("utf-8"),
                               file_name="trades_overlay.csv", use_container_width=True)
        else:
            st.info("未启用或暂无交易。")

    # 原始参数快照
    st.caption(
        f"区间：{m_full['start']} ~ {m_full['end']} ｜ 初始资金：${initial_cap:,.0f} ｜ 成本（单边）：{tcost_bps} bps ｜ 执行：T+1开盘"
    )

else:
    st.info("设置好参数后，点击左侧 **运行回测**。")
