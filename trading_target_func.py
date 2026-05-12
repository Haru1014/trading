import numpy as np
import polars as pl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def resample_to_daily(df: pl.DataFrame,inst_id: str = "BTC-USDT-SWAP") :
    # 確保 timestamp 欄位是日期時間格式
    print(len(df))
    clean_df = df.filter(pl.col("inst_id") == inst_id)
    daily_df = (
    clean_df.with_columns(
        pl.col("ts")
          .str.slice(0, 19)
          .str.to_datetime("%Y-%m-%dT%H:%M:%S")
          .alias("ts")
    )
    .sort("ts")
    .group_by_dynamic("ts", every="1d")
    .agg([
        pl.col("open").sort_by("ts").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").sort_by("ts").last(),
    ])
    .rename({"ts": "date"})) 
    print(daily_df.sort("date").head(1))
    print(len(daily_df))
    print(daily_df["date"].min())
    print(daily_df["date"].max())
    print(daily_df["date"].n_unique())
    return daily_df


def turtle_trading_system(df: pl.DataFrame, 
    enter_term: int = 20, 
    leave_term: int = 10, 
    vertical_barrier: int = 30, 
    position: float = 1.0, 
    fee: float = 0.003) :
    
    # 1. 計算過去 term 的極值 (不包含當天，所以使用 shift(1))
    df = df.with_columns([
        pl.col("high").rolling_max(window_size=enter_term).shift(1).alias("last_enter_term_max"),
        pl.col("low").rolling_min(window_size=leave_term).shift(1).alias("last_leave_term_min")
    ])
    # 顯示前 25 筆資料 (因為 enter_term=20，前 20 筆應該會是 null)
    print(df.select([
        "date", "high", "open", "close", "low", "last_enter_term_max", "last_leave_term_min"
    ]).head(25))
    # 2. 初始化交易欄位
    records = df.to_dicts()
    in_position = False
    entry_day_count = 0
    instant_cumulative_profit = 0.0
    
    for i in range(len(records)):
        row = records[i]
        buy_act = 0
        sell_act = 0
        profit = 0.0
        vertical_barrier_act = 0
        # 排除前期的空值 (Rolling Window 導致的 null)
        if row["last_enter_term_max"] is None or row["last_leave_term_min"] is None:
            row.update({"buy_action": 0, "sell_action": 0, "profit": 0.0})
            continue
        '''
        空手 → 只看入場訊號
        持倉 → 只看出場訊號
        '''
        # --- 交易邏輯 ---
        if not in_position:
            # 入場檢查：當天最高 > 過去 N 日最大
            if row["high"] > row["last_enter_term_max"]:
                buy_act = 1
                in_position = True
                entry_day_count = 0 # 重設持有天數
                # 買入支出：價格 * 數量 * (1 + 手續費)
                profit = -(row["close"] * position * (1 + fee))
                instant_cumulative_profit += profit
                whole_asset = instant_cumulative_profit + row["close"] * position
                
        else:
            entry_day_count += 1
            # 出場檢查：當天最低 < 過去 M 日最小 OR 達到垂直屏障
            if row["low"] < row["last_leave_term_min"] :
                sell_act = 1
                in_position = False
                # 賣出收入：價格 * 數量 * (1 - 手續費)
                profit = (row["close"] * position * (1 - fee))
                instant_cumulative_profit += profit
            elif entry_day_count >= vertical_barrier:
                sell_act = 1
                vertical_barrier_act = 1  # 標記是垂直屏障平倉
                in_position = False
                profit = (row["close"] * position * (1 - fee))
                instant_cumulative_profit += profit
            whole_asset = instant_cumulative_profit
        row.update({
            "buy_action": buy_act,
            "sell_action": sell_act,
            "vertical_barrier_exit": vertical_barrier_act,
            "profit": profit,
            "whole_asset": whole_asset
        })
        print(f"date: {row['date']}  instant_cumulative_profit: {instant_cumulative_profit} vertical_barrier_act: {vertical_barrier_act} whole_asset: {whole_asset}")
    # 3. 轉回 Polars 並計算累計欄位                                
    result_df = pl.from_dicts(records)
    
    result_df = result_df.with_columns([
        pl.col("buy_action").cum_sum().alias("cumulative_buy_position"),
        pl.col("sell_action").cum_sum().alias("cumulative_sell_position"),
        pl.col("profit").cum_sum().alias("cumulative_profit"),
        pl.col("whole_asset").cum_sum().alias("cumulative_whole_asset")
    ])

    return result_df

def plot_turtle_trading(result_df: pl.DataFrame):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=["K線圖", "累計損益"]
    )

    # --- K線圖 ---
    fig.add_trace(go.Candlestick(
        x=result_df["date"],
        open=result_df["open"],
        high=result_df["high"],
        low=result_df["low"],
        close=result_df["close"],
        name="OHLC"
    ), row=1, col=1)

    # --- 入場線 ---
    fig.add_trace(go.Scatter(
        x=result_df["date"],
        y=result_df["last_enter_term_max"],
        mode="lines",
        line=dict(color="green", width=1, dash="dash"),
        name="入場線"
    ), row=1, col=1)

    # --- 出場線 ---
    fig.add_trace(go.Scatter(
        x=result_df["date"],
        y=result_df["last_leave_term_min"],
        mode="lines",
        line=dict(color="red", width=1, dash="dash"),
        name="出場線"
    ), row=1, col=1)

    # --- 買進標記 ---
    buy_df = result_df.filter(pl.col("buy_action") == 1)
    fig.add_trace(go.Scatter(
        x=buy_df["date"],
        y=buy_df["low"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="green"),
        name="買進"
    ), row=1, col=1)

    # --- 正常出場標記 ---
    sell_df = result_df.filter(
        (pl.col("sell_action") == 1) & (pl.col("vertical_barrier_exit") == 0)
    )
    fig.add_trace(go.Scatter(
        x=sell_df["date"],
        y=sell_df["high"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="red"),
        name="出場"
    ), row=1, col=1)

    # --- 垂直屏障出場標記 ---
    vb_df = result_df.filter(pl.col("vertical_barrier_exit") == 1)
    fig.add_trace(go.Scatter(
        x=vb_df["date"],
        y=vb_df["high"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="orange"),
        name="垂直屏障出場"
    ), row=1, col=1)

    # --- 累計損益曲線 ---
    fig.add_trace(go.Scatter(
        x=result_df["date"],
        y=result_df["profit"].cum_sum(),
        mode="lines",
        line=dict(color="blue"),
        name="累計損益"
    ), row=2, col=1)
    # --- 累計資產曲線 ---
    fig.add_trace(go.Scatter(
        x=result_df["date"],
        y=result_df["whole_asset"].cum_sum(),
        mode="lines",
        line=dict(color="green"),
        name="累計資產"
    ), row=2, col=1)
    fig.update_layout(
        title="海龜交易系統",
        xaxis_rangeslider_visible=False,
        height=800
    )

    fig.show()