import numpy as np
import polars as pl
import pandas as pd
def resample_to_daily(df: pl.DataFrame) :
    # 確保 timestamp 欄位是日期時間格式
    print(len(df))
    
    daily_df = (
    df.with_columns(
        pl.col("ts")
          .str.slice(0, 19)
          .str.to_datetime("%Y-%m-%dT%H:%M:%S")
          .alias("ts")
    )
    .sort("ts")
    .group_by_dynamic("ts", every="1d")
    .agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
    ])
    .rename({"ts": "date"})) 
    print(len(daily_df))
    print(daily_df["date"].min())
    print(daily_df["date"].max())
    print(daily_df["date"].n_unique())
    return daily_df


def turtle_trading_system(df: pl.DataFrame, 
    enter_term: int = 20, 
    leave_term: int = 10, 
    vertical_barrier: int = 5, 
    position: float = 1.0, 
    fee: float = 0.003) :
    
    # 1. 計算過去 term 的極值 (不包含當天，所以使用 shift(1))
    df = df.with_columns([
        pl.col("high").rolling_max(window_size=enter_term).shift(1).alias("last_enter_term_max"),
        pl.col("low").rolling_min(window_size=leave_term).shift(1).alias("last_leave_term_min")
    ])

    # 2. 初始化交易欄位
    records = df.to_dicts()
    in_position = False
    entry_day_count = 0
    
    for i in range(len(records)):
        row = records[i]
        buy_act = 0
        sell_act = 0
        profit = 0.0
        
        # 排除前期的空值 (Rolling Window 導致的 null)
        if row["last_enter_term_max"] is None or row["last_leave_term_min"] is None:
            row.update({"buy_action": 0, "sell_action": 0, "profit": 0.0})
            continue

        # --- 交易邏輯 ---
        if not in_position:
            # 入場檢查：當天最高 > 過去 N 日最大
            if row["high"] > row["last_enter_term_max"]:
                buy_act = 1
                in_position = True
                entry_day_count = 0 # 重設持有天數
                # 買入支出：價格 * 數量 * (1 + 手續費)
                profit = -(row["close"] * position * (1 + fee))
        else:
            entry_day_count += 1
            # 出場檢查：當天最低 < 過去 M 日最小 OR 達到垂直屏障
            if row["low"] < row["last_leave_term_min"] or entry_day_count >= vertical_barrier:
                sell_act = 1
                in_position = False
                # 賣出收入：價格 * 數量 * (1 - 手續費)
                profit = (row["close"] * position * (1 - fee))
        
        row.update({
            "buy_action": buy_act,
            "sell_action": sell_act,
            "profit": profit
        })

    # 3. 轉回 Polars 並計算累計欄位
    result_df = pl.from_dicts(records)
    
    result_df = result_df.with_columns([
        pl.col("buy_action").cum_sum().alias("cumulative_buy_position"),
        pl.col("sell_action").cum_sum().alias("cumulative_sell_position")
    ])

    return result_df