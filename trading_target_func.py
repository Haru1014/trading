import numpy as np
import polars as pl
import pandas as pd
def resample_to_daily(df: pl.DataFrame) :
    # 確保 timestamp 欄位是日期時間格式
    # 如果已經是 datetime 則可省略此步驟
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
    daily_df = (
        df.sort("timestamp") # 確保時間是由小到大排序
        .group_by_dynamic(
            "timestamp",
            every="1d",      # 以「一天」為單位進行分組
        )
        .agg([
            pl.col("open").first(),   # 取當日第一筆
            pl.col("high").max(),     # 取當日最高
            pl.col("low").min(),      # 取當日最低
            pl.col("close").last(),   # 取當日最後一筆
        ])
        .rename({"timestamp": "date"}) # 依照你的需求更名
    )

    return daily_df