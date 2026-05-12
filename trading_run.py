from pathlib import Path
import trading_target_func as ttf
import polars as pl
csv_path = Path(r"C:\trading_data\btc_1m.csv")
df = pl.read_csv(csv_path)
print(df.head())
daily_df = ttf.resample_to_daily(df,"BTC-USDT-SWAP")
result_df = ttf.turtle_trading_system(daily_df)
print(result_df.tail())
ttf.plot_turtle_trading(result_df)