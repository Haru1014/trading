from pathlib import Path
import trading_target_func as ttf
import polars as pl
csv_path = Path(r"C:\trading_data\btc_1m.csv")
df = pl.read_csv(csv_path)
print(df.head())
daily_df = ttf.resample_to_daily(df,"BTC-USDT-SWAP")
result_df = ttf.turtle_trading_system(daily_df, 20, 55, 10, 30, 1.0, 0.003, 'system1')
result_df_2 = result_df.drop("last_enter_term_max")
result_df_2.write_csv("result_df_2.csv")

print(result_df.tail())
#ttf.plot_turtle_trading(result_df)
trade_df=ttf.get_trade_signals(result_df)   
trade_df.write_csv("trade_signals.csv")
ttf.evaluate_all(trade_df, daily_df, 0.003)
print("Done")
sweep_df =  ttf.sweep_enter_term(daily_df, mode='system1')