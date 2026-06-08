from pathlib import Path
from turtle import mode
import trading_target_func as ttf
import polars as pl
'''
#=========================以下是找出高原的部分=========================
csv_path = Path(r"C:\trading_data\參數分析結果\index_parameter_result_sys2_60.csv")
result_df_all = pl.read_csv(csv_path)
plateaus_list=ttf.find_plateau(result_df_all,free_params=["enter_term_sys2", "leave_term"],
                min_asset = 0,
                min_expectancy= 0,
                min_plr= 1.0,
                mode='system2',
                plot = True)
'''
#========================以下跑策略的部分=========================
csv_path = Path(r"C:\trading_data\btc_1m.csv")
df = pl.read_csv(csv_path)
print(df.head())
daily_df = ttf.resample_to_daily(df,"BTC-USDT-SWAP")
#trade_df_budget= ttf.turtle_trading_system_gold_standard(daily_df,20,55,10,30,10000,0.00000001,0.003,2000,True, 'system1')
#trade_df_budget.write_csv("trade_df_budget20_10.csv")
'''
trade_df_budget=ttf.turtle_trading_system_full(daily_df,20,55,10,20,4,6,100000.0,0.01,min_position=0.00000001,fee= 0.003,atr_period = 20)
trade_df_budget.write_csv("trade_df_full.csv")
'''
#ttf.plot_turtle_trading(result_df)

sweep_df = ttf.sweep_params_interactive_full(daily_df,100000.0,0.01,0.001,0.003,20, 4,6,100000.0)
print("Done")
'''
sweep_df =  ttf.sweep_params_interactive(daily_df,30,1.0, 0.003, 'system1')
print("Done")
'''