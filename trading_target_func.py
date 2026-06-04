from random import seed
import numpy as np
import polars as pl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from scipy import stats
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
def calc_atr(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    df = df.sort("date")
    
    prev_close = pl.col("close").shift(1)
    
    df = df.with_columns([
        # True Range = max(high-low, |high-prev_close|, |prev_close-low|)
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (prev_close - pl.col("low")).abs()
        ).alias("true_range")
    ])
    
    df = df.with_columns([
        pl.col("true_range")
          .rolling_mean(window_size=period)
          .alias("ATR")
    ])
    
    return df
def gt(a, b):
    if isinstance(b, list):
        return any(a > x for x in b)
    return a > b
def gt_all(a, b):
    if isinstance(b, list):
        return all(a > x for x in b)
    return a > b
def is_none_val(val):
    if isinstance(val, list):
        return any(x is None for x in val)
    return val is None
def turtle_trading_system(df: pl.DataFrame, enter_term_sys1: int = 20,enter_term_sys2: int = 55, leave_term: int = 10, vertical_barrier: int = 30,position: float = 1.0, fee: float = 0.003,mode='system1') :
    
    # 1. 計算過去 term 的極值 (不包含當天，所以使用 shift(1))
    df = df.with_columns([
        pl.col("high").rolling_max(window_size=enter_term_sys1).shift(1).alias("last_enter_term_max_sys1"),
        pl.col("high").rolling_max(window_size=enter_term_sys2).shift(1).alias("last_enter_term_max_sys2"),
        pl.col("low").rolling_min(window_size=leave_term).shift(1).alias("last_leave_term_min")
    ])
    
    # 2. 初始化交易欄位
    records = df.to_dicts()
    in_position = False
    entry_day_count = 0
    instant_cumulative_profit = 0.0
    position_value= 0.0
    for i in range(len(records)):
        row = records[i]
        buy_act = 0
        sell_act = 0
        profit = 0.0
        vertical_barrier_act = 0
        # 排除前期的空值 (Rolling Window 導致的 null)
        if mode == 'system1':
            row["last_enter_term_max"] = row["last_enter_term_max_sys1"]
        elif mode == 'system2':
            row["last_enter_term_max"] = row["last_enter_term_max_sys2"]    
        else:
            row["last_enter_term_max"] = [row["last_enter_term_max_sys1"],row["last_enter_term_max_sys2"]]
            
        if is_none_val(row["last_enter_term_max"]) or is_none_val(row["last_leave_term_min"]):
            row.update({"buy_action": 0, "sell_action": 0, "profit": 0.0})
            continue
        '''
        空手 → 只看入場訊號
        持倉 → 只看出場訊號
        '''
        # --- 交易邏輯 ---
        if not in_position:
            # 入場檢查：當天最高 > 過去 N 日最大
            if gt(row["high"], row["last_enter_term_max"]):
                buy_act = 1
                in_position = True
                entry_day_count = 0 # 重設持有天數
                # 買入支出：價格 * 數量 * (1 + 手續費)
                profit = -(row["close"] * position * (1 + fee))
                instant_cumulative_profit += profit
                position_value = row["close"] * position
            else:
                profit = 0.0    
        else:
            entry_day_count += 1
            # 出場檢查：當天最低 < 過去 M 日最小 OR 達到垂直屏障
            if row["low"] < row["last_leave_term_min"]:
                sell_act = 1
                in_position = False
                # 賣出收入：價格 * 數量 * (1 - 手續費)
                profit = (row["close"] * position * (1 - fee))
                instant_cumulative_profit += profit
                position_value = 0
            elif entry_day_count >= vertical_barrier:
                sell_act = 1
                vertical_barrier_act = 1  # 標記是垂直屏障平倉
                in_position = False
                profit = (row["close"] * position * (1 - fee))
                instant_cumulative_profit += profit
                position_value = 0
            else:
                profit = 0.0
        if in_position:
            position_value = row["close"] * position 
        else:
            position_value = 0 
        row.update({
            "buy_action": buy_act,
            "sell_action": sell_act,
            "vertical_barrier_exit": vertical_barrier_act,
            "profit": profit,
            "position_value": position_value
        })
        # print(f"date: {row['date']}  instant_cumulative_profit: {instant_cumulative_profit} buy: {buy_act} sell: {sell_act} vertical_exit: {vertical_barrier_act} position: {position_value}")
    # 3. 轉回 Polars 並計算累計欄位                                
    result_df = pl.from_dicts(records)
    
    result_df = result_df.with_columns([
        pl.col("buy_action").cum_sum().alias("cumulative_buy_position"),
        pl.col("sell_action").cum_sum().alias("cumulative_sell_position"),
        pl.col("profit").cum_sum().alias("cumulative_profit")
    ])
    #print(result_df.columns)
    return result_df
def turtle_trading_system_with_budget(
        df: pl.DataFrame,
        enter_term_sys1: int = 20,
        enter_term_sys2: int = 55,
        leave_term: int = 10,
        vertical_barrier: int = 30,
        own_capital: float = 100000.0,   # 自有資金
        invest_pct: float = 0.5,          # 願意投入比例
        min_position: float = 0.00000001,      # 最小單位
        fee: float = 0.003,
        mode: str = 'system1'):

    # 投資資金
    invest_capital = own_capital * invest_pct

    # 1. 計算極值
    df = df.with_columns([
        pl.col("high").rolling_max(window_size=enter_term_sys1).shift(1).alias("last_enter_term_max_sys1"),
        pl.col("high").rolling_max(window_size=enter_term_sys2).shift(1).alias("last_enter_term_max_sys2"),
        pl.col("low").rolling_min(window_size=leave_term).shift(1).alias("last_leave_term_min")
    ])

    records = df.to_dicts()
    in_position   = False
    entry_day_count = 0
    instant_cumulative_profit = 0.0
    position_value = 0.0
    cash           = invest_capital   # 手頭現金，初始等於投資資金
    current_position = 0.0            # 目前持倉數量

    for i in range(len(records)):
        row = records[i]
        buy_act  = 0
        sell_act = 0
        profit   = 0.0
        vertical_barrier_act = 0

        if mode == 'system1':
            row["last_enter_term_max"] = row["last_enter_term_max_sys1"]
        elif mode == 'system2':
            row["last_enter_term_max"] = row["last_enter_term_max_sys2"]
        else:
            row["last_enter_term_max"] = [row["last_enter_term_max_sys1"], row["last_enter_term_max_sys2"]]

        if is_none_val(row["last_enter_term_max"]) or is_none_val(row["last_leave_term_min"]):
            row.update({"buy_action": 0, "sell_action": 0, "profit": 0.0,
                        "position_value": 0.0, "cash": cash})
            continue

        if not in_position:
            if gt(row["high"], row["last_enter_term_max"]):
                # 計算可買數量（含手續費後現金不為負）
                max_affordable = cash / (row["close"] * (1 + fee))
                # 無條件捨去到 min_position 的整數倍
                position = math.floor(max_affordable / min_position) * min_position

                if position >= min_position:
                    buy_act  = 1
                    in_position = True
                    entry_day_count = 0
                    current_position = position
                    cost   = row["close"] * position * (1 + fee)
                    profit = -cost
                    cash  -= cost
                    instant_cumulative_profit += profit
                    position_value = row["close"] * position
                # 買不到最小單位 → 跳過，不進場
                else:
                    profit = 0.0
            else:
                profit = 0.0

        else:
            entry_day_count += 1
            exit_triggered = False

            if row["low"] < row["last_leave_term_min"]:
                sell_act = 1
                exit_triggered = True
            elif entry_day_count >= vertical_barrier:
                sell_act = 1
                vertical_barrier_act = 1
                exit_triggered = True

            if exit_triggered:
                in_position = False
                revenue = row["close"] * current_position * (1 - fee)
                profit  = revenue
                cash   += revenue
                instant_cumulative_profit += profit
                position_value   = 0.0
                current_position = 0.0
            else:
                profit = 0.0

        if in_position:
            position_value = row["close"] * current_position
        else:
            position_value = 0.0

        row.update({
            "buy_action":            buy_act,
            "sell_action":           sell_act,
            "vertical_barrier_exit": vertical_barrier_act,
            "profit":                profit,
            "position_value":        position_value,
            "cash":                  cash,
            "current_position":      current_position,
        })

    result_df = pl.from_dicts(records)
    result_df = result_df.with_columns([
        pl.col("buy_action").cum_sum().alias("cumulative_buy_position"),
        pl.col("sell_action").cum_sum().alias("cumulative_sell_position"),
        pl.col("profit").cum_sum().alias("cumulative_profit")
    ])
    return result_df

def turtle_trading_system_gold_standard(
        df: pl.DataFrame,
        enter_term_sys1: int = 20,
        enter_term_sys2: int = 55,
        leave_term: int = 10,
        vertical_barrier: int = 30,
        initial_fund: float = 10000.0,
        min_position: float = 0.00000001,
        fee: float = 0.003,
        one_position_cash: float = 2000.0,
        budget_line: bool = True,
        mode: str = 'system1') -> pl.DataFrame:


    # 1. 計算極值
    df = df.with_columns([
        pl.col("high").rolling_max(window_size=enter_term_sys1).shift(1).alias("last_enter_term_max_sys1"),
        pl.col("high").rolling_max(window_size=enter_term_sys2).shift(1).alias("last_enter_term_max_sys2"),
        pl.col("low").rolling_min(window_size=leave_term).shift(1).alias("last_leave_term_min")
    ])

    records        = df.to_dicts()
    in_position    = False
    entry_day_count = 0
    instant_cumulative_profit = 0.0
    position_value   = 0.0
    current_position = 0.0
    cash = initial_fund  # 手頭現金，budget_line=False 時僅追蹤不限制

    for i in range(len(records)):
        row = records[i]
        buy_act  = 0
        sell_act = 0
        profit   = 0.0
        vertical_barrier_act = 0

        if mode == 'system1':
            row["last_enter_term_max"] = row["last_enter_term_max_sys1"]
        elif mode == 'system2':
            row["last_enter_term_max"] = row["last_enter_term_max_sys2"]
        else:
            row["last_enter_term_max"] = [row["last_enter_term_max_sys1"], row["last_enter_term_max_sys2"]]

        if is_none_val(row["last_enter_term_max"]) or is_none_val(row["last_leave_term_min"]):
            row.update({
                "buy_action": 0, "sell_action": 0,
                "vertical_barrier_exit": 0, "profit": 0.0,
                "position_value": 0.0, "cash": cash,
                "current_position": 0.0, "position": 0.0
            })
            continue

        if not in_position:
            if gt(row["high"], row["last_enter_term_max"]):

                # 換算本次買入數量（以 one_position_cash 為基準）
                max_affordable  = one_position_cash / (row["close"] * (1 + fee))
                position        = math.floor(max_affordable / min_position) * min_position
                actual_cost     = row["close"] * position * (1 + fee)

                # 判斷是否可以買入
                can_buy = position >= min_position
                if budget_line:
                    can_buy = can_buy and (cash >= actual_cost)

                if can_buy:
                    buy_act          = 1
                    in_position      = True
                    entry_day_count  = 0
                    current_position = position
                    profit           = -actual_cost
                    cash            += profit
                    instant_cumulative_profit += profit
                    position_value   = row["close"] * position
                else:
                    profit = 0.0
            else:
                profit = 0.0

        else:
            entry_day_count += 1
            exit_triggered  = False

            if row["low"] < row["last_leave_term_min"]:
                sell_act       = 1
                exit_triggered = True
            elif entry_day_count >= vertical_barrier:
                sell_act             = 1
                vertical_barrier_act = 1
                exit_triggered       = True

            if exit_triggered:
                in_position      = False
                revenue          = row["close"] * current_position * (1 - fee)
                profit           = revenue
                cash            += revenue
                instant_cumulative_profit += profit
                position_value   = 0.0
                current_position = 0.0
            else:
                profit = 0.0

        if in_position:
            position_value = row["close"] * current_position
        else:
            position_value = 0.0

        row.update({
            "buy_action":            buy_act,
            "sell_action":           sell_act,
            "vertical_barrier_exit": vertical_barrier_act,
            "profit":                profit,
            "position_value":        position_value,
            "cash":                  cash,
            "current_position":      current_position,
        })

        # cash 檢查
        if cash < 0:
            print(f"enter_term_sys1={enter_term_sys1}, enter_term_sys2={enter_term_sys2}, "
                  f"leave_term={leave_term}, vertical_barrier={vertical_barrier}, error")

    result_df = pl.from_dicts(records)
    result_df = result_df.with_columns([
        pl.col("buy_action").cum_sum().alias("cumulative_buy_position"),
        pl.col("sell_action").cum_sum().alias("cumulative_sell_position"),
        pl.col("profit").cum_sum().alias("cumulative_profit")
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
        # 單條入場線
    fig.add_trace(go.Scatter(
        x=result_df["date"],
        y=result_df["last_enter_term_max_sys1"],
        mode="lines",
        line=dict(color="green", width=1, dash="dash"),
        name="短天期入場線"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=result_df["date"],
        y=result_df["last_enter_term_max_sys2"],
        mode="lines",
        line=dict(color="gray", width=1, dash="dash"),
        name="長天期入場線"
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
        y=buy_df["close"],
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
        y=sell_df["close"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="red"),
        name="出場"
    ), row=1, col=1)

    # --- 垂直屏障出場標記 ---
    vb_df = result_df.filter(pl.col("vertical_barrier_exit") == 1)
    fig.add_trace(go.Scatter(
        x=vb_df["date"],
        y=vb_df["close"],
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
        y=result_df["position_value"]+result_df["profit"].cum_sum(),
        mode="lines",
        line=dict(color="green"),
        name="累計資產"
    ), row=2, col=1)
    fig.update_layout(
        title="海龜交易系統",
        xaxis_rangeslider_visible=False,
        height=800
    )
    fig.write_html("turtle_trading.html")   
    fig.show()

def get_trade_signals(result_df: pl.DataFrame, symbol: str = "BTC-USDT-SWAP"):
    # 1. 篩選有動作的日期並標記資訊
    trade_df = result_df.filter(
        (pl.col("buy_action") == 1) | (pl.col("sell_action") == 1)
    ).with_columns([
        pl.lit(symbol).alias("symbol"),
        pl.when(pl.col("buy_action") == 1)
          .then(pl.lit("BUY"))
          .otherwise(pl.lit("SELL"))
          .alias("side"),
        # 假設你的 result_df 裡有 vertical_barrier_exit 欄位
        pl.when(pl.col("vertical_barrier_exit") == 1)
          .then(pl.lit("Vertical Barrier"))
          .when(pl.col("sell_action") == 1)
          .then(pl.lit("Signal"))
          .otherwise(pl.lit("-"))
          .alias("exit_reason"),
        (pl.col("close") * 0.003).alias("fee"),
    ])

    # 2. 計算累計盈虧與權益價值
    trade_df = trade_df.with_columns([
        # 累計 profit 欄位
        pl.col("profit").cum_sum().alias("cumulative_profit"),
    ]).with_columns([
        # 權益曲線 = 累計盈虧 + 目前持倉價值
        (pl.col("cumulative_profit") + pl.col("position_value")).alias("position_value_plus_cumulative_profit")
    ])

    # 3. 整理最後欄位並重新命名
    return trade_df.select([
        "date",
        "symbol",
        "side",
        "close",
        "fee",
        "profit",
        "position_value",
        "cumulative_profit",
        "position_value_plus_cumulative_profit",
        "exit_reason",
    ]).rename({"close": "price"})

def sweep_enter_term(daily_df: pl.DataFrame, 
                     leave_term: int = 10, 
                     vertical_barrier: int = 30,
                     position_size: float = 1.0, 
                     fee_rate: float = 0.003,
                     mode: str = 'system1'):
   
    import plotly.graph_objects as go

    enter_terms = list(range(5, 31))
    total_profits = []
    final_whole_assets = []

    for enter_term in enter_terms:
        result_df = turtle_trading_system(
            daily_df, enter_term, 55, leave_term, vertical_barrier, position_size, fee_rate, mode
        )
        total_profit = result_df["profit"].sum()
        total_profits.append(total_profit)

        # 計算 whole_asset（累積profit + position_value，null當0）
        cum_profits = result_df["profit"].cum_sum()
        position_values = result_df["position_value"].fill_null(0)
        whole_asset = cum_profits + position_values
        final_whole_assets.append(whole_asset[-1])

        print(f"enter_term={enter_term}, total_profit={total_profit:.4f}, final_whole_asset={whole_asset[-1]:.4f}")

    # 找最佳（以最後 whole_asset 最大為準）
    best_idx = int(np.argmax(final_whole_assets))
    best_term = enter_terms[best_idx]
    best_whole_asset = final_whole_assets[best_idx]

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=enter_terms,
        y=final_whole_assets,
        mode="lines+markers",
        line=dict(color="blue"),
        marker=dict(size=6),
        name="最終 Whole Asset"
    ))
    fig.add_trace(go.Scatter(
        x=[best_term],
        y=[best_whole_asset],
        mode="markers",
        marker=dict(size=12, color="red", symbol="star"),
        name=f"最佳 enter_term={best_term}"
    ))
    fig.update_layout(
        title=f"Enter Term Sweep（mode={mode}）",
        xaxis_title="enter_term (sys1)",
        yaxis_title="最終 Whole Asset",
        xaxis=dict(tickmode="linear", tick0=5, dtick=1),
        height=500
    )
    fig.write_html("sweep_enter_term.html")
    fig.show()

    return pl.DataFrame({
        "enter_term": enter_terms,
        "total_profit": total_profits,
        "final_whole_asset": final_whole_assets,
    })
def save_sweep_result(results: list,
                      rules: dict,
                      total_combinations: int,
                      mode: str,
                      cv_sys1: float,
                      cv_sys2: float,
                      cv_leave: float) -> str:
    from datetime import datetime

    best_asset   = max(results, key=lambda x: x["final_whole_asset"])
    best_winrate = max(results, key=lambda x: x["win_rate"])
    best_mdd     = max(results, key=lambda x: x["mdd"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = f"sweep_result_{timestamp}.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("參數規則\n")
        f.write("=" * 50 + "\n")
        for k, v in rules.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n測試規模: {total_combinations} 種合法組合\n")
        f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"mode: {mode}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("各指標最佳組合\n")
        f.write("=" * 50 + "\n")

        f.write("\n[最佳 Final Whole Asset]\n")
        f.write(f"  sys1={best_asset['enter_term_sys1']}, sys2={best_asset['enter_term_sys2']}, "
                f"leave_term={best_asset['leave_term']}\n")
        f.write(f"  final_whole_asset={best_asset['final_whole_asset']:.4f}, "
                f"win_rate={best_asset['win_rate']:.4f}, mdd={best_asset['mdd']:.4f}\n")

        f.write("\n[最佳 Win Rate]\n")
        f.write(f"  sys1={best_winrate['enter_term_sys1']}, sys2={best_winrate['enter_term_sys2']}, "
                f"leave_term={best_winrate['leave_term']}\n")
        f.write(f"  final_whole_asset={best_winrate['final_whole_asset']:.4f}, "
                f"win_rate={best_winrate['win_rate']:.4f}, mdd={best_winrate['mdd']:.4f}\n")

        f.write("\n[最佳 MDD（最小回撤）]\n")
        f.write(f"  sys1={best_mdd['enter_term_sys1']}, sys2={best_mdd['enter_term_sys2']}, "
                f"leave_term={best_mdd['leave_term']}\n")
        f.write(f"  final_whole_asset={best_mdd['final_whole_asset']:.4f}, "
                f"win_rate={best_mdd['win_rate']:.4f}, mdd={best_mdd['mdd']:.4f}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("參數敏感度（CV，越小越穩健）\n")
        f.write("=" * 50 + "\n")
        f.write(f"  enter_term_sys1 敏感度: {cv_sys1:.4f}\n")
        f.write(f"  enter_term_sys2 敏感度: {cv_sys2:.4f}\n")
        f.write(f"  leave_term      敏感度: {cv_leave:.4f}\n")

    print(f"結果已儲存至 {txt_path}")
    return txt_path
def calc_mdd(arr_series, plot=False, filter_zero=False):
    import plotly.graph_objects as go

    raw = arr_series.to_numpy()

    if filter_zero:
        mask = raw != 0
        arr  = raw[mask]
        x    = np.where(mask)[0].tolist()
        if len(arr) == 0:
            return 0.0
    else:
        arr = np.array(raw)
        x   = list(range(len(arr)))

    peak     = np.maximum.accumulate(arr)
    drawdown = np.where(peak != 0, (arr - peak) / np.where(peak != 0, peak, 1), 0.0)
    mdd      = drawdown.min()

    if plot:
        peak_arr_idx = [0]
        for i in range(1, len(arr)):
            if peak[i] > peak[i - 1]:
                peak_arr_idx.append(i)

        drawdown_segments = []
        i = 0
        while i < len(arr):
            if drawdown[i] < 0:
                seg_start = i
                peak_val  = peak[i]
                peak_pos  = i
                while peak_pos > 0 and arr[peak_pos] != peak_val:
                    peak_pos -= 1
                seg_end = i
                while seg_end < len(arr) - 1 and peak[seg_end + 1] == peak_val:
                    seg_end += 1
                min_pos = seg_start + int(np.argmin(arr[seg_start:seg_end + 1]))
                drawdown_segments.append((peak_pos, min_pos, peak_val, arr[min_pos]))
                i = seg_end + 1
            else:
                i += 1

        label = "Position Value" if filter_zero else "Whole Asset"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=arr.tolist(),
            mode="lines", line=dict(color="blue", width=2), name=label
        ))
        fig.add_trace(go.Scatter(
            x=x, y=peak.tolist(),
            mode="lines", line=dict(color="green", width=1.5, dash="dash"), name="Peak"
        ))
        fig.add_trace(go.Scatter(
            x=[x[i] for i in peak_arr_idx],
            y=[arr[i] for i in peak_arr_idx],
            mode="markers",
            marker=dict(color="green", size=8, symbol="triangle-up"),
            name="新高點"
        ))
        for p_pos, m_pos, p_val, m_val in drawdown_segments:
            dd_pct = (m_val - p_val) / p_val * 100
            is_mdd = abs(dd_pct / 100 - abs(mdd)) ==0
            color  = "red" if is_mdd else "orange"
            width  = 2.5  if is_mdd else 1.5
            fig.add_shape(
                type="line",
                x0=x[p_pos], y0=p_val, x1=x[m_pos], y1=m_val,
                line=dict(color=color, width=width, dash="dot")
            )
            fig.add_annotation(
                x=x[m_pos], y=m_val,
                text=f"{dd_pct:.1f}%",
                showarrow=False, font=dict(color=color, size=11), yshift=-15
            )

        if filter_zero:
            title, xaxis_title, yaxis_title = (
                f"Position Value 回撤分析（MDD={mdd:.4f}）", "天數（原始 index）", "Position Value"
            )
            html_file = "mdd_position_plot.html"
        else:
            title, xaxis_title, yaxis_title = (
                f"Whole Asset 與回撤分析（MDD={mdd:.4f}）", "天數", "Whole Asset"
            )
            html_file = "mdd_plot.html"

        fig.update_layout(
            title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
            height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        fig.write_html(html_file)
        fig.show()

    return mdd
def calc_whole_asset_stats(whole_asset):
    whole_asset = (whole_asset).to_numpy()
    arr = np.array(whole_asset)
    positive_rate = (arr > 0).sum() / len(arr)
    median_asset = float(np.median(arr))
    mean_asset = float(np.mean(arr))
    return positive_rate ,median_asset , mean_asset
def sweep_params(daily_df: pl.DataFrame,
                 position_size: float = 1.0,
                 fee_rate: float = 0.003,
                 mode: str = 'system1'):

    import plotly.graph_objects as go
    from itertools import product
    from tqdm import tqdm
    results = []

    # 參數範圍
    leave_terms  = range(5, 31)
    sys1_terms   = range(5, 51)
    sys2_terms   = range(10, 61)
    # 先算出合法組合
    valid_combinations = [
        (leave_term, sys1, sys2)
        for leave_term, sys1, sys2 in product(leave_terms, sys1_terms, sys2_terms)
        if sys2 - sys1 >= 1 and sys1 > leave_term and leave_term >= 1
    ]
    rules = {
    "leave_term":       "5 ~ 30",
    "enter_term_sys1":  "5 ~ 50，且 sys1 > leave_term",
    "enter_term_sys2":  "10 ~ 60，且 sys2 - sys1 >= 10",
    "vertical_barrier": "sys2 + 5",
    }
    for leave_term, sys1, sys2 in tqdm(valid_combinations, desc="Sweeping params"):
        # 約束條件：sys2 > sys1 > leave_term >= 5，且 sys2 - sys1 >= 10
    
        vertical_barrier = sys2 + 5

        result_df = turtle_trading_system(
            daily_df, sys1, sys2, leave_term, vertical_barrier, position_size, fee_rate, mode
        )

        cum_profits = result_df["profit"].cum_sum()
        position_values = result_df["position_value"].fill_null(0)
        whole_asset = cum_profits + position_values
        final_whole_asset = whole_asset[-1]
        # 1. 勝率
        win_rate = calc_win_rate(result_df)

        # 2. MDD
        mdd = calc_mdd(whole_asset)
        mdd = calc_mdd(position_values, plot=True, filter_zero=True)
        # 3. 最後 whole_asset
        final_whole_asset = whole_asset[-1]
        # 4. MDD
        positive_rate ,median_asset , mean_asset= calc_whole_asset_stats(whole_asset)                                

        results.append({
            "enter_term_sys1": sys1,
            "enter_term_sys2": sys2,
            "leave_term": leave_term,
            "win_rate": win_rate,
            "mdd": mdd,
            "final_whole_asset": final_whole_asset,
            "positive_rate": positive_rate,
            "median_asset": median_asset,
            "mean_asset": mean_asset    
        })
        
    result_df_all = pl.DataFrame(results)
    # 固定 sys2, leave_term，看 sys1 的敏感度
    sensitivity_sys1 = (
        result_df_all.group_by(["enter_term_sys2", "leave_term"])
        .agg(pl.col("final_whole_asset").std().alias("std_by_sys1"))
        .get_column("std_by_sys1").mean()
    )

    # 固定 sys1, leave_term，看 sys2 的敏感度
    sensitivity_sys2 = (
        result_df_all.group_by(["enter_term_sys1", "leave_term"])
        .agg(pl.col("final_whole_asset").std().alias("std_by_sys2"))
        .get_column("std_by_sys2").mean()
    )

    # 固定 sys1, sys2，看 leave_term 的敏感度
    sensitivity_leave = (
        result_df_all.group_by(["enter_term_sys1", "enter_term_sys2"])
        .agg(pl.col("final_whole_asset").std().alias("std_by_leave"))
        .get_column("std_by_leave").mean()
    )

    print(f"sys1 敏感度：{sensitivity_sys1:.4f}")
    print(f"sys2 敏感度：{sensitivity_sys2:.4f}")
    print(f"leave_term 敏感度：{sensitivity_leave:.4f}")
    # 找最佳
    best = max(results, key=lambda x: x["final_whole_asset"])
    print(f"最佳組合: sys1={best['enter_term_sys1']}, sys2={best['enter_term_sys2']}, "
          f"leave_term={best['leave_term']}, final_whole_asset={best['final_whole_asset']:.4f}")
   
    # 敏感度改成 CV
    cv_sys1 = (
        result_df_all.group_by(["enter_term_sys2", "leave_term"])
        .agg((pl.col("final_whole_asset").std() / pl.col("final_whole_asset").mean()).alias("cv"))
        .get_column("cv").mean()
    )
    cv_sys2 = (
        result_df_all.group_by(["enter_term_sys1", "leave_term"])
        .agg((pl.col("final_whole_asset").std() / pl.col("final_whole_asset").mean()).alias("cv"))
        .get_column("cv").mean()
    )
    cv_leave = (
        result_df_all.group_by(["enter_term_sys1", "enter_term_sys2"])
        .agg((pl.col("final_whole_asset").std() / pl.col("final_whole_asset").mean()).alias("cv"))
        .get_column("cv").mean()
    )
    print(f"sys1 敏感度(/均值)：{cv_sys1 :.4f}")
    print(f"sys2 敏感度(/均值)：{cv_sys2 :.4f}")
    print(f"leave_term 敏感度(/均值)：{cv_leave :.4f}")    
    save_sweep_result(results, rules,len(valid_combinations),mode,cv_sys1,cv_sys2,cv_leave)
    # 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=[r["enter_term_sys1"] for r in results],
        y=[r["enter_term_sys2"] for r in results],
        z=[r["leave_term"]      for r in results],
        mode="markers",
        marker=dict(
            size=4,
            color=[r["final_whole_asset"] for r in results],
            colorscale="Viridis",
            colorbar=dict(title="Final Whole Asset"),
            showscale=True,
        ),
        text=[f"sys1={r['enter_term_sys1']}, sys2={r['enter_term_sys2']}, "
              f"leave={r['leave_term']}<br>asset={r['final_whole_asset']:.4f}"
              for r in results],
        hoverinfo="text"
    ))
    fig.update_layout(
        title=f"Parameter Sweep（mode={mode}）",
        scene=dict(
            xaxis_title="enter_term_sys1",
            yaxis_title="enter_term_sys2",
            zaxis_title="leave_term",
        ),
        height=700
    )
    fig.write_html("sweep_params.html")
    fig.show()
   
    return result_df_all
def calc_win_rate(result_df: pl.DataFrame):
    profits = result_df["profit"].to_numpy()
    buy_actions  = result_df["buy_action"].to_numpy()
    sell_actions = result_df["sell_action"].to_numpy()
    buy_profits  = profits[buy_actions == 1]
    sell_profits = profits[sell_actions == 1]
    n_trades = min(len(buy_profits), len(sell_profits))
    trade_pnl = sell_profits[:n_trades] + buy_profits[:n_trades]
    win_rate = (trade_pnl > 0).sum() / n_trades if n_trades > 0 else 0
    return win_rate
def calc_profit_loss_ratio(result_df: pl.DataFrame):
    profits = result_df["profit"].to_numpy()
    buy_actions  = result_df["buy_action"].to_numpy()
    sell_actions = result_df["sell_action"].to_numpy()
    buy_profits  = profits[buy_actions == 1]
    sell_profits = profits[sell_actions == 1]
    n_trades = min(len(buy_profits), len(sell_profits))
    trade_pnl = sell_profits[:n_trades] + buy_profits[:n_trades]
    wins   = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    avg_win  = wins.mean()   if len(wins)   > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0  # 負數

    profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
    return profit_loss_ratio
def calc_expectancy(result_df: pl.DataFrame):
    profits      = result_df["profit"].to_numpy()
    buy_actions  = result_df["buy_action"].to_numpy()
    sell_actions = result_df["sell_action"].to_numpy()
    buy_profits  = profits[buy_actions == 1]
    sell_profits = profits[sell_actions == 1]
    n_trades  = min(len(buy_profits), len(sell_profits))
    trade_pnl = sell_profits[:n_trades] + buy_profits[:n_trades]
    if n_trades == 0:
        return 0.0
    wins   = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    win_rate = len(wins) / n_trades
    loss_rate = 1 - win_rate
    avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0  # 負數
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    return expectancy
def mean_abs_gradient(df, metric_col, param_col, fixed_cols):
        grads = []
        for _, grp in df.sort([*fixed_cols, param_col]).group_by(fixed_cols):
            grp   = grp.sort(param_col)
            vals  = grp[metric_col].to_numpy().astype(float)
            steps = grp[param_col].to_numpy().astype(float)
            if len(vals) > 1:
                grads.extend((np.abs(np.diff(vals)) / np.diff(steps)).tolist())
        return float(np.mean(grads)) if grads else 0.0


def sweep_params_interactive(daily_df: pl.DataFrame, vertical_barrier: float = 30, position_size: float = 1.0, fee_rate: float = 0.003, mode: str = 'system1'):

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from itertools import product
    from tqdm import tqdm

    param_names = ["leave_term", "enter_term_sys1", "enter_term_sys2"]
    param_defaults = {
        "leave_term":      (5, 30),
        "enter_term_sys1": (5, 30),
        "enter_term_sys2": (10, 60),
    }

    print("\n可調整的參數：")
    for i, name in enumerate(param_names):
        lo, hi = param_defaults[name]
        print(f"  [{i+1}] {name}  (預設範圍: {lo} ~ {hi})")

    fixed_input = input("\n請輸入要固定的參數編號（多個用逗號分隔，不固定直接按 Enter）: ").strip()

    fixed_indices = set()
    if fixed_input:
        for part in fixed_input.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(param_names):
                fixed_indices.add(int(part) - 1)

    fixed_params = {}
    free_ranges  = {}

    print()
    for i, name in enumerate(param_names):
        lo_default, hi_default = param_defaults[name]
        if i in fixed_indices:
            val = input(f"  {name} 固定值: ").strip()
            fixed_params[name] = int(val)
        else:
            lo = input(f"  {name} 起始值 (預設 {lo_default}): ").strip()
            hi = input(f"  {name} 結束值 (預設 {hi_default}): ").strip()
            free_ranges[name] = range(
                int(lo) if lo else lo_default,
                (int(hi) if hi else hi_default) + 1
            )

    leave_vals = [fixed_params["leave_term"]]      if "leave_term"      in fixed_params else list(free_ranges["leave_term"])
    sys1_vals  = [fixed_params["enter_term_sys1"]] if "enter_term_sys1" in fixed_params else list(free_ranges["enter_term_sys1"])
    sys2_vals  = [fixed_params["enter_term_sys2"]] if "enter_term_sys2" in fixed_params else list(free_ranges["enter_term_sys2"])

    if mode == 'system1':
        valid_combinations = [
            (leave_term, sys1, sys2)
            for leave_term, sys1, sys2 in product(leave_vals, sys1_vals, sys2_vals)
            if sys1 > leave_term and leave_term >= 5
        ]
    elif mode == 'system2':
        valid_combinations = [
            (leave_term, sys1, sys2)
            for leave_term, sys1, sys2 in product(leave_vals, sys1_vals, sys2_vals)
            if sys2 > leave_term and leave_term >= 5
        ]
    else:
        valid_combinations = [
            (leave_term, sys1, sys2)
            for leave_term, sys1, sys2 in product(leave_vals, sys1_vals, sys2_vals)
            if sys2 - sys1 >= 10 and sys1 > leave_term and sys2 > leave_term and leave_term >= 5
        ]

    print(f"\n合法組合數: {len(valid_combinations)}")
    if len(valid_combinations) == 0:
        print("沒有合法組合，請重新確認參數範圍與約束條件。")
        return None

    rules = {
        "leave_term":       f"{leave_vals[0]} ~ {leave_vals[-1]}",
        "enter_term_sys1":  f"{sys1_vals[0]} ~ {sys1_vals[-1]}，且 sys1 > leave_term",
        "enter_term_sys2":  f"{sys2_vals[0]} ~ {sys2_vals[-1]}，且 sys2 - sys1 >= 10",
        "vertical_barrier": str(vertical_barrier),
    }

    results = []
    for leave_term, sys1, sys2 in tqdm(valid_combinations, desc="Sweeping params"):
        '''
        result_df = turtle_trading_system(
            daily_df, sys1, sys2, leave_term, vertical_barrier, position_size, fee_rate, mode
        )
        result_df = turtle_trading_system_with_budget(
            daily_df, sys1, sys2, leave_term, vertical_barrier, own_capital, invest_pct, 0.00000001,fee_rate,mode
        )
        '''
        initial_fund=10000
        result_df=turtle_trading_system_gold_standard(daily_df,sys1,sys2,leave_term,vertical_barrier,initial_fund,0.00000001,fee_rate,2000,True, mode)
        own_capital=10000
        invest_pct=0.5
        
        
        cum_profits     = result_df["profit"].cum_sum()
        
        position_values = result_df["position_value"].fill_null(0)
        whole_asset     = result_df["cash"] + position_values-initial_fund
        '''
        if sys1==20 and leave_term==10:
            print("final cash:", result_df["cash"][-1])
            print("final whole_asset:", whole_asset[-1])
            print("cum_profits:", cum_profits[-1])
            print("position_values:", position_values[-1])
        '''
        win_rate          = calc_win_rate(result_df)
        profit_loss_ratio = calc_profit_loss_ratio(result_df)
        expectancy        = calc_expectancy(result_df)
        mdd               = calc_mdd(position_values, filter_zero=True)
        positive_rate, median_asset, mean_asset = calc_whole_asset_stats(whole_asset)
        final_profit_rate=whole_asset[-1]/(own_capital*invest_pct)
        results.append({
            "enter_term_sys1":   sys1,
            "enter_term_sys2":   sys2,
            "leave_term":        leave_term,
            "win_rate":          win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "expectancy":        expectancy/2000,
            "mdd":               mdd,
            "final_whole_asset": float(whole_asset[-1]),
            "positive_rate":     positive_rate,
            "median_asset":      median_asset,
            "mean_asset":        mean_asset,
            #"final_profit_rate": final_profit_rate
        })

    result_df_all = pl.DataFrame(results)

    virdict = evaluate_single_strategy_system1(result_df_all)
    result_df_all.write_csv("index_parameter_result.csv")

    # ---- 畫圖 ----
    metrics      = ["mdd", "win_rate", "final_whole_asset", "profit_loss_ratio", "expectancy"]
    metric_names = ["MDD", "Win Rate", "Final Asset relative to initial fund", "Profit Loss Ratio", "Expectancy relative to cash position"]

    def build_z(df, x_col, y_col, metric_col):
        xs = sorted(df[x_col].unique().to_list())
        ys = sorted(df[y_col].unique().to_list())
        xi = {v: i for i, v in enumerate(xs)}
        yi = {v: i for i, v in enumerate(ys)}
        z  = np.full((len(ys), len(xs)), np.nan)
        for row in df.to_dicts():
            z[yi[row[y_col]]][xi[row[x_col]]] = row[metric_col]
        return xs, ys, z

    if mode in ('system1', 'system2'):
        x_col   = "enter_term_sys1" if mode == 'system1' else "enter_term_sys2"
        x_label = "sys1" if mode == 'system1' else "sys2"

        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'surface'}] * 3, [{'type': 'surface'}] * 3],
            subplot_titles=metric_names,
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
        )
        scene_keys = ["scene", "scene2", "scene3", "scene4", "scene5"]
        for idx, (metric, mname) in enumerate(zip(metrics, metric_names)):
            row = idx // 3 + 1
            col = idx % 3 + 1
            xs, ys, z = build_z(result_df_all, x_col, "leave_term", metric)
            fig.add_trace(
                go.Surface(
                    x=xs, y=ys, z=z,
                    colorscale="Viridis",
                    showscale=False,
                    name=mname,
                    hovertemplate=f"{x_label}=%{{x}}<br>leave=%{{y}}<br>{metric}=%{{z:.4f}}<extra></extra>",
                ),
                row=row, col=col,
            )
        scene_kwargs = dict(xaxis_title=x_label, yaxis_title="leave_term")
        layout_scenes = {}
        for idx, skey in enumerate(scene_keys):
            layout_scenes[skey] = dict(**scene_kwargs, zaxis_title=metric_names[idx])
        fig.update_layout(
            title=f"Parameter Sweep Surface（mode={mode}）",
            height=900,
            **layout_scenes,
        )

    else:
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'scatter3d'}] * 3, [{'type': 'scatter3d'}] * 3],
            subplot_titles=metric_names,
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
        )
        rows_data = result_df_all.to_dicts()
        x_vals = [r["enter_term_sys1"] for r in rows_data]
        y_vals = [r["enter_term_sys2"] for r in rows_data]
        z_vals = [r["leave_term"]      for r in rows_data]

        for idx, (metric, mname) in enumerate(zip(metrics, metric_names)):
            row = idx // 3 + 1
            col = idx % 3 + 1
            c_vals = [r[metric] for r in rows_data]
            fig.add_trace(
                go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=c_vals,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(
                            title=mname,
                            len=0.35,
                            y=0.78 if row == 1 else 0.22,
                        ),
                    ),
                    text=[
                        f"sys1={r['enter_term_sys1']}, sys2={r['enter_term_sys2']}, "
                        f"leave={r['leave_term']}<br>{metric}={r[metric]:.4f}"
                        for r in rows_data
                    ],
                    hoverinfo="text",
                    name=mname,
                ),
                row=row, col=col,
            )
        scene_kwargs = dict(xaxis_title="sys1", yaxis_title="sys2", zaxis_title="leave_term")
        layout_scenes = {skey: scene_kwargs for skey in ["scene", "scene2", "scene3", "scene4", "scene5"]}
        fig.update_layout(
            title="Parameter Sweep 3D Scatter（both）",
            height=900,
            **layout_scenes,
        )

    fig.write_html("sweep_params_sys1_60.html")
    fig.show()
    return result_df_all



def bm_monte_carlo(daily_df,n_simulations=1000, fee_rate=0.003, seed=None):
    """
    Monte Carlo benchmark：每天隨機決定是否持有
    回傳所有模擬的 trade_profits 平均值
    """
    daily = daily_df.sort("date")
    closes = daily["close"].to_list()
    dates = daily["date"].to_list()
    n = len(closes)

    if seed is not None:
        np.random.seed(seed)
    
    all_trade_profits = []

    for _ in range(n_simulations):
        # 每天隨機 0 或 1，決定是否持有
        position = np.random.randint(0, 2, size=n)  # shape: (n,)

        daily_position = []
        trade_profits = []

        prev_holding = False

        for i in range(n):
            holding = bool(position[i])
            daily_position.append(closes[i] if holding else 0)

            if i == 0:
                if holding:
                    trade_profits.append(-closes[0] * fee_rate)  # 買入手續費
                else:
                    trade_profits.append(0)

            elif i == n - 1:
                if holding and prev_holding:
                    # 最後一天持有，賣出
                    trade_profits.append((closes[-1] - closes[-2]) - closes[-1] * fee_rate)
                elif holding and not prev_holding:
                    # 最後一天才買，馬上賣
                    trade_profits.append(-closes[-1] * fee_rate - closes[-1] * fee_rate)
                elif not holding and prev_holding:
                    # 昨天持有今天不持有，賣出
                    trade_profits.append(-closes[-2] * fee_rate + (closes[-1] - closes[-2]))
                    # 修正：其實應在 i 的前一天處理賣出
                else:
                    trade_profits.append(0)

            else:
                if holding and prev_holding:
                    trade_profits.append(closes[i] - closes[i-1])
                elif holding and not prev_holding:
                    trade_profits.append(-closes[i] * fee_rate)  # 買入
                elif not holding and prev_holding:
                    trade_profits.append((closes[i] - closes[i-1]) - closes[i] * fee_rate)  # 賣出
                else:
                    trade_profits.append(0)

            prev_holding = holding

        all_trade_profits.append(trade_profits)

    # 取每天的平均
    avg_trade_profits = np.mean(all_trade_profits, axis=0).tolist()
    avg_daily_position = [closes[i] for i in range(n)]  # position 僅供參考

    return avg_trade_profits, avg_daily_position
def evaluate_all(trade_df: pl.DataFrame, daily_df: pl.DataFrame, fee_rate: float = 0.003):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    daily = daily_df.sort("date")
    closes = daily["close"].to_list()
    dates = daily["date"].to_list()
    n = len(closes)

    # ==================== 通用計算函式 ====================
    def calc_metrics(profits: list, label: str):
        if not profits:
            return {}
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        win_rate = len(wins) / len(profits)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        equity = np.cumsum(profits)
        peak = np.maximum.accumulate(equity)
        mdd = (peak - equity).max()
        return {
            "strategy":      label,
            "n_trades":      len(profits),
            "win_rate_%":    round(win_rate * 100, 2),
            "avg_win":       round(avg_win, 4),
            "avg_loss":      round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "expectancy":    round(expectancy, 4),
            "mdd":           round(mdd, 4),
            "total_profit":  round(sum(profits), 4),
        }

    # ==================== Benchmark 1: Buy & Hold ====================
    def bm1():
        daily_position = []
        for i in range(n):
            daily_position.append(closes[i])  # 全程持有

        trade_profits = [0] * n  # 預設全部為 0

        # 買入日
        trade_profits[0] = -closes[0] * fee_rate  # 只有手續費，還沒賣出所以沒價差

        # 賣出日
        trade_profits[-1] = (closes[-1] - closes[0]) - (closes[0] + closes[-1]) * fee_rate

        return trade_profits, daily_position
    
   

    # ==================== Benchmark 3: 至少持有30天 ====================
    def bm3():
        trade_profits = []
        daily_position = [0.0] * n
        i = 0
        while i < n - 1:
            buy_price = closes[i]
            buy_fee = buy_price * fee_rate
            hold_days = 0
            daily_position[i] = closes[i]
            i += 1
            sold = False
            while i < n - 1:
                daily_position[i] = closes[i]
                hold_days += 1
                if hold_days >= 30 and closes[i - 1] > buy_price:
                    sell_price = closes[i]
                    sell_fee = sell_price * fee_rate
                    profit = (sell_price - buy_price) - buy_fee - sell_fee
                    trade_profits.append(profit)
                    i += 1
                    sold = True
                    break
                i += 1
            if not sold:
                sell_price = closes[-1]
                sell_fee = sell_price * fee_rate
                profit = (sell_price - buy_price) - buy_fee - sell_fee
                trade_profits.append(profit)
        if daily_position[n - 2] != 0:##最後一天不可以持倉，強制賣出
            sell_price = closes[-1]
            sell_fee = sell_price * fee_rate
            profit = (sell_price - buy_price) - buy_fee - sell_fee
            trade_profits.append(profit)
            daily_position[n - 1] = closes[n - 1]

        return trade_profits, daily_position

    # ==================== 策略本身 ====================
    def get_strategy_curves():
        # profit 曲線：對齊到 daily dates
        profit_by_date = {}
        position_by_date = {}
        for row in trade_df.iter_rows(named=True):
            d = row["date"]
            profit_by_date[d] = row["profit"]
            position_by_date[d] = row["position_value"]

        daily_profits = []
        daily_positions = []
        cum = 0
        for d, c in zip(dates, closes):
            p = profit_by_date.get(d, 0)
            pos = position_by_date.get(d)
            pos = pos if pos is not None else 0
            daily_profits.append(p)
            daily_positions.append(pos)

        return daily_profits, daily_positions

    # ==================== 執行 ====================
    bm1_profits, bm1_pos = bm1()
    bm3_profits, bm3_pos = bm3()
    st_profits,  st_pos  = get_strategy_curves()

    # 對齊長度到 n（daily）
    def pad(lst, length):
        return lst + [0] * (length - len(lst))

    bm1_cum = np.cumsum(pad(bm1_profits, n))
  
    bm3_cum = np.cumsum(pad(bm3_profits, n))
    st_cum  = np.cumsum(pad(st_profits,  n))
    bm1_whole = bm1_cum + np.array(bm1_pos)
    bm3_whole = bm3_cum + np.array(pad(bm3_pos, n))
    st_whole  = st_cum  + np.array(pad(st_pos,  n))
    # ==================== 指標表 ====================
    rows = [
        calc_metrics(st_profits,  "My Strategy"),
        calc_metrics(bm1_profits, "BM1: Buy & Hold"),
        calc_metrics(bm3_profits, "BM3: Hold 30d then sell when profit"),
    ]
    result_df = pl.DataFrame(rows)
    result_df.write_csv("strategy_comparison.csv")
    print(result_df)
    # ==================== 圖像化 ====================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["圖1: 累積已實現獲利", "圖2: Whole Asset（累積獲利 + 持倉價值）"],
        row_heights=[0.5, 0.5]
    )
    # 圖1: profit 曲線
    for cum, label, color in [
        (st_cum,  "My Strategy",                     "blue"),
        (bm1_cum, "BM1: Buy & Hold",                 "orange"),
        (bm3_cum, "BM3: Hold 30d then sell",         "red"),
    ]:
        fig.add_trace(go.Scatter(
            x=dates, y=cum,
            mode="lines", name=label,
            line=dict(color=color)
        ), row=1, col=1)
    # 圖2: whole asset 曲線
    for whole, label, color in [
        (st_whole,  "My Strategy",                     "blue"),
        (bm1_whole, "BM1: Buy & Hold",                 "orange"),
        (bm3_whole, "BM3: Hold 30d then sell",         "red"),
    ]:
        fig.add_trace(go.Scatter(
            x=dates, y=whole,
            mode="lines", name=label,
            line=dict(color=color),
            showlegend=False
        ), row=2, col=1)
    fig.update_layout(
        title="策略比較",
        height=800,
        xaxis_rangeslider_visible=False
    )
    fig.write_html("strategy_comparison.html")
    fig.show()

    return result_df
def evaluate_single_strategy_system1(result_df_all: pl.DataFrame) -> dict:

    df = result_df_all.to_pandas()
    failures = []
    verdicts = {}

    params  = ["enter_term_sys1", "enter_term_sys2", "leave_term"]
    metrics = ["final_whole_asset", "win_rate", "mdd", "positive_rate", "median_asset", "mean_asset", "profit_loss_ratio", "expectancy"]

    # =========================================
    # 1. 分布判定
    # =========================================
    fail_median  = (df["median_asset"] < 0).mean() >= 0.2
    fail_mean    = (df["mean_asset"] < 0).mean() >= 0.2
    fail_asset   = (df["final_whole_asset"] < 0).mean() >= 0.2
    fail_posrate = (df["positive_rate"] < 0).mean() >= 0.4

    if fail_median or fail_mean:
        failures.append("median/mean_asset: 超過20%組合資產為負")
    if fail_asset:
        failures.append("final_whole_asset: 超過20%組合最終資產為負")
    if fail_posrate:
        failures.append("positive_rate: 超過40%組合獲利比例為負")

    # positive_rate > 0.8 的比例
    high_posrate_ratio = (df["positive_rate"] > 0.8).mean()
    verdicts["high_positive_rate_ratio"] = round(float(high_posrate_ratio), 4)
    if high_posrate_ratio < 0.3:
        failures.append(f"positive_rate>0.8 的組合比例過低 ({high_posrate_ratio:.2%})")

    # =========================================
    # 2. 統計檢定（t-test only）
    # =========================================
    t_stat, p_value = stats.ttest_1samp(df["final_whole_asset"], popmean=0)
    if p_value > 0.05 or t_stat < 0:
        failures.append(f"t-test: 獲利不顯著 (t={t_stat:.2f}, p={p_value:.4f})")
    verdicts["t_stat"]  = round(float(t_stat), 4)
    verdicts["p_value"] = round(float(p_value), 4)

    # =========================================
    # 3. 風險與勝率
    # =========================================
    mdd_median = df["mdd"].median()
    wr_median  = df["win_rate"].median()

    if mdd_median < -0.7:
        failures.append(f"MDD 中位數過大 ({mdd_median:.4f})")
    if wr_median < 0.3:
        failures.append(f"勝率中位數過低 ({wr_median:.4f})")

    verdicts["mdd_median"]      = round(float(mdd_median), 4)
    verdicts["win_rate_median"] = round(float(wr_median), 4)

    # =========================================
    # 4. 盈虧比
    # =========================================
    plr_median = df["profit_loss_ratio"].median()
    plr_mean   = df["profit_loss_ratio"].mean()

    if plr_median < 1.0:
        failures.append(f"盈虧比中位數過低 ({plr_median:.4f})")
    
    verdicts["profit_loss_ratio_median"] = round(float(plr_median), 4)
    verdicts["profit_loss_ratio_mean"]   = round(float(plr_mean), 4)
    # 4.5 Expectancy
    # =========================================
    exp_median = df["expectancy"].median()
    exp_mean   = df["expectancy"].mean()

    if exp_median < 0:
        failures.append(f"Expectancy 中位數為負 ({exp_median:.4f})")

    verdicts["expectancy_median"] = round(float(exp_median), 4)
    verdicts["expectancy_mean"]   = round(float(exp_mean), 4)
    # =========================================
    # 5. 參數變動度（normalized gradient）
    # =========================================
    GRAD_THRESHOLD = 0.10
    grad_results = {}
    for metric in ["final_whole_asset", "win_rate", "mdd", "positive_rate", "profit_loss_ratio", "expectancy"]:
        metric_mean = abs(df[metric].mean())
        for param, fixed in [
            ("enter_term_sys1", ["enter_term_sys2", "leave_term"]),
            ("enter_term_sys2", ["enter_term_sys1", "leave_term"]),
            ("leave_term",      ["enter_term_sys1", "enter_term_sys2"]),
        ]:
            grad = mean_abs_gradient(result_df_all, metric, param, fixed)
            norm_grad = grad / metric_mean if metric_mean != 0 else 0.0
            key = f"{metric}__{param}"
            grad_results[key] = round(float(norm_grad), 4)
            if norm_grad > GRAD_THRESHOLD:
                failures.append(f"參數敏感: {metric} 對 {param} 變動過大 (norm_grad={norm_grad:.4f})")

    verdicts["param_sensitivity"] = grad_results

    # =========================================
    # 6. 參數與指標相關係數
    # =========================================
    available_metrics = [m for m in metrics if m in df.columns]
    corr = df[params + available_metrics].corr().loc[params, available_metrics]
    verdicts["correlation"] = corr.round(3).to_dict()

    # =========================================
    # 7. 最終判定
    # =========================================
    verdicts["failures"] = failures
    verdicts["verdict"]  = "FAILURE" if failures else "PASS"

    if verdicts["verdict"] == "PASS":
        conclusion = "所有參數空間的組合皆符合獲利條件，策略具備穩健性。"
    else:
        conclusion = "並非所有參數空間的組合都符合獲利情況，策略存在以下風險需注意。"
    verdicts["conclusion"] = conclusion

    # =========================================
    # 輸出
    # =========================================
    print("=" * 60)
    print(f"策略評判結果：{verdicts['verdict']}")
    print(f"結論：{conclusion}")
    print("=" * 60)

    if failures:
        print("\n失敗原因：")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("\n所有指標通過")

    print(f"\n[統計檢定]")
    print(f"  t-test : t={verdicts['t_stat']}, p={verdicts['p_value']}")

    print(f"\n[風險與勝率]")
    print(f"  MDD 中位數              : {verdicts['mdd_median']}")
    print(f"  勝率中位數              : {verdicts['win_rate_median']}")
    print(f"  盈虧比中位數            : {verdicts['profit_loss_ratio_median']}")
    print(f"  盈虧比平均              : {verdicts['profit_loss_ratio_mean']}")
    print(f"  positive_rate>0.8 比例  : {verdicts['high_positive_rate_ratio']:.2%}")
    print(f"  Expectancy 中位數          : {verdicts['expectancy_median']}")
    print(f"  Expectancy 平均            : {verdicts['expectancy_mean']}")
    print(f"\n[參數敏感度（normalized gradient，>{GRAD_THRESHOLD} 為敏感）]")
    for k, v in grad_results.items():
        flag = " ← 敏感" if v > GRAD_THRESHOLD else ""
        print(f"  {k}: {v}{flag}")

    print(f"\n[參數與指標相關係數]")
    print(corr.round(3).to_string())

    print("=" * 60)

    return verdicts
def find_plateau(result_df_all: pl.DataFrame,
                 free_params: list,
                 min_asset: float = 0,
                 min_expectancy: float = 0,
                 min_plr: float = 1.0,
                 mode: str = 'system1',
                 plot: bool = True) -> list:

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from itertools import combinations

    all_params = ["enter_term_sys1", "enter_term_sys2", "leave_term"]
    df = result_df_all.to_pandas()

    # =========================================
    # 1. 過濾滿足基本條件的點
    # =========================================
    def is_valid(row):
        return (row["final_whole_asset"] > min_asset and
                row["expectancy"] > min_expectancy and
                row["profit_loss_ratio"] > min_plr)

    valid_mask = df.apply(is_valid, axis=1)
    valid_df   = df[valid_mask].copy()

    if valid_df.empty:
        print("沒有滿足條件的參數組合")
        return []

    valid_set = set(
        tuple(row[p] for p in all_params)
        for _, row in valid_df.iterrows()
    )

    # =========================================
    # 2. 鄰域檢查（只對 free_params 做 ±1）
    # =========================================
    def all_neighbors_valid(row):
        for param in free_params:
            for delta in [-1, 1]:
                neighbor = tuple(
                    row[p] + delta if p == param else row[p]
                    for p in all_params
                )
                neighbor_rows = df[
                    (df[all_params[0]] == neighbor[0]) &
                    (df[all_params[1]] == neighbor[1]) &
                    (df[all_params[2]] == neighbor[2])
                ]
                if not neighbor_rows.empty and neighbor not in valid_set:
                    return False
        return True

    plateau_mask = valid_df.apply(all_neighbors_valid, axis=1)
    plateau_df   = valid_df[plateau_mask].copy()

    if plateau_df.empty:
        print("沒有找到符合鄰域條件的高原點")
        return []

    # =========================================
    # 3. BFS 群聚
    # =========================================
    plateau_set = set(
        tuple(row[p] for p in all_params)
        for _, row in plateau_df.iterrows()
    )

    visited  = set()
    plateaus = []

    def bfs(start):
        cluster = []
        queue   = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            cluster.append(node)
            for param_idx, param in enumerate(all_params):
                if param not in free_params:
                    continue
                for delta in [-1, 1]:
                    neighbor = list(node)
                    neighbor[param_idx] += delta
                    neighbor = tuple(neighbor)
                    if neighbor in plateau_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return cluster

    for point in plateau_set:
        if point not in visited:
            plateaus.append(bfs(point))

    print(f"\n找到 {len(plateaus)} 個高原")
    for i, p in enumerate(plateaus):
        print(f"  高原 {i+1}：{len(p)} 個參數組合")

    # =========================================
    # 4. 計算每個高原的統計
    # =========================================
    def get_plateau_stats(cluster):
        rows = []
        for point in cluster:
            mask = (df[all_params[0]] == point[0])
            for j, p in enumerate(all_params[1:], 1):
                mask &= (df[p] == point[j])
            rows.append(df[mask].iloc[0])
        sub = pd.DataFrame(rows)
        return {
            "median_asset":  sub["final_whole_asset"].median(),
            "median_expect": sub["expectancy"].median(),
            "median_plr":    sub["profit_loss_ratio"].median(),
            "size":          len(cluster),
        }

    stats_list = [get_plateau_stats(p) for p in plateaus]

    for i, s in enumerate(stats_list):
        print(f"\n  高原 {i+1} 指標：")
        print(f"    final_whole_asset median : {s['median_asset']:.4f}")
        print(f"    expectancy median        : {s['median_expect']:.4f}")
        print(f"    profit_loss_ratio median : {s['median_plr']:.4f}")
        print(f"    面積（組合數）           : {s['size']}")

    # =========================================
    # 5. 兩兩比較，計算分數
    # =========================================
    n      = len(plateaus)
    scores = [0] * n

    for i, j in combinations(range(n), 2):
        si, sj = stats_list[i], stats_list[j]

        if si["median_asset"] >= sj["median_asset"] * 1.1:
            scores[i] += 1
        elif sj["median_asset"] >= si["median_asset"] * 1.1:
            scores[j] += 1

        if si["median_expect"] >= sj["median_expect"] * 1.1:
            scores[i] += 1
        elif sj["median_expect"] >= si["median_expect"] * 1.1:
            scores[j] += 1

        if si["median_plr"] >= sj["median_plr"] * 1.1:
            scores[i] += 1
        elif sj["median_plr"] >= si["median_plr"] * 1.1:
            scores[j] += 1

        if si["size"] >= sj["size"] * 3:
            scores[i] += 2
        elif sj["size"] >= si["size"] * 3:
            scores[j] += 2
        elif si["size"] >= sj["size"] * 2:
            scores[i] += 1
        elif sj["size"] >= si["size"] * 2:
            scores[j] += 1

    print(f"\n各高原得分：")
    for i, s in enumerate(scores):
        print(f"  高原 {i+1}：{s} 分")

    max_score   = max(scores)
    best_indices = [i for i, s in enumerate(scores) if s == max_score]
    best_sets   = [set(plateaus[i]) for i in best_indices]
    valid_only_set = valid_set - plateau_set  # 寬鬆條件但不在高原內的點

    print(f"\n最佳高原（得分 {max_score}）：高原 {[i+1 for i in best_indices]}")

    # =========================================
    # 6. 畫圖
    # =========================================
    if plot:
        metrics      = ["win_rate", "profit_loss_ratio", "expectancy", "mdd", "final_whole_asset"]
        metric_names = ["Win Rate", "Profit Loss Ratio", "Expectancy", "MDD", "Final Asset"]
        rows_data    = df.to_dict("records")

        def point_category(point_tuple):
            if any(point_tuple in bs for bs in best_sets):
                return "best"
            elif point_tuple in plateau_set:
                return "plateau"
            elif point_tuple in valid_only_set:
                return "valid"
            else:
                return "normal"

        categories = [
            point_category(tuple(r[p] for p in all_params))
            for r in rows_data
        ]

        color_map = {"normal": "lightgrey", "valid": "yellow", "plateau": "orange", "best": "red"}
        size_map  = {"normal": 3, "valid": 5, "plateau": 6, "best": 8}

        if mode in ('system1', 'system2'):
            x_col   = "enter_term_sys1" if mode == 'system1' else "enter_term_sys2"
            x_label = "sys1" if mode == 'system1' else "sys2"

            fig = make_subplots(
                rows=2, cols=3,
                specs=[[{'type': 'scatter3d'}] * 3, [{'type': 'scatter3d'}] * 3],
                subplot_titles=metric_names,
                horizontal_spacing=0.05,
                vertical_spacing=0.08,
            )

            x_vals = [r[x_col]        for r in rows_data]
            y_vals = [r["leave_term"] for r in rows_data]

            for idx, (metric, mname) in enumerate(zip(metrics, metric_names)):
                row = idx // 3 + 1
                col = idx % 3 + 1
                z_vals = [r[metric] for r in rows_data]

                # 一般點（Viridis）
                normal_idx = [i for i, c in enumerate(categories) if c == "normal"]
                fig.add_trace(go.Scatter3d(
                    x=[x_vals[i] for i in normal_idx],
                    y=[y_vals[i] for i in normal_idx],
                    z=[z_vals[i] for i in normal_idx],
                    mode="markers",
                    marker=dict(size=3, color=[z_vals[i] for i in normal_idx],
                                colorscale="Viridis", showscale=False),
                    hovertemplate=f"{x_label}=%{{x}}<br>leave=%{{y}}<br>{metric}=%{{z:.4f}}<extra></extra>",
                    name="一般",
                    showlegend=(idx == 0),
                ), row=row, col=col)

                # 寬鬆條件點（黃色）
                valid_idx = [i for i, c in enumerate(categories) if c == "valid"]
                if valid_idx:
                    fig.add_trace(go.Scatter3d(
                        x=[x_vals[i] for i in valid_idx],
                        y=[y_vals[i] for i in valid_idx],
                        z=[z_vals[i] for i in valid_idx],
                        mode="markers",
                        marker=dict(size=5, color="yellow", symbol="diamond"),
                        hovertemplate=f"{x_label}=%{{x}}<br>leave=%{{y}}<br>{metric}=%{{z:.4f}}<extra></extra>",
                        name="寬鬆條件",
                        showlegend=(idx == 0),
                    ), row=row, col=col)

                # 高原點（橘色）
                plateau_idx = [i for i, c in enumerate(categories) if c == "plateau"]
                if plateau_idx:
                    fig.add_trace(go.Scatter3d(
                        x=[x_vals[i] for i in plateau_idx],
                        y=[y_vals[i] for i in plateau_idx],
                        z=[z_vals[i] for i in plateau_idx],
                        mode="markers",
                        marker=dict(size=6, color="orange", symbol="diamond"),
                        hovertemplate=f"{x_label}=%{{x}}<br>leave=%{{y}}<br>{metric}=%{{z:.4f}}<extra></extra>",
                        name="高原",
                        showlegend=(idx == 0),
                    ), row=row, col=col)

                # 最佳高原點（紅色）
                best_idx = [i for i, c in enumerate(categories) if c == "best"]
                if best_idx:
                    fig.add_trace(go.Scatter3d(
                        x=[x_vals[i] for i in best_idx],
                        y=[y_vals[i] for i in best_idx],
                        z=[z_vals[i] for i in best_idx],
                        mode="markers",
                        marker=dict(size=8, color="red", symbol="diamond"),
                        hovertemplate=f"{x_label}=%{{x}}<br>leave=%{{y}}<br>{metric}=%{{z:.4f}}<extra></extra>",
                        name="最佳高原",
                        showlegend=(idx == 0),
                    ), row=row, col=col)

            scene_kwargs = dict(xaxis_title=x_label, yaxis_title="leave_term")
            layout_scenes = {}
            for idx, skey in enumerate(["scene","scene2","scene3","scene4","scene5"]):
                layout_scenes[skey] = dict(**scene_kwargs, zaxis_title=metric_names[idx])
            fig.update_layout(
                title=f"Parameter Plateau（mode={mode}）",
                height=900,
                **layout_scenes,
            )

        else:  # both
            fig = make_subplots(
                rows=2, cols=3,
                specs=[[{'type': 'scatter3d'}] * 3, [{'type': 'scatter3d'}] * 3],
                subplot_titles=metric_names,
                horizontal_spacing=0.05,
                vertical_spacing=0.08,
            )
            x_vals = [r["enter_term_sys1"] for r in rows_data]
            y_vals = [r["enter_term_sys2"] for r in rows_data]
            z_vals_leave = [r["leave_term"] for r in rows_data]

            for idx, (metric, mname) in enumerate(zip(metrics, metric_names)):
                row = idx // 3 + 1
                col = idx % 3 + 1
                c_vals = [r[metric] for r in rows_data]

                normal_idx  = [i for i, c in enumerate(categories) if c == "normal"]
                valid_idx   = [i for i, c in enumerate(categories) if c == "valid"]
                plateau_idx = [i for i, c in enumerate(categories) if c == "plateau"]
                best_idx    = [i for i, c in enumerate(categories) if c == "best"]

                fig.add_trace(go.Scatter3d(
                    x=[x_vals[i] for i in normal_idx],
                    y=[y_vals[i] for i in normal_idx],
                    z=[z_vals_leave[i] for i in normal_idx],
                    mode="markers",
                    marker=dict(size=3, color=[c_vals[i] for i in normal_idx],
                                colorscale="Viridis", showscale=False),
                    hovertemplate=f"sys1=%{{x}}<br>sys2=%{{y}}<br>leave=%{{z}}<br>{metric}=%{{customdata:.4f}}<extra></extra>",
                    customdata=[c_vals[i] for i in normal_idx],
                    name="一般", showlegend=(idx == 0),
                ), row=row, col=col)

                for cat_idx, (cat_color, cat_name, cat_size) in zip(
                    [valid_idx, plateau_idx, best_idx],
                    [("yellow","寬鬆條件",5), ("orange","高原",6), ("red","最佳高原",8)]
                ):
                    if cat_idx:
                        fig.add_trace(go.Scatter3d(
                            x=[x_vals[i] for i in cat_idx],
                            y=[y_vals[i] for i in cat_idx],
                            z=[z_vals_leave[i] for i in cat_idx],
                            mode="markers",
                            marker=dict(size=cat_size, color=cat_color, symbol="diamond"),
                            hovertemplate=f"sys1=%{{x}}<br>sys2=%{{y}}<br>leave=%{{z}}<br>{metric}=%{{customdata:.4f}}<extra></extra>",
                            customdata=[c_vals[i] for i in cat_idx],
                            name=cat_name, showlegend=(idx == 0),
                        ), row=row, col=col)

            scene_kwargs = dict(xaxis_title="sys1", yaxis_title="sys2", zaxis_title="leave_term")
            layout_scenes = {skey: scene_kwargs for skey in ["scene","scene2","scene3","scene4","scene5"]}
            fig.update_layout(
                title="Parameter Plateau（both）",
                height=900,
                **layout_scenes,
            )

        fig.write_html("plateau_plot.html")
        fig.show()

    # =========================================
    # 7. 組裝 plateaus_list
    # =========================================
    param_header  = tuple(all_params)
    plateaus_list = [param_header]
    for i in best_indices:
        plateaus_list.append(plateaus[i])

    return plateaus_list