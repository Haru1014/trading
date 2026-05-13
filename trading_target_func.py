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
def gt(a, b):
    if isinstance(b, list):
        return any(a > x for x in b)
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
        print(f"date: {row['date']}  instant_cumulative_profit: {instant_cumulative_profit} buy: {buy_act} sell: {sell_act} vertical_exit: {vertical_barrier_act} position: {position_value}")
    # 3. 轉回 Polars 並計算累計欄位                                
    result_df = pl.from_dicts(records)
    
    result_df = result_df.with_columns([
        pl.col("buy_action").cum_sum().alias("cumulative_buy_position"),
        pl.col("sell_action").cum_sum().alias("cumulative_sell_position"),
        pl.col("profit").cum_sum().alias("cumulative_profit")
    ])
    print(result_df.columns)
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
import polars as pl

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

    for enter_term in enter_terms:
        result_df = turtle_trading_system(
            daily_df, enter_term, 55, leave_term, vertical_barrier, position_size, fee_rate, mode
        )
        total_profit = result_df["profit"].sum()
        total_profits.append(total_profit)
        print(f"enter_term={enter_term}, total_profit={total_profit:.4f}")

    # 找最佳
    best_idx = int(np.argmax(total_profits))
    best_term = enter_terms[best_idx]
    best_profit = total_profits[best_idx]

    # 畫圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=enter_terms,
        y=total_profits,
        mode="lines+markers",
        line=dict(color="blue"),
        marker=dict(size=6),
        name="累積 Profit"
    ))
    fig.add_trace(go.Scatter(
        x=[best_term],
        y=[best_profit],
        mode="markers",
        marker=dict(size=12, color="red", symbol="star"),
        name=f"最佳 enter_term={best_term}"
    ))
    fig.update_layout(
        title=f"Enter Term Sweep（mode={mode}）",
        xaxis_title="enter_term (sys1)",
        yaxis_title="累積 Profit",
        xaxis=dict(tickmode="linear", tick0=5, dtick=1),
        height=500
    )
    fig.write_html("sweep_enter_term.html")
    fig.show()

    return pl.DataFrame({
        "enter_term": enter_terms,
        "total_profit": total_profits,
    })
def evaluate_all(trade_df: pl.DataFrame, daily_df: pl.DataFrame, fee_rate: float = 0.003):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    daily = daily_df.sort("date")
    closes = daily["close"].to_list()
    dates = daily["date"].to_list()
    n = len(closes)

    # ==================== 通用計算函式 ====================
    def calc_metrics(profits: list, label: str) -> dict:
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
        profits = []
        daily_position = []  # 每天是否持有
        for i in range(n):
            daily_position.append(closes[i])  # 全程持有
        trade_profits = []
        trade_profits.append(-closes[0] * fee_rate)  # 買入手續費
        for i in range(1, n - 1):
            trade_profits.append(closes[i] - closes[i-1])
        trade_profits.append((closes[-1] - closes[-2]) - closes[-1] * fee_rate)  # 最後賣出
        return trade_profits, daily_position

    # ==================== Benchmark 2: 前一天close>成本就賣 ====================
    def bm2():
        trade_profits = []
        # 每天的持倉價值（持有中=close, 空手=0）
        daily_position = [0.0] * n
        i = 0
        while i < n - 1:
            buy_price = closes[i]
            buy_fee = buy_price * fee_rate
            daily_position[i] = closes[i]
            i += 1
            sold = False
            while i < n - 1:
                daily_position[i] = closes[i]
                if closes[i - 1] > buy_price:
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
            pos = position_by_date.get(d, 0)
            daily_profits.append(p)
            daily_positions.append(pos)

        return daily_profits, daily_positions

    # ==================== 執行 ====================
    bm1_profits, bm1_pos = bm1()
    bm2_profits, bm2_pos = bm2()
    bm3_profits, bm3_pos = bm3()
    st_profits,  st_pos  = get_strategy_curves()

    # 對齊長度到 n（daily）
    def pad(lst, length):
        return lst + [0] * (length - len(lst))

    bm1_cum = np.cumsum(pad(bm1_profits, n))
    bm2_cum = np.cumsum(pad(bm2_profits, n))
    bm3_cum = np.cumsum(pad(bm3_profits, n))
    st_cum  = np.cumsum(pad(st_profits,  n))

    bm1_whole = bm1_cum + np.array(bm1_pos)
    bm2_whole = bm2_cum + np.array(pad(bm2_pos, n))
    bm3_whole = bm3_cum + np.array(pad(bm3_pos, n))
    st_whole  = st_cum  + np.array(pad(st_pos,  n))

    # ==================== 指標表 ====================
    rows = [
        calc_metrics(st_profits,  "My Strategy"),
        calc_metrics(bm1_profits, "BM1: Buy & Hold"),
        calc_metrics(bm2_profits, "BM2: Sell when profit"),
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
        (bm2_cum, "BM2: Sell when profit",           "green"),
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
        (bm2_whole, "BM2: Sell when profit",           "green"),
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