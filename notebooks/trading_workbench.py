# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "duckdb>=2.0",
#     "polars",
#     "altair",
#     "pyarrow",
#     "numpy",
#     "pandas",
#     "torch",
#     "einops",
#     "huggingface_hub",
#     "safetensors",
#     "tqdm",
# ]
# ///
"""Trading workbench: 5% spike short (primary) and 5-minute ORB (baseline).

Paper research only. Nothing in this notebook places orders.
Spec: docs/trading/SPEC.md. Engine functions are importable:
    from trading_workbench import build_universe, spike_short_signals, ...
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Trading Workbench")

with app.setup:
    import math
    from datetime import date, datetime, time, timedelta
    from pathlib import Path

    import altair as alt
    alt.theme.enable("dark")
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl

    TAXONOMY = [
        "strong_positive_earnings_shock",
        "positive_guidance_shock",
        "positive_guidance_mixed_quarter",
        "moderate_positive_earnings",
        "mixed_earnings_shock",
        "negative_earnings_shock",
        "regulatory_policy_shock",
        "sector_macro_shock",
        "analyst_action",
        "corporate_transaction",
        "no_identifiable_catalyst",
        "routine_disclosure",
        "ambiguous_secondary",
    ]

    # Every feature is computed from rows strictly before the session
    # (ROWS BETWEEN n PRECEDING AND 1 PRECEDING). No look-ahead.
    UNIVERSE_SQL = """
    WITH daily AS (
        SELECT
            symbol,
            CAST(dt AS DATE) AS trade_date,
            arg_min(open, dt) AS day_open,
            max(high) AS day_high,
            min(low) AS day_low,
            arg_max(close, dt) AS day_close,
            sum(volume) AS day_volume,
            sum(close * volume) AS dollar_volume,
            sum(CASE WHEN CAST(dt AS TIME) < TIME '09:35:00' THEN volume ELSE 0 END) AS orb_volume,
            min(CAST(dt AS TIME)) AS first_bar_time,
            count(*) FILTER (WHERE CAST(dt AS TIME) < TIME '10:00:00') AS bars_pre10
        FROM bars
        GROUP BY symbol, CAST(dt AS DATE)
    ),
    lagged AS (
        SELECT
            *,
            lag(day_close) OVER w AS prev_close,
            CASE WHEN lag(day_close) OVER w IS NULL THEN NULL ELSE greatest(
                day_high - day_low,
                abs(day_high - lag(day_close) OVER w),
                abs(day_low - lag(day_close) OVER w)
            ) END AS true_range
        FROM daily
        WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
    ),
    feat AS (
        SELECT
            *,
            avg(true_range) OVER w14 AS atr14,
            count(true_range) OVER w14 AS n_tr,
            median(dollar_volume) OVER w20 AS addv20,
            count(dollar_volume) OVER w20 AS n_dv,
            avg(orb_volume) OVER w14 AS avg_orb_volume
        FROM lagged
        WINDOW
            w14 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING),
            w20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
    ),
    screened AS (
        SELECT
            *,
            atr14 / prev_close AS atr_pct,
            (day_open - prev_close) / atr14 AS gap_atr,
            CASE WHEN avg_orb_volume > 0 THEN orb_volume / avg_orb_volume ELSE NULL END AS rvol,
            CASE
                WHEN n_tr < 14 OR n_dv < 20 THEN 'history'
                WHEN prev_close < $min_price THEN 'price'
                WHEN atr14 / prev_close < $min_atr_pct THEN 'atr'
                WHEN addv20 < $min_addv THEN 'addv'
                ELSE 'pass'
            END AS screen
        FROM feat
    ),
    ranked AS (
        SELECT
            *,
            CASE WHEN screen = 'pass'
                 THEN row_number() OVER (PARTITION BY trade_date, screen ORDER BY addv20 DESC, symbol)
                 ELSE NULL END AS addv_rank
        FROM screened
    )
    SELECT
        symbol, trade_date, day_open, day_high, day_low, day_close, day_volume,
        dollar_volume, prev_close, atr14, atr_pct, addv20, gap_atr, rvol,
        first_bar_time, bars_pre10, addv_rank,
        CASE WHEN screen = 'pass' AND addv_rank > $cap THEN 'cap'
             WHEN screen = 'pass' THEN 'ok' ELSE screen END AS status,
        (screen = 'pass' AND addv_rank <= $cap) AS in_universe
    FROM ranked
    ORDER BY trade_date, symbol
    """

    TRADE_SCHEMA = {
        "strategy": pl.Utf8,
        "symbol": pl.Utf8,
        "trade_date": pl.Date,
        "direction": pl.Int64,
        "entry_dt": pl.Datetime("us"),
        "entry_px": pl.Float64,
        "stop_px": pl.Float64,
        "exit_dt": pl.Datetime("us"),
        "exit_px": pl.Float64,
        "exit_reason": pl.Utf8,
        "mae_bps": pl.Float64,
    }

    EVENT_SCHEMA = {
        "symbol": pl.Utf8,
        "trade_date": pl.Date,
        "gap_atr": pl.Float64,
        "day_open": pl.Float64,
        "threshold": pl.Float64,
        "reached": pl.Boolean,
        "trigger_dt": pl.Datetime("us"),
        "armed": pl.Boolean,
        "arm_dt": pl.Datetime("us"),
        "limit_px": pl.Float64,
        "filled": pl.Boolean,
        "outcome": pl.Utf8,
    }


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'trading_workbench'
    _nxt = _ORDER[(_ORDER.index(_me) + 1) % len(_ORDER)]
    _mode = "wasm" if _sys.platform == "emscripten" else _os.environ.get("SPIKE_HUB_MODE", "live")
    if _mode == "wasm":
        _hub, _next = "../../index.html", (f"../{_nxt}/" if _nxt in _WASM else f"../../notebooks/{_nxt}.html")
    elif _mode == "static":
        _hub, _next = "https://jeromyjsmith.github.io/opening-spike-lab/", f"https://jeromyjsmith.github.io/opening-spike-lab/notebooks/{_nxt}.html"
    else:
        _hub, _next = "http://127.0.0.1:2750/", f"http://localhost:{_PORTS[_nxt]}/"
    _molab = f"https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/{_me}.py"
    _btn = ("display:inline-block;padding:.45rem .9rem;border:1px solid #2a3631;border-radius:999px;"
            "color:#e5ece8;text-decoration:none;font:500 14px system-ui;background:#161f1c")
    _mo.Html(f'<div style="display:flex;gap:.6rem;justify-content:space-between;align-items:center;flex-wrap:wrap">'
             f'<a href="{_hub}" target="_top" style="{_btn}">← Back to hub</a>'
             f'<span style="color:#9aaba4;font:13px system-ui">Notebook {_ORDER.index(_me) + 1} of {len(_ORDER)}</span>'
             f'<a href="{_molab}" target="_blank" rel="noopener" style="{_btn};border-color:#e3a948;color:#e3a948">Open in molab ↗</a>'
             f'<a href="{_next}" target="_top" style="{_btn};border-color:#5cc2a2;color:#5cc2a2">Next notebook →</a></div>')
    return


@app.function
def make_synthetic_bars(
    n_symbols: int = 60, n_days: int = 60, seed: int = 7, spike_prob: float = 0.06
) -> pl.DataFrame:
    """Seeded SYNTHETIC 1-minute bars. Not market data.

    Random-walk prices with injected opening spikes so every stage of the
    pipeline has something to process. Results on this data say nothing
    about whether a strategy works.
    """
    rng = np.random.default_rng(seed)
    days = []
    d = date(2026, 11, 13)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days = days[::-1]
    minutes = 390
    start = np.datetime64("1970-01-01T09:30:00", "us")
    minute_offsets = np.arange(minutes).astype("timedelta64[m]").astype("timedelta64[us]")
    u_shape = 0.6 + 1.8 * (np.linspace(-1, 1, minutes) ** 2)

    frames = []
    for s in range(n_symbols):
        symbol = f"SYN{s:03d}"
        price = float(np.exp(rng.uniform(np.log(20), np.log(600))))
        daily_vol = float(rng.uniform(0.02, 0.07))
        addv = float(np.exp(rng.normal(np.log(1.5e8), 0.9)))
        sigma = daily_vol * 0.75 / math.sqrt(minutes)
        prev_close = price
        for day in days:
            spike = rng.random() < spike_prob
            gap = rng.uniform(0.3, 0.9) * daily_vol if spike else rng.normal(0, daily_vol * 0.4)
            day_open = prev_close * (1 + gap)
            r = rng.normal(0, sigma, minutes) * np.sqrt(u_shape)
            vol_mult = 1.0
            if spike:
                k = int(rng.integers(8, 26))
                r[:k] += rng.uniform(0.055, 0.09) / k
                total = float(r[:k].sum())
                m = int(rng.integers(30, 61))
                if rng.random() < 0.65:
                    r[k : k + m] -= total * rng.uniform(0.4, 0.9) / m
                else:
                    r[k : k + m] += rng.uniform(0.01, 0.03) / m
                vol_mult = 3.0
            close = day_open * np.exp(np.cumsum(r))
            open_ = np.concatenate([[day_open], close[:-1]])
            wick = np.abs(rng.normal(0, sigma * 0.5, (2, minutes)))
            high = np.maximum(open_, close) * (1 + wick[0])
            low = np.minimum(open_, close) * (1 - wick[1])
            volume = np.round(
                addv / day_open / minutes * u_shape * vol_mult * np.exp(rng.normal(0, 0.4, minutes))
            )
            day_start = start + (np.datetime64(day, "D") - np.datetime64("1970-01-01", "D")).astype("timedelta64[us]")
            frames.append(
                pl.DataFrame(
                    {
                        "symbol": [symbol] * minutes,
                        "dt": (day_start + minute_offsets),
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )
            )
            prev_close = float(close[-1])
    return pl.concat(frames).with_columns(pl.col("dt").cast(pl.Datetime("us")))


@app.function
def build_universe(
    bars: pl.DataFrame,
    min_price: float = 50.0,
    min_atr_pct: float = 0.03,
    min_addv: float = 100e6,
    cap: int = 400,
) -> pl.DataFrame:
    """One row per symbol-day with prior-day-only features and a screen status.

    status: ok | history | price | atr | addv | cap. Spec FR-U1 to FR-U7.
    """
    con = duckdb.connect(database=":memory:")
    try:
        con.register("bars", bars.select(["symbol", "dt", "open", "high", "low", "close", "volume"]).to_arrow())
        out = con.sql(
            UNIVERSE_SQL,
            params={
                "min_price": float(min_price),
                "min_atr_pct": float(min_atr_pct),
                "min_addv": float(min_addv),
                "cap": int(cap),
            },
        ).pl()
    finally:
        con.close()
    return out


@app.function
def spike_day(
    times, dts, o, h, l, c,
    threshold: float,
    entry_cutoff: time,
    exit_time: time,
    arm_pullback: float,
    fill_window: int,
    stop_pct: float,
    target_pct: float,
) -> dict:
    """State machine for one symbol-day of the 5% spike short.

    trigger: first bar before entry_cutoff whose high reaches threshold.
    arm:     PLACEHOLDER. First bar whose close is arm_pullback below the
             running high since the trigger.
    entry:   protected sell limit at the arm bar close. Fills only if a later
             bar (within fill_window, before entry_cutoff) trades at or above
             it. Never chases lower.
    exit:    PLACEHOLDER. Stop above entry, target below entry, or time exit.
             If stop and target are both touched in one bar, the stop wins.
    """
    n = len(o)
    res = {
        "reached": False, "trigger_dt": None, "armed": False, "arm_dt": None,
        "limit_px": None, "filled": False, "outcome": "no_trigger", "trade": None,
    }
    trig = None
    for i in range(n):
        if times[i] >= entry_cutoff:
            break
        if h[i] >= threshold:
            trig = i
            break
    if trig is None:
        return res
    res.update(reached=True, trigger_dt=dts[trig], outcome="no_arm")

    arm = None
    run_high = h[trig]
    for j in range(trig, n):
        if times[j] >= entry_cutoff:
            break
        run_high = max(run_high, h[j])
        if c[j] <= run_high * (1.0 - arm_pullback):
            arm = j
            break
    if arm is None:
        return res
    limit = c[arm]
    res.update(armed=True, arm_dt=dts[arm], limit_px=limit, outcome="missed_fill")

    fill = None
    entry = None
    for k in range(arm + 1, min(n, arm + 1 + fill_window)):
        if times[k] >= entry_cutoff:
            break
        if h[k] >= limit:
            fill = k
            entry = max(limit, o[k])
            break
    if fill is None:
        return res

    stop = entry * (1.0 + stop_pct)
    target = entry * (1.0 - target_pct)
    worst = entry
    exit_px = c[n - 1]
    exit_dt = dts[n - 1]
    reason = "EOD"
    for m in range(fill, n):
        if m > fill and times[m] >= exit_time:
            exit_px, exit_dt, reason = o[m], dts[m], "TIME"
            break
        if h[m] >= stop:
            exit_px = stop if (m == fill or o[m] <= stop) else o[m]
            exit_dt, reason = dts[m], "STOP"
            worst = max(worst, exit_px)
            break
        if m > fill and l[m] <= target:
            exit_px = target if o[m] >= target else o[m]
            exit_dt, reason = dts[m], "TARGET"
            worst = max(worst, h[m])
            break
        worst = max(worst, h[m])
    res.update(
        filled=True,
        outcome="filled",
        trade={
            "strategy": "spike_short",
            "direction": -1,
            "entry_dt": dts[fill],
            "entry_px": float(entry),
            "stop_px": float(stop),
            "exit_dt": exit_dt,
            "exit_px": float(exit_px),
            "exit_reason": reason,
            "mae_bps": float((worst - entry) / entry * 1e4),
        },
    )
    return res


@app.function
def spike_short_signals(
    bars: pl.DataFrame,
    daily: pl.DataFrame,
    gap_min: float = 0.25,
    gap_max: float = 1.00,
    spike_pct: float = 0.05,
    entry_cutoff: time = time(10, 0),
    exit_time: time = time(10, 30),
    arm_pullback: float = 0.005,
    fill_window: int = 5,
    stop_pct: float = 0.02,
    target_pct: float = 0.02,
    excluded: pl.DataFrame | None = None,
):
    """Returns (trades, events). events has one row per gap-eligible symbol-day.

    excluded: optional frame of (symbol, trade_date) red-flag names to drop
    before the gap filter (spec FR-C4).
    """
    cand = daily.filter(
        pl.col("in_universe") & pl.col("gap_atr").is_between(gap_min, gap_max, closed="both")
    ).select(["symbol", "trade_date", "gap_atr", "day_open"])
    if excluded is not None and excluded.height > 0:
        cand = cand.join(excluded.select(["symbol", "trade_date"]), on=["symbol", "trade_date"], how="anti")
    cand = cand.with_columns((pl.col("day_open") * (1.0 + spike_pct)).alias("threshold"))
    if cand.height == 0:
        return pl.DataFrame(schema=TRADE_SCHEMA), pl.DataFrame(schema=EVENT_SCHEMA)

    day_bars = (
        bars.with_columns(pl.col("dt").dt.date().alias("trade_date"))
        .join(cand, on=["symbol", "trade_date"], how="inner")
        .sort(["symbol", "trade_date", "dt"])
    )
    trades, events = [], []
    for (sym, tdate), g in day_bars.partition_by(["symbol", "trade_date"], as_dict=True, maintain_order=True).items():
        dts = g["dt"].to_list()
        res = spike_day(
            [x.time() for x in dts], dts,
            g["open"].to_list(), g["high"].to_list(), g["low"].to_list(), g["close"].to_list(),
            threshold=g["threshold"][0],
            entry_cutoff=entry_cutoff, exit_time=exit_time, arm_pullback=arm_pullback,
            fill_window=fill_window, stop_pct=stop_pct, target_pct=target_pct,
        )
        events.append(
            {
                "symbol": sym, "trade_date": tdate, "gap_atr": g["gap_atr"][0],
                "day_open": g["day_open"][0], "threshold": g["threshold"][0],
                "reached": res["reached"], "trigger_dt": res["trigger_dt"],
                "armed": res["armed"], "arm_dt": res["arm_dt"], "limit_px": res["limit_px"],
                "filled": res["filled"], "outcome": res["outcome"],
            }
        )
        if res["trade"] is not None:
            trades.append({"symbol": sym, "trade_date": tdate, **res["trade"]})
    return (
        pl.DataFrame(trades, schema=TRADE_SCHEMA).sort(["trade_date", "symbol"]),
        pl.DataFrame(events, schema=EVENT_SCHEMA).sort(["trade_date", "symbol"]),
    )


@app.function
def orb_signals(
    bars: pl.DataFrame,
    daily: pl.DataFrame,
    top_n: int = 20,
    min_rvol: float = 0.0,
    tp_mult: float = 10.0,
) -> pl.DataFrame:
    """Baseline: 5-minute opening-range breakout, long and short. Needs 1-minute bars.

    Direction from the 09:30-09:34 range (close vs open). Entry at the 09:35
    open, stop at the far side of the range, target tp_mult x range, flat at
    the 15:59 open. If stop and target are both touched in one bar, the stop wins.
    """
    slots = (
        daily.filter(pl.col("in_universe") & pl.col("rvol").is_not_null() & (pl.col("rvol") >= min_rvol))
        .with_columns(pl.col("rvol").rank("ordinal", descending=True).over("trade_date").alias("rvol_rank"))
        .filter(pl.col("rvol_rank") <= top_n)
        .select(["symbol", "trade_date", "day_close"])
    )
    if slots.height == 0:
        return pl.DataFrame(schema=TRADE_SCHEMA)
    b = (
        bars.with_columns(pl.col("dt").dt.date().alias("trade_date"), pl.col("dt").dt.time().alias("bar_time"))
        .join(slots, on=["symbol", "trade_date"], how="inner")
    )
    keys = ["symbol", "trade_date"]
    rng = (
        b.filter(pl.col("bar_time") < time(9, 35))
        .group_by(keys)
        .agg(
            pl.col("open").sort_by("dt").first().alias("r_open"),
            pl.col("close").sort_by("dt").last().alias("r_close"),
            pl.col("high").max().alias("r_high"),
            pl.col("low").min().alias("r_low"),
        )
        .with_columns(
            (pl.col("r_high") - pl.col("r_low")).alias("r_range"),
            pl.when(pl.col("r_close") > pl.col("r_open")).then(1)
            .when(pl.col("r_close") < pl.col("r_open")).then(-1).otherwise(0).alias("direction"),
        )
        .filter((pl.col("direction") != 0) & (pl.col("r_range") > 0))
    )
    entry = b.filter(pl.col("bar_time") == time(9, 35)).select(
        keys + [pl.col("open").alias("entry_px"), pl.col("dt").alias("entry_dt")]
    )
    eod = b.filter(pl.col("bar_time") == time(15, 59)).select(
        keys + [pl.col("open").alias("eod_px"), pl.col("dt").alias("eod_dt")]
    )
    setups = (
        rng.join(entry, on=keys, how="inner").join(eod, on=keys, how="inner")
        .with_columns(
            pl.when(pl.col("direction") == 1).then(pl.col("r_low")).otherwise(pl.col("r_high")).alias("stop_px"),
            (pl.col("entry_px") + pl.col("direction") * pl.col("r_range") * tp_mult).alias("tp_px"),
        )
        # entry already through the stop: no trade
        .filter(pl.col("direction") * (pl.col("entry_px") - pl.col("stop_px")) > 0)
    )
    hits = (
        b.filter((pl.col("bar_time") >= time(9, 35)) & (pl.col("bar_time") < time(15, 59)))
        .join(setups.select(keys + ["direction", "stop_px", "tp_px"]), on=keys, how="inner")
        .with_columns(
            pl.when(pl.col("direction") == 1).then(pl.col("low") <= pl.col("stop_px"))
            .otherwise(pl.col("high") >= pl.col("stop_px")).alias("hit_stop"),
            pl.when(pl.col("direction") == 1).then(pl.col("high") >= pl.col("tp_px"))
            .otherwise(pl.col("low") <= pl.col("tp_px")).alias("hit_tp"),
        )
        .filter(pl.col("hit_stop") | pl.col("hit_tp"))
        .group_by(keys)
        .agg(
            pl.col("dt").sort_by("dt").first().alias("hit_dt"),
            pl.col("hit_stop").sort_by("dt").first().alias("first_stop"),
            pl.col("open").sort_by("dt").first().alias("hit_open"),
        )
    )
    out = (
        setups.join(hits, on=keys, how="left")
        .with_columns(
            pl.when(pl.col("first_stop").is_null()).then(pl.lit("EOD"))
            .when(pl.col("first_stop")).then(pl.lit("STOP")).otherwise(pl.lit("TARGET")).alias("exit_reason"),
        )
        .with_columns(
            pl.when(pl.col("exit_reason") == "EOD").then(pl.col("eod_px"))
            .when((pl.col("exit_reason") == "STOP") & (pl.col("direction") == 1))
            .then(pl.min_horizontal("stop_px", "hit_open"))
            .when(pl.col("exit_reason") == "STOP").then(pl.max_horizontal("stop_px", "hit_open"))
            .otherwise(pl.col("tp_px")).alias("exit_px"),
            pl.when(pl.col("exit_reason") == "EOD").then(pl.col("eod_dt")).otherwise(pl.col("hit_dt")).alias("exit_dt"),
            pl.lit("orb_baseline").alias("strategy"),
            pl.lit(None, dtype=pl.Float64).alias("mae_bps"),
        )
        .with_columns(pl.col("direction").cast(pl.Int64))
        .select(list(TRADE_SCHEMA.keys()))
        .sort(["trade_date", "symbol"])
    )
    return out.cast(TRADE_SCHEMA)


@app.function
def simulate_portfolio(
    trades: pl.DataFrame,
    calendar: list,
    initial_cash: float = 25_000.0,
    risk_pct: float = 0.01,
    max_slots: int = 20,
    max_leverage: float = 4.0,
    max_bps: int = 5,
    fee_per_share: float = 0.005,
    min_fee: float = 1.0,
):
    """Daily-compounding portfolio with risk sizing, per-share fees and a slippage sweep.

    Slippage is adverse on both sides. Short borrow cost is NOT modeled.
    Returns (summary, equity, trade_log).
    """
    by_day: dict = {}
    for row in trades.iter_rows(named=True):
        by_day.setdefault(row["trade_date"], []).append(row)

    summaries, equities, logs = [], [], []
    for bps in range(0, int(max_bps) + 1):
        slip = bps / 10_000.0
        equity = float(initial_cash)
        peak, max_dd = equity, 0.0
        day_rets, log = [], []
        for day in calendar:
            start_equity = equity
            day_pnl = 0.0
            slot_bp = start_equity * max_leverage / max_slots
            risk_budget = start_equity * risk_pct
            for t in by_day.get(day, []):
                d = t["direction"]
                fill_in = t["entry_px"] * (1.0 + d * slip)
                fill_out = t["exit_px"] * (1.0 - d * slip)
                stop_dist = abs(t["entry_px"] - t["stop_px"])
                if stop_dist <= 0 or fill_in <= 0:
                    continue
                shares = max(0, min(int(risk_budget // stop_dist), int(slot_bp // fill_in)))
                if shares == 0:
                    continue
                fees = 2 * max(min_fee, shares * fee_per_share)
                net = d * (fill_out - fill_in) * shares - fees
                day_pnl += net
                log.append(
                    {
                        **t,
                        "slippage_bps": bps,
                        "shares": shares,
                        "fees": fees,
                        "net_pnl": net,
                        "ret_bps": d * (fill_out - fill_in) / fill_in * 1e4,
                    }
                )
            equity = start_equity + day_pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak * 100.0 if peak > 0 else 0.0)
            day_rets.append(day_pnl / start_equity if start_equity > 0 else 0.0)
            equities.append({"trade_date": day, "equity": equity, "slippage_bps": bps})
        rets = np.array(day_rets) if day_rets else np.zeros(1)
        sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
        pnl = np.array([x["net_pnl"] for x in log])
        rb = np.array([x["ret_bps"] for x in log])
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
        summaries.append(
            {
                "slippage_bps": bps,
                "trades": len(log),
                "end_equity": round(equity, 2),
                "net_return_pct": round((equity / initial_cash - 1.0) * 100.0, 3),
                "avg_trade_bps": round(float(rb.mean()), 2) if len(rb) else None,
                "worst_trade_bps": round(float(rb.min()), 2) if len(rb) else None,
                "win_rate_pct": round(float((pnl > 0).mean() * 100.0), 2) if len(pnl) else None,
                "profit_factor": round(float(wins / losses), 3) if losses > 0 else None,
                "sharpe": round(float(rets.mean() / sd * math.sqrt(252)), 2) if sd > 0 else None,
                "max_drawdown_pct": round(max_dd, 3),
                "total_fees": round(float(sum(x["fees"] for x in log)), 2),
            }
        )
        logs.extend(log)
    log_schema = {
        **TRADE_SCHEMA, "slippage_bps": pl.Int64, "shares": pl.Int64,
        "fees": pl.Float64, "net_pnl": pl.Float64, "ret_bps": pl.Float64,
    }
    return (
        pl.DataFrame(summaries),
        pl.DataFrame(equities, schema={"trade_date": pl.Date, "equity": pl.Float64, "slippage_bps": pl.Int64}),
        pl.DataFrame(logs, schema=log_schema),
    )


@app.function
def group_stats(log: pl.DataFrame, by: str) -> pl.DataFrame:
    """Trade count, average return, win rate, profit factor and tails per group."""
    return (
        log.group_by(by)
        .agg(
            pl.len().alias("trades"),
            pl.col("ret_bps").mean().round(1).alias("avg_bps"),
            ((pl.col("ret_bps") > 0).mean() * 100).round(1).alias("win_rate_pct"),
            (
                pl.col("ret_bps").filter(pl.col("ret_bps") > 0).sum()
                / -pl.col("ret_bps").filter(pl.col("ret_bps") <= 0).sum()
            ).round(2).alias("profit_factor"),
            pl.col("ret_bps").min().round(1).alias("worst_bps"),
            pl.col("mae_bps").mean().round(1).alias("avg_mae_bps"),
        )
        .sort("trades", descending=True)
    )


@app.function
def kronos_forecast(
    hist: pl.DataFrame,
    future_ts: list,
    model_id: str = "NeoQuasar/Kronos-small",
    n_paths: int = 20,
    temperature: float = 1.0,
    repo_dir: str = "./Kronos",
) -> dict:
    """Sample n_paths Kronos close paths for one symbol. Uses the GPU when present.

    hist: columns timestamps, open, high, low, close, volume (oldest first).
    Kronos training data ends June 2024; only evaluate on later dates.
    The model runs in float32. The upstream predictor builds float32 inputs,
    so a bfloat16 model would mismatch them.
    """
    import subprocess
    import sys

    repo = Path(repo_dir)
    if not (repo / "model").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/shiyu-coder/Kronos.git", str(repo)],
            check=True,
        )
    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))
    import pandas as pd
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    mini = "mini" in model_id
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(
        "NeoQuasar/Kronos-Tokenizer-2k" if mini else "NeoQuasar/Kronos-Tokenizer-base"
    )
    model = Kronos.from_pretrained(model_id)
    model.eval()
    max_context = 2048 if mini else 512
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)

    h = hist.tail(max_context).to_pandas()
    x_df = h[["open", "high", "low", "close", "volume"]]
    x_ts = pd.Series(pd.to_datetime(h["timestamps"]))
    y_ts = pd.Series(pd.to_datetime(pd.Series(future_ts)))
    with torch.inference_mode():
        preds = predictor.predict_batch(
            df_list=[x_df] * int(n_paths),
            x_timestamp_list=[x_ts] * int(n_paths),
            y_timestamp_list=[y_ts] * int(n_paths),
            pred_len=len(y_ts),
            T=float(temperature),
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
    paths = np.vstack([p["close"].to_numpy() for p in preds])
    last_close = float(h["close"].iloc[-1])
    terminal = (paths[:, -1] - last_close) / last_close * 100.0
    p10, p50, p90 = (np.percentile(paths, q, axis=0) for q in (10, 50, 90))
    return {
        "device": torch.cuda.get_device_name(0) if device != "cpu" else "CPU",
        "paths": paths,
        "p10": p10, "p50": p50, "p90": p90,
        "last_close": last_close,
        "prob_down_pct": float((terminal < 0).mean() * 100.0),
        "median_move_pct": float(np.median(terminal)),
        "spread_pct": float((p90[-1] - p10[-1]) / last_close * 100.0),
    }


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Trading workbench

    **Primary:** 5% spike short (Chris's frozen program). **Baseline:** 5-minute opening-range breakout.
    Paper research only. Nothing here places orders. Spec: `docs/trading/SPEC.md`.
    """)
    return


@app.cell(hide_code=True)
def _():
    data_source = mo.ui.radio(
        options=["Synthetic demo bars", "Parquet (path, glob or https URL)"],
        value="Synthetic demo bars",
        label="Bar data",
    )
    parquet_path = mo.ui.text(value="data/minutes/*.parquet", label="Parquet source", full_width=True)
    syn_symbols = mo.ui.slider(20, 200, step=20, value=60, label="Synthetic symbols")
    syn_days = mo.ui.slider(30, 120, step=10, value=60, label="Synthetic days")

    min_price = mo.ui.number(1, 1000, step=1, value=50, label="Min prior close ($)")
    min_atr_pct = mo.ui.slider(0.0, 8.0, step=0.5, value=3.0, label="Min ATR14 (% of price)")
    min_addv = mo.ui.number(0, 5000, step=10, value=100, label="Min median 20d ADDV ($M)")
    cap = mo.ui.number(10, 2000, step=10, value=400, label="Max universe size")

    gap_range = mo.ui.range_slider(0.0, 3.0, step=0.05, value=[0.25, 1.0], label="Opening gap (ATR)")
    spike_pct = mo.ui.slider(1.0, 10.0, step=0.5, value=5.0, label="Spike from open (%)")
    arm_pullback = mo.ui.slider(0.1, 3.0, step=0.1, value=0.5, label="Arm pullback from high (%) [placeholder]")
    fill_window = mo.ui.slider(1, 30, step=1, value=5, label="Limit fill window (bars)")
    stop_pct = mo.ui.slider(0.5, 6.0, step=0.25, value=2.0, label="Stop above entry (%) [placeholder]")
    target_pct = mo.ui.slider(0.5, 6.0, step=0.25, value=2.0, label="Target below entry (%) [placeholder]")
    exit_time = mo.ui.dropdown(
        options=["10:00", "10:30", "11:00", "12:00", "15:58"], value="10:30", label="Time exit (ET) [placeholder]"
    )

    orb_top_n = mo.ui.slider(5, 50, step=5, value=20, label="ORB: top N by opening RVOL")
    orb_tp = mo.ui.slider(2.0, 15.0, step=1.0, value=10.0, label="ORB: target (x range)")

    cash = mo.ui.number(5000, 1_000_000, step=5000, value=25000, label="Initial cash ($)")
    risk_pct = mo.ui.slider(0.25, 3.0, step=0.25, value=1.0, label="Risk per trade (%)")
    max_slots = mo.ui.slider(1, 50, step=1, value=20, label="Max concurrent slots")
    max_bps = mo.ui.slider(0, 5, step=1, value=5, label="Slippage sweep max (bps)")

    mo.accordion(
        {
            "Data": mo.vstack([data_source, parquet_path, mo.hstack([syn_symbols, syn_days], justify="start")]),
            "Universe (spec 4.1)": mo.hstack([min_price, min_atr_pct, min_addv, cap], justify="start"),
            "Spike short (spec 4.2, 4.4)": mo.vstack(
                [
                    mo.hstack([gap_range, spike_pct, fill_window], justify="start"),
                    mo.hstack([arm_pullback, stop_pct, target_pct, exit_time], justify="start"),
                ]
            ),
            "ORB baseline": mo.hstack([orb_top_n, orb_tp], justify="start"),
            "Portfolio": mo.hstack([cash, risk_pct, max_slots, max_bps], justify="start"),
        },
        multiple=True,
    )
    return (
        arm_pullback,
        cap,
        cash,
        data_source,
        exit_time,
        fill_window,
        gap_range,
        max_bps,
        max_slots,
        min_addv,
        min_atr_pct,
        min_price,
        orb_top_n,
        orb_tp,
        parquet_path,
        risk_pct,
        spike_pct,
        stop_pct,
        syn_days,
        syn_symbols,
        target_pct,
    )


@app.cell(hide_code=True)
def _(data_source, parquet_path, syn_days, syn_symbols):
    is_synthetic = data_source.value.startswith("Synthetic")
    if is_synthetic:
        bars = make_synthetic_bars(n_symbols=syn_symbols.value, n_days=syn_days.value)
        _note = mo.callout(
            mo.md(
                "**Synthetic bars.** Seeded random walks with injected spikes. "
                "Every number below tests the pipeline only. It is not a strategy result."
            ),
            kind="danger",
        )
    else:
        try:
            _con = duckdb.connect(database=":memory:")
            bars = _con.sql(
                """
                SELECT symbol, CAST(date AS TIMESTAMP) AS dt,
                       CAST(open AS DOUBLE) AS open, CAST(high AS DOUBLE) AS high,
                       CAST(low AS DOUBLE) AS low, CAST(close AS DOUBLE) AS close,
                       CAST(volume AS DOUBLE) AS volume
                FROM read_parquet($src)
                WHERE CAST(date AS TIME) >= TIME '09:30:00' AND CAST(date AS TIME) < TIME '16:00:00'
                """,
                params={"src": parquet_path.value},
            ).pl().with_columns(pl.col("dt").cast(pl.Datetime("us")))
            _con.close()
        except Exception as _exc:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        f"Could not read `{parquet_path.value}`: `{_exc}`\n\n"
                        "Expected columns: `symbol, date, open, high, low, close, volume`, "
                        "with `date` in US/Eastern market time."
                    ),
                    kind="warn",
                ),
            )
        _note = mo.callout(mo.md(f"Loaded `{parquet_path.value}`."), kind="info")
    _one = bars.filter(pl.col("symbol") == bars["symbol"][0]).sort("dt")
    bar_seconds = int(_one["dt"].diff().dt.total_seconds().drop_nulls().median() or 60)
    _note
    return bar_seconds, bars, is_synthetic


@app.cell(hide_code=True)
def _(bars, cap, min_addv, min_atr_pct, min_price):
    daily = build_universe(
        bars,
        min_price=min_price.value,
        min_atr_pct=min_atr_pct.value / 100.0,
        min_addv=min_addv.value * 1e6,
        cap=cap.value,
    )
    calendar = daily["trade_date"].unique().sort().to_list()
    return calendar, daily


@app.cell(hide_code=True)
def _(
    arm_pullback,
    bars,
    daily,
    exit_time,
    fill_window,
    gap_range,
    spike_pct,
    stop_pct,
    target_pct,
):
    _hh, _mm = (int(x) for x in exit_time.value.split(":"))
    spike_trades, spike_events = spike_short_signals(
        bars,
        daily,
        gap_min=gap_range.value[0],
        gap_max=gap_range.value[1],
        spike_pct=spike_pct.value / 100.0,
        exit_time=time(_hh, _mm),
        arm_pullback=arm_pullback.value / 100.0,
        fill_window=fill_window.value,
        stop_pct=stop_pct.value / 100.0,
        target_pct=target_pct.value / 100.0,
    )
    return spike_events, spike_trades


@app.cell(hide_code=True)
def _(bar_seconds, bars, daily, orb_top_n, orb_tp):
    if bar_seconds == 60:
        orb_trades = orb_signals(bars, daily, top_n=orb_top_n.value, tp_mult=orb_tp.value)
    else:
        orb_trades = pl.DataFrame(schema=TRADE_SCHEMA)
    return (orb_trades,)


@app.cell(hide_code=True)
def _(
    calendar,
    cash,
    max_bps,
    max_slots,
    orb_trades,
    risk_pct,
    spike_trades,
):
    _kw = dict(
        calendar=calendar,
        initial_cash=float(cash.value),
        risk_pct=risk_pct.value / 100.0,
        max_slots=max_slots.value,
        max_bps=max_bps.value,
    )
    spike_summary, spike_equity, spike_log = simulate_portfolio(spike_trades, **_kw)
    orb_summary, orb_equity, orb_log = simulate_portfolio(orb_trades, **_kw)
    return (
        orb_equity,
        orb_log,
        orb_summary,
        spike_equity,
        spike_log,
        spike_summary,
    )


@app.cell(hide_code=True)
def _(bar_seconds, bars, daily):
    # ---- Tab: universe and coverage ----
    _expected = max(1, 1800 // bar_seconds)
    _uni = daily.filter(pl.col("in_universe"))
    _complete = _uni.filter(
        (pl.col("first_bar_time") == time(9, 30)) & (pl.col("bars_pre10") >= _expected)
    ).height
    _coverage = _complete / _uni.height * 100.0 if _uni.height else 0.0
    _per_day = daily.group_by("trade_date").agg(pl.col("in_universe").sum().alias("universe_size")).sort("trade_date")
    _avg_size = _per_day.filter(pl.col("universe_size") > 0)["universe_size"].mean() or 0

    _size_chart = (
        alt.Chart(_per_day)
        .mark_area(opacity=0.5, line=True)
        .encode(x=alt.X("trade_date:T", title="Session"), y=alt.Y("universe_size:Q", title="Stocks in universe"))
        .properties(title="Universe size by session", height=220, width="container")
    )
    _status = daily.group_by("status").agg(pl.len().alias("symbol_days"))
    _status_chart = (
        alt.Chart(_status)
        .mark_bar()
        .encode(
            x=alt.X("symbol_days:Q", title="Symbol-days"),
            y=alt.Y("status:N", sort="-x", title="Screen result"),
            tooltip=["status", "symbol_days"],
        )
        .properties(title="Why symbol-days were kept or dropped", height=220, width="container")
    )
    _last = daily.filter((pl.col("trade_date") == daily["trade_date"].max()) & (pl.col("status") != "history")).select(
        ["symbol", "prev_close", "atr_pct", "addv20", "status"]
    )
    _scatter = (
        alt.Chart(_last)
        .mark_circle(size=60)
        .encode(
            x=alt.X("addv20:Q", scale=alt.Scale(type="log"), title="Median 20d ADDV ($, log)"),
            y=alt.Y("atr_pct:Q", axis=alt.Axis(format="%"), title="ATR14 / price"),
            color=alt.Color("status:N", title="Screen"),
            tooltip=["symbol", alt.Tooltip("prev_close:Q", format="$.2f"),
                     alt.Tooltip("atr_pct:Q", format=".1%"), alt.Tooltip("addv20:Q", format="$,.0f"), "status"],
        )
        .properties(title="Latest session: liquidity vs volatility", height=300, width="container")
    )
    _missing = _uni.filter(
        (pl.col("first_bar_time") != time(9, 30)) | (pl.col("bars_pre10") < _expected)
    ).select(["trade_date", "symbol", "first_bar_time", "bars_pre10"])

    uni_size = mo.ui.altair_chart(_size_chart, chart_selection="interval")
    uni_status = mo.ui.altair_chart(_status_chart, chart_selection="point")
    uni_scatter = mo.ui.altair_chart(_scatter, chart_selection="interval")
    universe_view = mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{bars.height:,}", label=f"Bars ({bar_seconds}s)", bordered=True),
                    mo.stat(f"{bars['symbol'].n_unique():,}", label="Symbols", bordered=True),
                    mo.stat(f"{daily['trade_date'].n_unique():,}", label="Sessions", bordered=True),
                    mo.stat(f"{_avg_size:.0f}", label="Avg universe per session", bordered=True),
                    mo.stat(f"{_coverage:.1f}%", label="Open-window coverage (target 98%)", bordered=True),
                ],
                justify="start",
            ),
            mo.md("**Select to drill in:** drag across the size chart to pick sessions, click a bar to pick a screen "
                  "result, or drag a box on the scatter. The table under the charts follows your selection."),
            mo.hstack([uni_size, uni_status], widths="equal"),
            uni_scatter,
            mo.md(f"**Universe symbol-days with an incomplete 09:30 to 10:00 window:** {_missing.height}"),
            mo.ui.table(_missing, selection=None, page_size=10) if _missing.height else mo.md(""),
        ]
    )
    return uni_scatter, uni_size, uni_status, universe_view


@app.cell(hide_code=True)
def _(daily, uni_scatter, uni_size, uni_status):
    _rows = daily.filter(pl.col("status") != "history")
    _notes = []
    _sv = uni_status.value
    if _sv is not None and len(_sv):
        _st = list(pl.DataFrame(_sv)["status"].unique())
        _rows = _rows.filter(pl.col("status").is_in(_st))
        _notes.append(f"screen result {', '.join(_st)}")
    _dv = uni_size.value
    if _dv is not None and len(_dv):
        _days = pl.DataFrame(_dv)["trade_date"].cast(pl.Date).unique()
        _rows = _rows.filter(pl.col("trade_date").is_in(_days.implode()))
        _notes.append(f"{_days.len()} sessions")
    _pv = uni_scatter.value
    if _pv is not None and len(_pv):
        _syms = pl.DataFrame(_pv)["symbol"].unique()
        _rows = _rows.filter((pl.col("symbol").is_in(_syms.implode())) & (pl.col("trade_date") == daily["trade_date"].max()))
        _notes.append(f"{_syms.len()} symbols from the scatter")
    _cols = [c for c in ["trade_date", "symbol", "status", "prev_close", "atr_pct", "addv20", "gap_atr", "in_universe"] if c in _rows.columns]
    universe_pick_view = mo.vstack([
        mo.md(f"**{_rows.height:,} symbol-days** match " + ("; ".join(_notes) if _notes else "(no selection yet: showing all screened symbol-days)")),
        mo.ui.table(_rows.select(_cols).sort("trade_date", descending=True).with_columns(
            pl.col(pl.Float64).round(3)), page_size=10),
    ])
    return (universe_pick_view,)


@app.cell(hide_code=True)
def _(daily, spike_equity, spike_events, spike_log, spike_summary):
    # ---- Tab: spike short ----
    _exits = spike_log.filter(pl.col("slippage_bps") == 0)
    _funnel = pl.DataFrame(
        {
            "stage": ["1 Universe symbol-days", "2 Gap 0.25-1.00 ATR", "3 Reached spike", "4 Armed", "5 Filled"],
            "count": [
                int(daily["in_universe"].sum()),
                spike_events.height,
                int(spike_events["reached"].sum()),
                int(spike_events["armed"].sum()),
                int(spike_events["filled"].sum()),
            ],
        }
    )
    _funnel_chart = (
        alt.Chart(_funnel)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", scale=alt.Scale(type="symlog"), title="Count (log scale)"),
            y=alt.Y("stage:N", title=None),
            tooltip=["stage", "count"],
        )
        .properties(title="Signal funnel (spec FR-A7)", height=200, width="container")
    )
    _gaps = (
        daily.filter(pl.col("in_universe") & pl.col("gap_atr").is_between(-3, 3))
        .with_columns(gap_bin=(pl.col("gap_atr") / 0.25).floor() * 0.25)
        .group_by("gap_bin").agg(pl.len().alias("symbol_days")).sort("gap_bin")
    )
    _gap_hist = (
        alt.Chart(_gaps)
        .mark_bar()
        .encode(
            x=alt.X("gap_bin:Q", title="Opening gap (ATR, 0.25 bins)"),
            y=alt.Y("symbol_days:Q", title="Symbol-days"),
            color=alt.condition(
                (alt.datum.gap_bin >= 0.25) & (alt.datum.gap_bin < 1.0), alt.value("#e45756"), alt.value("#bab0ac")
            ),
            tooltip=["gap_bin", "symbol_days"],
        )
        .properties(title="Opening gaps in the universe (eligible band in red)", height=200, width="container")
    )
    _eq = (
        alt.Chart(spike_equity)
        .mark_line()
        .encode(
            x=alt.X("trade_date:T", title="Session"),
            y=alt.Y("equity:Q", scale=alt.Scale(zero=False), title="Equity ($)"),
            color=alt.Color("slippage_bps:O", title="Slippage (bps)", scale=alt.Scale(scheme="viridis")),
            tooltip=["trade_date:T", "slippage_bps:O", alt.Tooltip("equity:Q", format="$,.2f")],
        )
        .properties(title="Equity by slippage tier", height=280, width="container")
    )
    _ret_hist = (
        alt.Chart(_exits.select([c for c in ["symbol", "trade_date", "entry_dt", "ret_bps", "exit_reason"] if c in _exits.columns]))
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=alt.X("entry_dt:T", title="Entry"),
            y=alt.Y("ret_bps:Q", title="Trade return (bps, 0 bps slippage)"),
            color=alt.Color("exit_reason:N", title="Exit"),
            tooltip=["symbol", "trade_date", "exit_reason", alt.Tooltip("ret_bps:Q", format=".0f")],
        )
        .properties(title="Every trade: drag a box to list those trades", height=300, width="container")
    )
    _by_day = (
        spike_events.group_by("trade_date")
        .agg(
            pl.len().alias("gap_eligible"),
            pl.col("reached").sum().alias("reached"),
            pl.col("armed").sum().alias("armed"),
            pl.col("filled").sum().alias("filled"),
        )
        .join(daily.group_by("trade_date").agg(pl.col("in_universe").sum().alias("universe")), on="trade_date")
        .select(["trade_date", "universe", "gap_eligible", "reached", "armed", "filled"])
        .sort("trade_date", descending=True)
    )
    spike_funnel = mo.ui.altair_chart(_funnel_chart, chart_selection="point")
    spike_gaps = mo.ui.altair_chart(_gap_hist, chart_selection="interval")
    spike_eq = mo.ui.altair_chart(_eq, chart_selection="interval")
    spike_trades_ui = mo.ui.altair_chart(_ret_hist, chart_selection="interval") if _exits.height else None
    spike_view = mo.vstack(
        [
            mo.callout(
                mo.md(
                    "Arm, stop, target and time exit are **placeholders**. "
                    "Chris's real entry formula and exits are still open (spec section 9)."
                ),
                kind="warn",
            ),
            mo.md("**Select to drill in:** click a funnel stage, drag across the gap bars or the equity dates, or box trades "
                  "on the scatter. The tables under *Your selection* follow."),
            mo.hstack([spike_funnel, spike_gaps], widths="equal"),
            mo.md("### Slippage sweep (after $0.005/share fees, $1 minimum; borrow cost not modeled)"),
            mo.ui.table(spike_summary, selection=None),
            spike_eq,
            mo.hstack([spike_trades_ui, mo.ui.table(group_stats(_exits, "exit_reason"), selection=None)], widths=[2, 1])
            if _exits.height
            else mo.md("No filled trades with these settings."),
            mo.md("### Daily funnel"),
            mo.ui.table(_by_day, selection=None, page_size=10),
        ]
    )
    return spike_eq, spike_funnel, spike_gaps, spike_trades_ui, spike_view


@app.cell(hide_code=True)
def _(daily, spike_eq, spike_events, spike_funnel, spike_gaps, spike_log, spike_trades_ui):
    _parts = []
    _fv = spike_funnel.value
    if _fv is not None and len(_fv):
        _stage = pl.DataFrame(_fv)["stage"][0]
        _ev = {"2": spike_events, "3": spike_events.filter(pl.col("reached")), "4": spike_events.filter(pl.col("armed")),
               "5": spike_events.filter(pl.col("filled"))}.get(_stage[0])
        if _ev is None:
            _ev = daily.filter(pl.col("in_universe"))
        _parts += [mo.md(f"**Funnel stage {_stage}: {_ev.height:,} rows**"), mo.ui.table(_ev, page_size=8)]
    _gv = spike_gaps.value
    if _gv is not None and len(_gv):
        _bins = pl.DataFrame(_gv)["gap_bin"]
        _g = daily.filter(pl.col("in_universe") & pl.col("gap_atr").is_between(_bins.min(), _bins.max() + 0.25, closed="left"))
        _parts += [mo.md(f"**Gap {_bins.min():.2f} to {_bins.max() + 0.25:.2f} ATR: {_g.height:,} universe symbol-days**"),
                   mo.ui.table(_g.select([c for c in ["trade_date", "symbol", "gap_atr", "prev_close", "atr_pct"] if c in _g.columns]), page_size=8)]
    _z = spike_log.filter(pl.col("slippage_bps") == 0)
    _ev2 = spike_eq.value
    if _ev2 is not None and len(_ev2):
        _days = pl.DataFrame(_ev2)["trade_date"].cast(pl.Date)
        _t = _z.filter(pl.col("trade_date").is_between(_days.min(), _days.max()))
        _parts += [mo.md(f"**Trades {_days.min()} to {_days.max()}: {_t.height}** · avg {(_t['ret_bps'].mean() or 0):+.1f} bps"),
                   mo.ui.table(_t, page_size=8)]
    if spike_trades_ui is not None:
        _tv = spike_trades_ui.value
        if _tv is not None and len(_tv):
            _sel = pl.DataFrame(_tv)
            _t = _z.join(_sel.select("symbol", "trade_date").unique(), on=["symbol", "trade_date"])
            _parts += [mo.md(f"**{_t.height} trades boxed** · win rate {(_t['ret_bps'] > 0).mean() * 100:.0f}% · "
                             f"avg {_t['ret_bps'].mean():+.1f} bps"), mo.ui.table(_t, page_size=8)]
    spike_pick_view = mo.vstack([mo.md("### Your selection")] + (_parts or [mo.md("_Nothing selected yet._")]))
    return (spike_pick_view,)


@app.cell(hide_code=True)
def _(bar_seconds, orb_equity, orb_log, orb_summary, spike_summary):
    # ---- Tab: baseline comparison ----
    if bar_seconds != 60:
        baseline_view = mo.callout(mo.md("The ORB baseline needs 1-minute bars."), kind="warn")
        orb_eq_ui = None
    else:
        _cmp = pl.concat(
            [
                spike_summary.select(["slippage_bps", "avg_trade_bps", "trades"]).with_columns(
                    pl.lit("spike short").alias("strategy")
                ),
                orb_summary.select(["slippage_bps", "avg_trade_bps", "trades"]).with_columns(
                    pl.lit("ORB baseline").alias("strategy")
                ),
            ]
        ).drop_nulls("avg_trade_bps")
        _cmp_chart = (
            alt.Chart(_cmp)
            .mark_line(point=True)
            .encode(
                x=alt.X("slippage_bps:O", title="Slippage per side (bps)"),
                y=alt.Y("avg_trade_bps:Q", title="Average trade return (bps)"),
                color=alt.Color("strategy:N", title=None),
                tooltip=["strategy", "slippage_bps", "avg_trade_bps", "trades"],
            )
            .properties(title="How fast slippage eats each edge", height=260, width="container")
        )
        _zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
        _eq_b = (
            alt.Chart(orb_equity)
            .mark_line()
            .encode(
                x=alt.X("trade_date:T", title="Session"),
                y=alt.Y("equity:Q", scale=alt.Scale(zero=False), title="Equity ($)"),
                color=alt.Color("slippage_bps:O", title="Slippage (bps)", scale=alt.Scale(scheme="viridis")),
            )
            .properties(title="ORB baseline equity by slippage tier", height=260, width="container")
        )
        _o0 = orb_log.filter(pl.col("slippage_bps") == 0)
        orb_eq_ui = mo.ui.altair_chart(_eq_b, chart_selection="interval")
        baseline_view = mo.vstack(
            [
                mo.ui.altair_chart(_cmp_chart, chart_selection="point"),
                mo.md("### ORB baseline slippage sweep"),
                mo.ui.table(orb_summary, selection=None),
                mo.md("Drag across the equity chart to list the ORB trades in those sessions."),
                orb_eq_ui,
                mo.ui.table(group_stats(_o0, "exit_reason"), selection=None) if _o0.height else mo.md(""),
            ]
        )
    return baseline_view, orb_eq_ui


@app.cell(hide_code=True)
def _(orb_eq_ui, orb_log):
    _v = None if orb_eq_ui is None else orb_eq_ui.value
    if _v is not None and len(_v):
        _days = pl.DataFrame(_v)["trade_date"].cast(pl.Date)
        _t = orb_log.filter((pl.col("slippage_bps") == 0) & pl.col("trade_date").is_between(_days.min(), _days.max()))
        orb_pick_view = mo.vstack([mo.md(f"**ORB trades {_days.min()} to {_days.max()}: {_t.height}**"), mo.ui.table(_t, page_size=8)])
    else:
        orb_pick_view = mo.md("_Drag across the ORB equity chart to list trades._")
    return (orb_pick_view,)


@app.cell(hide_code=True)
def _():
    labels_path = mo.ui.text(value="data/catalyst_labels.csv", label="Catalyst labels CSV", full_width=True)
    return (labels_path,)


@app.cell(hide_code=True)
def _(labels_path, spike_log):
    # ---- Tab: catalysts ----
    _reference = pl.DataFrame(
        {
            "category": [
                "Earnings event (5y, SEC-verified)", "No SEC earnings event (5y)",
                "Moderate positive earnings", "Positive guidance shock", "Mixed earnings",
                "Strong positive earnings shock", "Positive guidance / mixed quarter",
            ],
            "trades": [164, 122, 13, 5, 7, 5, 1],
            "avg_bps": [34.5, 7.5, 62.7, 25.4, -2.8, -109.5, -265.5],
            "win_rate_pct": [67.1, 56.6, 76.9, 80.0, 57.1, 40.0, 0.0],
            "profit_factor": [2.02, 1.24, 3.18, 3.47, 0.94, 0.28, 0.00],
        }
    )
    _ref_chart = (
        alt.Chart(_reference)
        .mark_bar()
        .encode(
            x=alt.X("avg_bps:Q", title="Average short return (bps)"),
            y=alt.Y("category:N", sort=None, title=None),
            color=alt.condition(alt.datum.avg_bps >= 0, alt.value("#4c78a8"), alt.value("#e45756")),
            tooltip=["category", "trades", "avg_bps", "win_rate_pct", "profit_factor"],
        )
        .properties(
            title="Reported by Chris, Nov 13 2026 email (not recomputed here)", height=240, width="container"
        )
    )
    _p = Path(labels_path.value)
    if _p.exists():
        _labels = pl.read_csv(_p, try_parse_dates=True)
        _bad = _labels.filter(~pl.col("category").is_in(TAXONOMY))
        _joined = spike_log.filter(pl.col("slippage_bps") == 0).join(
            _labels.select(["symbol", "trade_date", "category"]), on=["symbol", "trade_date"], how="left"
        ).with_columns(pl.col("category").fill_null("unlabeled"))
        _mine = mo.vstack(
            [
                mo.md(f"Labels loaded: {_labels.height} rows, {_bad.height} with a category outside the taxonomy."),
                mo.ui.table(group_stats(_joined, "category"), selection=None),
            ]
        )
    else:
        _mine = mo.callout(
            mo.md(
                f"No labels file at `{labels_path.value}`. Columns needed: "
                "`symbol, trade_date, category, status` (status is red, yellow or green). "
                "Once it exists, this tab reports fills, returns and tails by category."
            ),
            kind="info",
        )
    catalyst_view = mo.vstack(
        [
            mo.ui.altair_chart(_ref_chart, chart_selection="point"),
            mo.md("### This backtest, by category"),
            labels_path,
            _mine,
            mo.md("### Taxonomy v1 (frozen for evaluation)\n" + "\n".join(f"{i + 1}. `{c}`" for i, c in enumerate(TAXONOMY))),
        ]
    )
    return (catalyst_view,)


@app.cell(hide_code=True)
def _(spike_events):
    _slots = spike_events.filter(pl.col("reached")).sort(["trade_date", "symbol"], descending=[True, False])
    slot_labels = [
        f"{r['trade_date']} | {r['symbol']} | {r['outcome']}" for r in _slots.head(300).iter_rows(named=True)
    ]
    slot = mo.ui.dropdown(
        options=slot_labels or ["none"], value=(slot_labels[0] if slot_labels else "none"),
        label="Symbol-day that reached the spike threshold",
    )
    return (slot,)


@app.cell(hide_code=True)
def _(bars, slot, spike_events, spike_trades):
    # ---- Tab: trade review ----
    review_chart, review_bars = None, None
    if slot.value == "none":
        review_view = mo.md("No symbol-day reached the spike threshold with these settings.")
    else:
        _d, _sym, _ = [x.strip() for x in slot.value.split("|")]
        _day = date.fromisoformat(_d)
        _ev = spike_events.filter((pl.col("symbol") == _sym) & (pl.col("trade_date") == _day)).row(0, named=True)
        _b = bars.filter(
            (pl.col("symbol") == _sym) & (pl.col("dt").dt.date() == _day) & (pl.col("dt").dt.time() < time(11, 30))
        ).with_columns((pl.col("close") >= pl.col("open")).alias("up"))
        _x = alt.X("dt:T", title="Time (ET)")
        _color = alt.condition(alt.datum.up, alt.value("#26a69a"), alt.value("#ef5350"))
        _wicks = alt.Chart(_b).mark_rule().encode(
            x=_x, y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="Price"), y2="high:Q", color=_color
        )
        _bodies = alt.Chart(_b).mark_bar(size=4).encode(
            x=_x, y="open:Q", y2="close:Q", color=_color,
            tooltip=["dt:T", "open", "high", "low", "close", "volume"],
        )
        _levels = pl.DataFrame(
            {"level": ["open", "+spike threshold"], "price": [_ev["day_open"], _ev["threshold"]]}
        )
        _rules = alt.Chart(_levels).mark_rule(strokeDash=[5, 4]).encode(
            y="price:Q", color=alt.Color("level:N", title=None, scale=alt.Scale(range=["#888", "#f58518"]))
        )
        _marks = [{"event": "trigger", "dt": _ev["trigger_dt"], "price": _ev["threshold"]}]
        if _ev["armed"]:
            _marks.append({"event": "arm / limit", "dt": _ev["arm_dt"], "price": _ev["limit_px"]})
        _t = spike_trades.filter((pl.col("symbol") == _sym) & (pl.col("trade_date") == _day))
        if _t.height:
            _tr = _t.row(0, named=True)
            _marks += [
                {"event": "short entry", "dt": _tr["entry_dt"], "price": _tr["entry_px"]},
                {"event": f"exit ({_tr['exit_reason']})", "dt": _tr["exit_dt"], "price": _tr["exit_px"]},
            ]
        _pts = alt.Chart(pl.DataFrame(_marks)).mark_point(size=140, filled=True, shape="diamond").encode(
            x="dt:T", y="price:Q", color=alt.Color("event:N", title="Event"), tooltip=["event", "dt:T", "price"]
        )
        _chart = (_wicks + _bodies + _rules + _pts).resolve_scale(color="independent").properties(
            title=f"{_sym} on {_day}: gap {_ev['gap_atr']:.2f} ATR, outcome {_ev['outcome']}",
            height=380, width="container",
        )
        review_chart = mo.ui.altair_chart(_chart, chart_selection="interval")
        review_bars = _b
        review_view = mo.vstack([mo.md("Drag across the candles to list those bars below."), review_chart,
                                 mo.ui.table(_t, selection=None) if _t.height else mo.md("Not filled.")])
    return review_bars, review_chart, review_view


@app.cell(hide_code=True)
def _(review_bars, review_chart):
    review_pick_view = mo.md("")
    if review_chart is not None:
        _sel = review_chart.apply_selection(review_bars)
        if _sel is not None and len(_sel) and len(_sel) < review_bars.height:
            _sel = pl.DataFrame(_sel)
            _chg = (_sel["close"][-1] / _sel["open"][0] - 1) * 1e4 if _sel.height else 0
            review_pick_view = mo.vstack([mo.md(f"**{_sel.height} bars selected** · open to close {_chg:+.0f} bps · "
                                                f"volume {_sel['volume'].sum():,.0f}"), mo.ui.table(_sel.drop("up"), page_size=10)])
    return (review_pick_view,)


@app.cell(hide_code=True)
def _():
    k_model = mo.ui.dropdown(
        options=["NeoQuasar/Kronos-small", "NeoQuasar/Kronos-base", "NeoQuasar/Kronos-mini"],
        value="NeoQuasar/Kronos-small", label="Checkpoint",
    )
    k_lookback = mo.ui.slider(64, 512, step=32, value=256, label="Context bars (max 512)")
    k_horizon = mo.ui.slider(10, 120, step=10, value=30, label="Forecast bars")
    k_paths = mo.ui.slider(5, 100, step=5, value=20, label="Sampled paths")
    k_temp = mo.ui.slider(0.5, 1.5, step=0.1, value=1.0, label="Temperature")
    k_run = mo.ui.run_button(label="Run Kronos on the selected symbol-day")
    return k_horizon, k_lookback, k_model, k_paths, k_run, k_temp


@app.cell(hide_code=True)
def _(
    bars,
    is_synthetic,
    k_horizon,
    k_lookback,
    k_model,
    k_paths,
    k_run,
    k_temp,
    slot,
    spike_events,
):
    # ---- Tab: Kronos inspector ----
    _out = mo.md("Pick a symbol-day and run. The forecast starts at the bar where the spike threshold was reached.")
    if k_run.value and slot.value != "none":
        _d, _sym, _ = [x.strip() for x in slot.value.split("|")]
        _day = date.fromisoformat(_d)
        _ev = spike_events.filter((pl.col("symbol") == _sym) & (pl.col("trade_date") == _day)).row(0, named=True)
        _sb = bars.filter(pl.col("symbol") == _sym).sort("dt")
        _hist = _sb.filter(pl.col("dt") <= _ev["trigger_dt"]).tail(k_lookback.value).rename({"dt": "timestamps"})
        _fut = _sb.filter((pl.col("dt") > _ev["trigger_dt"]) & (pl.col("dt").dt.date() == _day)).head(k_horizon.value)
        try:
            with mo.status.spinner(title="Running Kronos"):
                _k = kronos_forecast(
                    _hist, _fut["dt"].to_list(), model_id=k_model.value,
                    n_paths=k_paths.value, temperature=k_temp.value,
                )
            _fan = pl.DataFrame(
                {"dt": _fut["dt"], "p10": _k["p10"], "median": _k["p50"], "p90": _k["p90"], "actual": _fut["close"]}
            )
            _band = alt.Chart(_fan).mark_area(opacity=0.25).encode(
                x=alt.X("dt:T", title="Time (ET)"),
                y=alt.Y("p10:Q", scale=alt.Scale(zero=False), title="Close"), y2="p90:Q",
            )
            _med = alt.Chart(_fan).mark_line().encode(x="dt:T", y="median:Q")
            _act = alt.Chart(_fan).mark_line(color="#e45756", strokeDash=[4, 4]).encode(x="dt:T", y="actual:Q")
            _flag = (
                "HIGH_DISPERSION_CAUTION"
                if _k["spread_pct"] > 5.0 or 40.0 <= _k["prob_down_pct"] <= 60.0
                else "MODERATE_CONVICTION"
            )
            _payload = {
                "symbol": _sym,
                "decision_timestamp": str(_ev["trigger_dt"]),
                "device": _k["device"],
                "last_close": round(_k["last_close"], 2),
                "prob_down_pct": round(_k["prob_down_pct"], 1),
                "median_projected_move_pct": round(_k["median_move_pct"], 2),
                "p10_p90_spread_pct": round(_k["spread_pct"], 2),
                "agent_guardrail_status": _flag,
                "agent_instruction": (
                    "Treat this forecast as a risk input, not a signal. Judge whether it is credible before "
                    "using it. Do not enter or skip a trade on the median move alone. Use the P10-P90 spread "
                    "only to scale size down."
                ),
            }
            _out = mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.stat(f"{_k['prob_down_pct']:.0f}%", label="Paths ending lower", bordered=True),
                            mo.stat(f"{_k['median_move_pct']:+.2f}%", label="Median projected move", bordered=True),
                            mo.stat(f"{_k['spread_pct']:.2f}%", label="P10 to P90 spread", bordered=True),
                            mo.stat(_k["device"], label="Compute", bordered=True),
                        ],
                        justify="start",
                    ),
                    (_band + _med + _act).properties(
                        title=f"Kronos paths for {_sym} on {_day} (band P10 to P90, dashed red = actual)",
                        height=340, width="container",
                    ),
                    mo.md("### Agent tool payload"),
                    mo.json(_payload),
                ]
            )
        except Exception as _exc:
            _out = mo.callout(mo.md(f"Kronos did not run: `{type(_exc).__name__}: {_exc}`"), kind="warn")
    kronos_view = mo.vstack(
        [
            mo.callout(
                mo.md("Bars are synthetic, so a forecast here only proves the model loads and runs."), kind="danger"
            ) if is_synthetic else mo.md(""),
            mo.hstack([k_model, k_lookback, k_horizon, k_paths, k_temp], justify="start"),
            k_run,
            _out,
        ]
    )
    return (kronos_view,)


@app.cell(hide_code=True)
def _(
    baseline_view,
    catalyst_view,
    kronos_view,
    orb_pick_view,
    review_pick_view,
    review_view,
    slot,
    spike_pick_view,
    spike_view,
    universe_pick_view,
    universe_view,
):
    mo.ui.tabs(
        {
            "1 Universe and coverage": mo.vstack([universe_view, universe_pick_view]),
            "2 Spike short": mo.vstack([spike_view, spike_pick_view]),
            "3 Baseline (ORB)": mo.vstack([baseline_view, orb_pick_view]),
            "4 Catalysts": catalyst_view,
            "5 Trade review": mo.vstack([slot, review_view, review_pick_view]),
            "6 Kronos inspector": mo.vstack([slot, kronos_view]),
        }
    )
    return


if __name__ == "__main__":
    app.run()
