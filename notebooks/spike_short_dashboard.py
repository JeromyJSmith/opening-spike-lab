# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "duckdb>=2.0.1",
#     "polars>=2.1.0",
#     "pyarrow",
#     "numpy",
#     "altair>=6.3.0",
#     "wigglystuff>=0.6.0",
#     "lib-pybroker>=2.0.3",
#     "opentelemetry-sdk>=1.45",
#     "httpx",
#     "sqlglot",
#     "scikit-learn",
# ]
# ///
"""Spike-short decision dashboard: traces, tail losers, filter ablations.

Paper research only. Nothing here places orders.
Run:  uv run marimo edit notebooks/spike_short_dashboard.py
Engine: src/spike_lab (pure per-bar state machine in rules.py).
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Spike Short Lab")

with app.setup:
    import sys
    from dataclasses import replace
    from datetime import date, time, timedelta
    from pathlib import Path

    import altair as alt
    alt.theme.enable("dark")
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))

    from spike_lab import engine, gate, metrics, synthetic, tracing
    from spike_lab import alpaca
    from spike_lab import pybroker_check as pbc
    from spike_lab.metrics import LOSER_BPS
    from spike_lab.rules import Params
    from spike_lab.universe import build_universe, screen_funnel
    from wigglystuff import FormulaDAG, TangleLatex, TraceView

    ALPACA_DIR = ROOT / "data" / "alpaca"
    GREEN, RED, BLUE, GREY, AMBER = "#2f9e6e", "#d64545", "#3b6fd6", "#8a8f98", "#d69a2d"
    PERIOD_COLORS = alt.Scale(domain=["in-sample", "out-of-sample", "holdout"], range=[BLUE, AMBER, GREY])


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'spike_short_dashboard'
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Spike Short Lab
    **Opening spike short**: liquid US stocks ($\ge$ \$50, ATR $\ge$ 3%, ADDV $\ge$ \$100M in two tiers) that open
    $0.25$–$1.0$ ATR above the prior close and run $+4$–$5\%$ by 10:00 ET (8:00 MT). Short the bar after the first red
    1-minute bar, cover 20 minutes later, 5 bps slippage. The open problem: one or two **continuation losers** a day
    ($\approx -3\%$) erase eight to ten $+30$ bps winners. This dashboard traces every decision stage and tests exits
    and filters aimed at that tail. Targets: win rate $60$–$70\%$, profit factor $\ge 1.5$.
    """)
    return


@app.cell(hide_code=True)
def _():
    pull_btn = mo.ui.run_button(label="Pull Alpaca SIP bars (read-only, a few minutes)")
    pull_start = mo.ui.date(value=date(2025, 5, 1), label="From")
    pull_end = mo.ui.date(value=date.today() - timedelta(days=1), label="To")
    _keys = alpaca.keys_present()
    mo.callout(
        mo.vstack([
            mo.md(f"**Alpaca keys:** {'found in `.env`' if _keys else f'not set. Paste `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` into `{ROOT / '.env'}` and rerun this cell.'}"),
            mo.hstack([pull_start, pull_end, pull_btn], justify="start", gap=2) if _keys else mo.md(""),
        ]),
        kind="success" if _keys else "warn",
    )
    return pull_btn, pull_end, pull_start


@app.cell(hide_code=True)
async def _(pull_btn, pull_end, pull_start):
    alpaca_pulled = False
    if pull_btn.value:
        with mo.status.spinner(title="Pulling daily bars, screening, then 1-minute mornings for gap days"):
            with mo.redirect_stdout():
                await alpaca.pull(pull_start.value, pull_end.value, min_price=25.0)
        alpaca_pulled = True
    return (alpaca_pulled,)


@app.cell(hide_code=True)
def _(alpaca_pulled):
    has_alpaca = (ALPACA_DIR / "intraday.parquet").exists() and (ALPACA_DIR / "daily.parquet").exists()
    source = mo.ui.dropdown(
        options=["Alpaca SIP (data/alpaca)", "Synthetic demo"] if has_alpaca else ["Synthetic demo"],
        value="Alpaca SIP (data/alpaca)" if has_alpaca else "Synthetic demo",
        label="Data",
    )
    tiers = mo.ui.multiselect(options=["100M-1B", "1B+"], value=["100M-1B", "1B+"], label="ADDV tiers")
    spike = mo.ui.dropdown(options={"4%": 0.04, "5%": 0.05}, value="5%", label="Run-up threshold")
    min_price = mo.ui.dropdown(options={"$25 (memo)": 25.0, "$50 (spec)": 50.0}, value="$50 (spec)", label="Min price")
    hold = mo.ui.slider(5, 40, value=20, step=1, label="Cover after (min)", show_value=True)
    slip = mo.ui.number(0, 30, value=5, step=0.5, label="Slippage bps (round trip)")
    arm_window = mo.ui.slider(3, 30, value=15, step=1, label="Arm window (min)", show_value=True)
    split = mo.ui.radio(
        options=["Tune on prior 12 months, validate on last quarter (decision stack)",
                 "Tune on last quarter, validate on prior 12 months (Chris's memo)"],
        value="Tune on prior 12 months, validate on last quarter (decision stack)", label="Sample split", inline=True,
    )
    mo.vstack(
        [
            mo.hstack([source, tiers, spike, min_price], justify="start", gap=2),
            mo.hstack([hold, arm_window, slip], justify="start", gap=2),
            mo.hstack([split], justify="start"),
        ]
    )
    return arm_window, hold, min_price, slip, source, spike, split, tiers


@app.cell(hide_code=True)
def _(split):
    _q, _y = (date(2026, 7, 1), date(2026, 9, 30)), (date(2025, 7, 1), date(2026, 6, 30))
    _tune, _val = (_y, _q) if split.value.startswith("Tune on prior") else (_q, _y)
    is_range = mo.ui.date_range(start=date(2016, 1, 1), stop=date(2026, 12, 31), value=_tune, label="Tune (in-sample)")
    oos_range = mo.ui.date_range(start=date(2016, 1, 1), stop=date(2026, 12, 31), value=_val, label="Validate (out-of-sample, untouched)")
    ho_range = mo.ui.date_range(start=date(2016, 1, 1), stop=date(2026, 12, 31), value=(date(2026, 10, 1), date(2026, 11, 30)), label="Forward holdout")
    mo.hstack([is_range, oos_range, ho_range], justify="start", gap=2)
    return ho_range, is_range, oos_range


@app.cell(hide_code=True)
def _(min_price, source):
    with mo.status.spinner(title="Loading bars and building the universe in DuckDB"):
        if source.value.startswith("Alpaca"):
            daily = pl.read_parquet(ALPACA_DIR / "daily.parquet")
            intraday = pl.read_parquet(ALPACA_DIR / "intraday.parquet")
            is_synthetic = False
        else:
            daily, intraday, _truth = synthetic.make_synthetic()
            is_synthetic = True
        db = duckdb.connect()
        universe = build_universe(daily, min_price=min_price.value, con=db)
    return daily, intraday, is_synthetic, universe


@app.cell(hide_code=True)
def _(daily, intraday, is_synthetic):
    _span = f"{daily['trade_date'].min()} to {daily['trade_date'].max()}"
    if is_synthetic:
        _banner = mo.callout(
            mo.md(
                f"**SYNTHETIC DATA. Not market prices.** Seeded random paths with planted spikes ({_span}). "
                "Every number below only exercises the pipeline and charts; none of it says whether the strategy works. "
                "Paste the Alpaca keys into `.env`, then press **Pull Alpaca SIP bars** above (or run `uv run python -m spike_lab.alpaca`)."
            ),
            kind="danger",
        )
    else:
        _prov = (ALPACA_DIR / "PROVENANCE.md").read_text() if (ALPACA_DIR / "PROVENANCE.md").exists() else ""
        _banner = mo.callout(
            mo.md(f"**Alpaca SIP bars**, {_span}: {daily['symbol'].n_unique():,} symbols daily, "
                  f"{intraday.height:,} one-minute morning bars. Survivorship and borrow caveats in PROVENANCE.md.\n\n{_prov}"),
            kind="info",
        )
    _banner
    return


@app.cell(hide_code=True)
def _(arm_window, hold, slip, spike):
    base = Params(spike_pct=spike.value, hold_minutes=hold.value, slippage_bps=slip.value, arm_window_min=arm_window.value)
    return (base,)


@app.cell(hide_code=True)
def _(base, ho_range, intraday, is_range, oos_range, tiers, universe):
    periods = {
        "in-sample": is_range.value,
        "out-of-sample": oos_range.value,
        "holdout": ho_range.value,
    }
    with mo.status.spinner(title="Running the baseline through the state machine"):
        runs = {k: engine.run(universe, intraday, base, a, b, tiers=tuple(tiers.value)) for k, (a, b) in periods.items()}
    trades_all = pl.concat(
        [r.trades.with_columns(pl.lit(k).alias("period")) for k, r in runs.items() if r.trades.height]
    ) if any(r.trades.height for r in runs.values()) else pl.DataFrame()
    outcomes_all = pl.concat([r.outcomes.with_columns(pl.lit(k).alias("period")) for k, r in runs.items() if r.outcomes.height])
    trades_is = runs["in-sample"].trades
    return periods, runs, trades_all, trades_is


@app.cell(hide_code=True)
def _(runs):
    def _kpis(name, s):
        if not s.get("trades"):
            return mo.md(f"**{name}**: no trades")
        _ok = metrics.meets_targets(s)
        return mo.vstack(
            [
                mo.md(f"### {name} {'✅' if _ok else '❌'}"),
                mo.hstack(
                    [
                        mo.stat(f"{s['win_rate']}%", label="Win rate", caption="target 60-70%", direction="increase" if s["win_rate"] >= 60 else "decrease", bordered=True),
                        mo.stat(f"{s['profit_factor']}", label="Profit factor", caption="target ≥ 1.5", direction="increase" if (s["profit_factor"] or 0) >= 1.5 else "decrease", bordered=True),
                        mo.stat(f"{s['avg_net_bps']} bps", label="Avg net / trade", caption=f"median {s['median_net_bps']}", bordered=True),
                        mo.stat(f"{s['trades_per_day']}", label="Trades / day", caption=f"{s['trades']} over {s['days']} days", bordered=True),
                        mo.stat(f"{s['tail_losers']}", label=f"Tail losers ≤ {LOSER_BPS:.0f} bps", caption=f"{s['tail_share_of_losses_pct']}% of all losses", direction="decrease", target_direction="decrease", bordered=True),
                        mo.stat(f"{s['max_dd_bps']:.0f}", label="Max DD (bps, sum)", caption=f"{s['pct_days_positive']}% days green", bordered=True),
                    ],
                    widths="equal",
                ),
            ]
        )

    summaries = {k: metrics.summary(r.trades) for k, r in runs.items()}
    _names = {"in-sample": "Tune (in-sample)", "out-of-sample": "Validate (out-of-sample)", "holdout": "Forward holdout"}
    mo.vstack([_kpis(f"{_names[k]} · {runs[k].trades['trade_date'].min()} to {runs[k].trades['trade_date'].max()}" if runs[k].trades.height else _names[k], s) for k, s in summaries.items()])
    return (summaries,)


@app.cell(hide_code=True)
def _(math_view, overview, overview_pick, tail_pick, tail_view, trade_charts, trade_pick, universe_view, winloss):
    mo.ui.tabs(
        {
            "Overview": mo.vstack([overview, overview_pick]),
            "Trades": mo.vstack([trade_charts, trade_pick]),
            "Continuation-loser tail": mo.vstack([tail_view, tail_pick]),
            "Math": mo.vstack([math_view, winloss]),
            "Universe": universe_view,
        }
    )
    return


@app.cell(hide_code=True)
def _(trades_all):
    # ---------- Overview charts
    if trades_all.height == 0:
        overview = mo.md("No trades.")
        overview_daily = None
    else:
        _eq = (
            trades_all.group_by(["period", "trade_date"]).agg(pl.col("net_bps").sum().alias("day_bps"), pl.len().alias("n"))
            .sort("trade_date")
            .with_columns(pl.col("day_bps").cum_sum().over("period").alias("cum_bps"))
            .with_columns((pl.col("cum_bps") - pl.col("cum_bps").cum_max().over("period")).alias("dd_bps"))
        )
        _base = alt.Chart(_eq).encode(x=alt.X("trade_date:T", title=None))
        _equity = _base.mark_line(strokeWidth=2).encode(
            y=alt.Y("cum_bps:Q", title="Cumulative net bps (1 unit / trade)"),
            color=alt.Color("period:N", scale=PERIOD_COLORS, legend=alt.Legend(orient="top")),
            tooltip=["trade_date:T", "period:N", alt.Tooltip("cum_bps:Q", format=",.0f"), "n:Q"],
        ).properties(height=260, width=1100, title="Equity curve")
        _dd = _base.mark_area(opacity=0.5).encode(
            y=alt.Y("dd_bps:Q", title="Drawdown bps"), color=alt.Color("period:N", scale=PERIOD_COLORS, legend=None)
        ).properties(height=110, width=1100)
        _daily = _base.mark_bar().encode(
            y=alt.Y("day_bps:Q", title="Daily net bps"),
            color=alt.condition("datum.day_bps > 0", alt.value(GREEN), alt.value(RED)),
            tooltip=["trade_date:T", alt.Tooltip("day_bps:Q", format=",.0f"), "n:Q"],
        ).properties(height=150, title="Daily P&L (sum of trades)")
        # the "one loser wipes the day" view: day P&L split into winners vs tail losers
        _split = (
            trades_all.with_columns(
                pl.when(pl.col("net_bps") <= LOSER_BPS).then(pl.lit("tail losers"))
                .when(pl.col("net_bps") > 0).then(pl.lit("winners")).otherwise(pl.lit("small losers")).alias("bucket")
            )
            .group_by(["period", "bucket"]).agg(pl.col("net_bps").sum().alias("bps"), pl.len().alias("n"))
        )
        _contrib = alt.Chart(_split).mark_bar().encode(
            x=alt.X("bps:Q", title="Total net bps"), y=alt.Y("bucket:N", title=None, sort=["winners", "small losers", "tail losers"]),
            color=alt.Color("bucket:N", scale=alt.Scale(domain=["winners", "small losers", "tail losers"], range=[GREEN, AMBER, RED]), legend=None),
            row=alt.Row("period:N", title=None), tooltip=["period", "bucket", "n", alt.Tooltip("bps:Q", format=",.0f")],
        ).properties(height=60, width=380, title="Where the P&L goes")
        overview_daily = mo.ui.altair_chart(_daily.properties(width="container", height=200,
                                            title="Daily P&L: drag across days to list their trades"), chart_selection="interval")
        overview = mo.vstack([mo.ui.altair_chart(alt.vconcat(_equity, _dd).resolve_scale(x="shared")), overview_daily, _contrib])
    return overview, overview_daily


@app.cell(hide_code=True)
def _(overview_daily, trades_all):
    overview_pick = mo.md("")
    if trades_all.height and overview_daily is not None and overview_daily.value is not None and len(overview_daily.value):
        _days = pl.DataFrame(overview_daily.value)["trade_date"].cast(pl.Date)
        _t = trades_all.filter(pl.col("trade_date").is_between(_days.min(), _days.max()))
        _w, _l = _t.filter(pl.col("net_bps") > 0)["net_bps"].sum(), -_t.filter(pl.col("net_bps") <= 0)["net_bps"].sum()
        overview_pick = mo.vstack([
            mo.md(f"**{_days.min()} to {_days.max()}: {_t.height} trades** · win rate {(_t['net_bps'] > 0).mean() * 100:.0f}% · "
                  f"PF {(_w / _l) if _l else float('nan'):.2f} · tail losers {int((_t['net_bps'] <= LOSER_BPS).sum())}"),
            mo.ui.table(_t.select([c for c in ["period", "symbol", "trade_date", "tier", "net_bps", "mae_bps", "mfe_bps", "exit_reason"] if c in _t.columns]).sort("net_bps"), page_size=10),
        ])
    return (overview_pick,)


@app.cell(hide_code=True)
def _(trades_all):
    # ---------- Trade-level charts
    _t = trades_all.with_columns(
        pl.when(pl.col("net_bps") <= LOSER_BPS).then(pl.lit("tail loser")).when(pl.col("net_bps") > 0).then(pl.lit("winner")).otherwise(pl.lit("loser")).alias("result"),
        (pl.col("entry_ts").dt.hour().cast(pl.Utf8) + ":" + (pl.col("entry_ts").dt.minute() // 5 * 5).cast(pl.Utf8).str.zfill(2)).alias("entry_bucket"),
    )
    _res_scale = alt.Scale(domain=["winner", "loser", "tail loser"], range=[GREEN, AMBER, RED])
    _tv = _t.select("period", "symbol", "trade_date", "tier", "net_bps", "mae_bps", "mfe_bps", "exit_reason", "result", "entry_bucket")
    _hb = (
        _tv.with_columns((pl.col("net_bps") / 25).floor().mul(25).alias("bin"))
        .group_by(["period", "result", "bin"]).len()
    )
    _hist = alt.Chart(_hb.group_by(["result", "bin"]).agg(pl.col("len").sum())).mark_bar().encode(
        x=alt.X("bin:Q", title="Net bps per trade (25 bps bins, all periods)"),
        y=alt.Y("len:Q", title="Trades", stack=True), color=alt.Color("result:N", scale=_res_scale, legend=alt.Legend(orient="top")),
        tooltip=["result", "bin", "len"],
    ).properties(height=200, width="container", title="Trade P&L distribution: drag across bins to list those trades")
    # scatter: every tail loser plus a sample of the rest, to keep the page light
    _tv_s = pl.concat([
        _tv.filter(pl.col("result") == "tail loser"),
        _tv.filter(pl.col("result") != "tail loser").sample(n=min(2500, _tv.filter(pl.col("result") != "tail loser").height), seed=1),
    ])
    _mae = alt.Chart(_tv_s).mark_circle(size=22, opacity=0.55).encode(
        x=alt.X("mae_bps:Q", title="MAE bps (worst move against the short)", scale=alt.Scale(type="symlog")),
        y=alt.Y("mfe_bps:Q", title="MFE bps (best move for the short)", scale=alt.Scale(type="symlog")),
        color=alt.Color("result:N", scale=_res_scale),
        tooltip=["symbol", "trade_date:T", "tier", alt.Tooltip("net_bps:Q", format=".0f"), alt.Tooltip("mae_bps:Q", format=".0f"), alt.Tooltip("mfe_bps:Q", format=".0f"), "exit_reason"],
    ).properties(height=340, width="container", title="MAE vs MFE: drag a box (e.g. the red cloud) to list those trades")
    _wr = (
        _t.group_by(["period", "tier"]).agg(((pl.col("net_bps") > 0).mean() * 100).alias("win_rate"), pl.len().alias("n"),
                                         (pl.col("net_bps").filter(pl.col("net_bps") > 0).sum() / -pl.col("net_bps").filter(pl.col("net_bps") <= 0).sum()).alias("pf"))
    )
    _wr_chart = alt.Chart(_wr).mark_bar().encode(
        x=alt.X("tier:N", title=None), y=alt.Y("win_rate:Q", title="Win rate %", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("period:N", scale=PERIOD_COLORS), xOffset="period:N",
        tooltip=["period", "tier", "n", alt.Tooltip("win_rate:Q", format=".1f"), alt.Tooltip("pf:Q", format=".2f")],
    ).properties(height=240, width="container", title="Win rate by tier (click a bar; target 60%)")
    _rule = alt.Chart(pl.DataFrame({"y": [60.0]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(y="y:Q")
    _time = alt.Chart(_t.group_by("entry_bucket").agg(pl.col("net_bps").mean().alias("avg"), pl.len().alias("n"), ((pl.col("net_bps") <= LOSER_BPS).mean() * 100).alias("tail_pct"))).mark_bar().encode(
        x=alt.X("entry_bucket:N", title="Entry time (ET, 5-min bucket)"), y=alt.Y("avg:Q", title="Avg net bps"),
        color=alt.condition("datum.avg > 0", alt.value(GREEN), alt.value(RED)), tooltip=["entry_bucket", "n", alt.Tooltip("avg:Q", format=".1f"), alt.Tooltip("tail_pct:Q", format=".1f")],
    ).properties(height=240, width="container", title="Avg trade by entry time (click a bucket)")
    # minute-by-minute path fan: median and 10-90% band, winners vs tail losers
    _p = (
        _t.select("result", "path_bps").with_row_index("tid").explode("path_bps")
        .with_columns(pl.col("tid").cum_count().over("tid").alias("minute"))
        .group_by(["result", "minute"]).agg(
            pl.col("path_bps").median().alias("med"), pl.col("path_bps").quantile(0.1).alias("q10"), pl.col("path_bps").quantile(0.9).alias("q90"), pl.len().alias("n"))
        .filter(pl.col("n") >= 5).sort("minute")
    )
    _pb = alt.Chart(_p).encode(x=alt.X("minute:Q", title="Minutes after entry"), color=alt.Color("result:N", scale=_res_scale, legend=None))
    _fan = (_pb.mark_area(opacity=0.18).encode(y="q10:Q", y2="q90:Q") + _pb.mark_line(strokeWidth=2).encode(y=alt.Y("med:Q", title="Short P&L at close, bps"))).properties(
        height=300, width=420, title="Trade path after entry (median, 10-90% band)")
    trade_hist = mo.ui.altair_chart(_hist, chart_selection="interval")
    trade_mae = mo.ui.altair_chart(_mae, chart_selection="interval")
    trade_tier = mo.ui.altair_chart(_wr_chart, chart_selection="point")
    trade_time = mo.ui.altair_chart(_time, chart_selection="point")
    trade_charts = mo.vstack([
        mo.md("**Every chart here is a filter.** Drag on the histogram or the scatter, click a tier or a time bucket; "
              "the trades that match are listed under *Your selection*, with their win rate and profit factor."),
        trade_hist,
        mo.hstack([trade_mae, _fan.properties(width="container")], widths="equal"),
        mo.hstack([trade_tier, trade_time], widths="equal"),
    ])
    return trade_charts, trade_hist, trade_mae, trade_tier, trade_time


@app.cell(hide_code=True)
def _(trade_hist, trade_mae, trade_tier, trade_time, trades_all):
    _t = trades_all.with_columns(
        (pl.col("entry_ts").dt.hour().cast(pl.Utf8) + ":" + (pl.col("entry_ts").dt.minute() // 5 * 5).cast(pl.Utf8).str.zfill(2)).alias("entry_bucket"))
    _notes = []
    _v = trade_hist.value
    if _v is not None and len(_v):
        _b = pl.DataFrame(_v)["bin"]
        _t = _t.filter(pl.col("net_bps").is_between(_b.min(), _b.max() + 25, closed="left"))
        _notes.append(f"net {_b.min():.0f} to {_b.max() + 25:.0f} bps")
    _v = trade_mae.value
    if _v is not None and len(_v):
        _k = pl.DataFrame(_v).select("symbol", "trade_date").unique()
        _t = _t.join(_k.with_columns(pl.col("trade_date").cast(pl.Date)), on=["symbol", "trade_date"])
        _notes.append(f"{_k.height} trades boxed on MAE/MFE")
    _v = trade_tier.value
    if _v is not None and len(_v):
        _k = pl.DataFrame(_v).select("period", "tier").unique()
        _t = _t.join(_k, on=["period", "tier"])
        _notes.append("tier " + ", ".join(f"{a}/{b}" for a, b in _k.iter_rows()))
    _v = trade_time.value
    if _v is not None and len(_v):
        _bk = pl.DataFrame(_v)["entry_bucket"]
        _t = _t.filter(pl.col("entry_bucket").is_in(_bk.implode()))
        _notes.append("entry " + ", ".join(_bk.to_list()))
    if not _notes:
        trade_pick = mo.md("### Your selection\n_Nothing selected yet._")
    else:
        _w, _l = _t.filter(pl.col("net_bps") > 0)["net_bps"].sum(), -_t.filter(pl.col("net_bps") <= 0)["net_bps"].sum()
        trade_pick = mo.vstack([
            mo.md(f"### Your selection\n{'; '.join(_notes)}  \n**{_t.height} trades** · win rate "
                  f"{(_t['net_bps'] > 0).mean() * 100 if _t.height else 0:.0f}% · PF {(_w / _l) if _l else float('nan'):.2f} · "
                  f"avg {(_t['net_bps'].mean() or 0):+.1f} bps · tail losers {int((_t['net_bps'] <= LOSER_BPS).sum())}"),
            mo.ui.table(_t.select([c for c in ["period", "symbol", "trade_date", "tier", "net_bps", "mae_bps", "mfe_bps", "exit_reason"] + ["entry_bucket"] if c in _t.columns]).sort("net_bps"), page_size=12),
        ])
    return (trade_pick,)


@app.cell(hide_code=True)
def _(trades_is):
    # ---------- Continuation-loser tail anatomy (in-sample only: features are what we would tune on)
    _t = trades_is.with_columns(
        pl.when(pl.col("net_bps") <= LOSER_BPS).then(pl.lit("tail loser")).when(pl.col("net_bps") > 0).then(pl.lit("winner")).otherwise(pl.lit("loser")).alias("result"))
    _feats = {
        "red_vol_ratio": "Red bar volume ÷ run-up avg",
        "close_vs_avwap_bps": "Red close vs anchored VWAP (bps)",
        "minutes_to_trigger": "Minutes open → +threshold",
        "runup_pct_at_arm": "Run-up at arm (%)",
        "entry_ext_atr": "Entry extension above prev close (ATR)",
        "gap_atr": "Opening gap (ATR)",
        "red_body_bps": "Red bar body (bps)",
        "entry_vs_hod_bps": "Entry below HOD (bps)",
    }
    _long = _t.select(["result"] + list(_feats)).unpivot(index="result", variable_name="feature").with_columns(
        pl.col("feature").replace_strict(_feats).alias("feature"))
    _bs = _long.group_by(["feature", "result"]).agg(
        pl.col("value").quantile(0.05).alias("p05"), pl.col("value").quantile(0.25).alias("q1"),
        pl.col("value").median().alias("med"), pl.col("value").quantile(0.75).alias("q3"),
        pl.col("value").quantile(0.95).alias("p95"), pl.len().alias("n"))
    _bx = alt.Chart(_bs).encode(
        x=alt.X("result:N", title=None, sort=["winner", "loser", "tail loser"]),
        color=alt.Color("result:N", scale=alt.Scale(domain=["winner", "loser", "tail loser"], range=[GREEN, AMBER, RED]), legend=None),
        tooltip=["feature", "result", "n", alt.Tooltip("med:Q", format=".2f"), alt.Tooltip("q1:Q", format=".2f"), alt.Tooltip("q3:Q", format=".2f")],
    )
    _box = alt.layer(
        _bx.mark_rule().encode(y=alt.Y("p05:Q", title=None), y2="p95:Q"),
        _bx.mark_bar(size=18, opacity=0.6).encode(y="q1:Q", y2="q3:Q"),
        _bx.mark_tick(size=18, thickness=2, color="#e5ece8").encode(y="med:Q"),
    ).properties(width=170, height=150).facet(facet=alt.Facet("feature:N", title=None), columns=4).resolve_scale(y="independent").properties(
        title="Arm-time features: winners vs tail losers (tune period; box = 25-75%, whisker = 5-95%)")
    # Separation: does early adverse movement flag tail losers before it costs the winners?
    _rows = []
    _adv = _t.with_columns(pl.col("path_bps").list.eval(-pl.element()).alias("adv"))
    for _k in (1, 2, 3, 5, 8):
        _m = _adv.with_columns(pl.col("adv").list.head(_k).list.max().fill_null(0).alias("adv_k"))
        _tail = _m.filter(pl.col("result") == "tail loser")
        _win = _m.filter(pl.col("result") == "winner")
        for _x in range(0, 301, 10):
            _rows.append({
                "minute": f"by minute {_k}", "threshold_bps": _x,
                "tail_caught_pct": float((_tail["adv_k"] >= _x).mean() * 100) if _tail.height else 0.0,
                "winners_cut_pct": float((_win["adv_k"] >= _x).mean() * 100) if _win.height else 0.0,
            })
    _roc = alt.Chart(pl.DataFrame(_rows)).mark_line(point=alt.OverlayMarkDef(size=14)).encode(
        x=alt.X("winners_cut_pct:Q", title="% of winners that would be cut"), y=alt.Y("tail_caught_pct:Q", title="% of tail losers caught"),
        color=alt.Color("minute:N", legend=alt.Legend(orient="right")), tooltip=["minute", "threshold_bps", alt.Tooltip("tail_caught_pct:Q", format=".0f"), alt.Tooltip("winners_cut_pct:Q", format=".0f")],
    ).properties(width="container", height=340, title="Early adverse-close rule: click a point to see what it catches")
    _diag = alt.Chart(pl.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(strokeDash=[4, 4], color=GREY).encode(x="x:Q", y="y:Q")
    _n = _t.group_by("result").len()
    tail_roc = mo.ui.altair_chart(_roc, chart_selection="point")
    tail_view = mo.vstack([
        mo.md(f"Counts in-sample: " + ", ".join(f"**{r}** {n}" for r, n in _n.iter_rows())),
        _box,
        mo.hstack([tail_roc, mo.md(
            r"""
    **Reading the curve.** Each point is a rule *cover if the short is down $\ge x$ bps at any close within the first
    $k$ minutes*. A useful rule sits far above the dashed diagonal: it catches tail losers while cutting few winners.
    A curve hugging the diagonal means early adverse movement does not separate the two groups, which is the
    "stops cut the winners too" problem in one picture.
    """)], justify="start", gap=2),
    ])
    return tail_roc, tail_view


@app.cell(hide_code=True)
def _(tail_roc, trades_is):
    _v = tail_roc.value
    if _v is None or not len(_v):
        tail_pick = mo.md("_Click a point on the curve to list the trades that rule would cover early._")
    else:
        _r = pl.DataFrame(_v).row(0, named=True)
        _k = int(str(_r["minute"]).split()[-1])
        _x = float(_r["threshold_bps"])
        _m = trades_is.with_columns(
            pl.col("path_bps").list.eval(-pl.element()).list.head(_k).list.max().fill_null(0).alias("adverse_bps"),
            pl.when(pl.col("net_bps") <= LOSER_BPS).then(pl.lit("tail loser")).when(pl.col("net_bps") > 0).then(pl.lit("winner")).otherwise(pl.lit("loser")).alias("result"),
        ).filter(pl.col("adverse_bps") >= _x)
        _c = _m.group_by("result").len()
        tail_pick = mo.vstack([
            mo.md(f"**Rule: cover if down ≥ {_x:.0f} bps within {_k} min** → it would cover {_m.height} trades: "
                  + ", ".join(f"{n} {r}" for r, n in _c.iter_rows()) + f". Their actual total: {_m['net_bps'].sum():+,.0f} bps."),
            mo.ui.table(_m.select("symbol", "trade_date", "result", "net_bps", "adverse_bps", "mae_bps").sort("net_bps"), page_size=10),
        ])
    return (tail_pick,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Filter ablations
    Can an exit or entry filter cut the continuation losers without cutting the winners?
    """)
    return


@app.cell(hide_code=True)
def _():
    run_ablation = mo.ui.run_button(label="Run filter ablations (in-sample + out-of-sample)")
    run_sweep = mo.ui.run_button(label="Sweep MAE bailout grid (in-sample)")
    mo.hstack([run_ablation, run_sweep], justify="start")
    return run_ablation, run_sweep


@app.cell(hide_code=True)
def _(base, intraday, periods, run_ablation, tiers, universe):
    mo.stop(not run_ablation.value, mo.md("_Press **Run filter ablations**. Each variant is a full re-run of the state machine._"))
    with mo.status.spinner(title="Running every variant on in-sample and out-of-sample"):
        ablation, ablation_trades = metrics.ablate(
            universe, intraday, metrics.default_variants(base),
            {k: v for k, v in periods.items() if k in ("in-sample", "out-of-sample")}, tiers=tuple(tiers.value),
        )
    return (ablation,)


@app.cell(hide_code=True)
def _(ablation):
    _cols = ["variant", "period", "trades", "win_rate", "profit_factor", "avg_net_bps", "worst_bps", "tail_losers",
             "winners_hurt", "winner_bps_given_up", "tail_losers_cut", "tail_bps_saved", "meets_targets"]
    _ab = ablation.select([c for c in _cols if c in ablation.columns])
    _pf = alt.Chart(_ab).mark_bar().encode(
        y=alt.Y("variant:N", title=None, sort=None), x=alt.X("profit_factor:Q", title="Profit factor"),
        color=alt.Color("period:N", scale=PERIOD_COLORS), yOffset="period:N", tooltip=_cols[:8],
    ).properties(width=360, height=380, title="Profit factor by variant")
    _wr = _pf.encode(x=alt.X("win_rate:Q", title="Win rate %", scale=alt.Scale(domain=[0, 100]))).properties(title="Win rate by variant")
    _pf_rule = alt.Chart(pl.DataFrame({"x": [1.5]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(x="x:Q")
    _wr_rule = alt.Chart(pl.DataFrame({"x": [60.0]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(x="x:Q")
    _trade = alt.Chart(_ab.filter(pl.col("winners_hurt").is_not_null()) if "winners_hurt" in _ab.columns else _ab).mark_circle(size=120).encode(
        x=alt.X("winner_bps_given_up:Q", title="Winner bps given up"), y=alt.Y("tail_bps_saved:Q", title="Tail-loser bps saved"),
        color=alt.Color("period:N", scale=PERIOD_COLORS), shape="period:N",
        tooltip=["variant", "period", "winners_hurt", "tail_losers_cut", "winner_bps_given_up", "tail_bps_saved"],
    ).properties(width=420, height=320, title="Trade-off: what each filter saves vs what it costs")
    _even = alt.Chart(pl.DataFrame({"x": [0.0, float(_ab["winner_bps_given_up"].max() or 1.0)]})).mark_line(strokeDash=[4, 4], color=GREY).encode(x="x:Q", y=alt.Y("x:Q"))
    ablation_view = mo.vstack([
        mo.md("Each filter is applied alone to the baseline. **Above the dashed line** in the trade-off chart, a filter saves more on tail losers than it gives up on winners. Judge filters on out-of-sample, not in-sample."),
        mo.hstack([alt.layer(_pf, _pf_rule), alt.layer(_wr, _wr_rule)]),
        alt.layer(_even, _trade),
        mo.ui.table(_ab, selection=None, page_size=30),
    ])
    ablation_view
    return


@app.cell(hide_code=True)
def _(base, intraday, is_range, run_sweep, universe):
    mo.stop(not run_sweep.value)
    with mo.status.spinner(title="Sweeping bailout threshold × window"):
        _sw = metrics.sweep_mae_bail(universe, intraday, base, *is_range.value)
    _hm = alt.Chart(_sw).mark_rect().encode(
        x=alt.X("window_min:O", title="Bail window (minutes after entry)"), y=alt.Y("bail_bps:O", title="Bail if adverse ≥ (bps)", sort="descending"),
        color=alt.Color("profit_factor:Q", scale=alt.Scale(scheme="redyellowgreen", domainMid=1.5), title="PF"),
        tooltip=["bail_bps", "window_min", "trades", "win_rate", "avg_net_bps", "profit_factor", "worst_bps"],
    ).properties(width=380, height=260, title="Early MAE bailout grid: profit factor (in-sample)")
    _txt = alt.Chart(_sw).mark_text(fontSize=11).encode(x="window_min:O", y=alt.Y("bail_bps:O", sort="descending"), text=alt.Text("profit_factor:Q", format=".2f"))
    mo.vstack([mo.md("A good parameter sits in a broad green plateau, not on a lone bright cell (an overfit peak)."), _hm + _txt])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Entry gate: decision tree on arm-time features
    A shallow tree sees only what is known when the red bar closes (gap, time to trigger, run-up, red-bar volume and body,
    distance from VWAP, extension). It is fit on the tune period to flag likely tail losers, then scored on periods it never
    saw. If it only helps where it was fit, it is overfit.
    """)
    return


@app.cell(hide_code=True)
def _():
    gate_depth = mo.ui.slider(1, 5, value=3, label="Tree depth", show_value=True)
    gate_leaf = mo.ui.slider(5, 100, value=25, step=5, label="Min trades per leaf", show_value=True)
    gate_veto = mo.ui.slider(0.3, 0.9, value=0.6, step=0.05, label="Veto if P(tail) ≥", show_value=True)
    mo.hstack([gate_depth, gate_leaf, gate_veto], justify="start", gap=2)
    return gate_depth, gate_leaf, gate_veto


@app.cell(hide_code=True)
def _(gate_depth, gate_leaf, gate_veto, runs):
    _train = runs["in-sample"].trades
    mo.stop(_train.height < 50 or int((_train["net_bps"] <= LOSER_BPS).sum()) < 5, mo.md("_Too few tune-period trades or tail losers to fit a gate._"))
    tree = gate.fit(_train, depth=gate_depth.value, min_leaf=gate_leaf.value)
    _rows, _scored = [], []
    for _k, _r in runs.items():
        if _r.trades.height == 0:
            continue
        _sc = gate.score(tree, _r.trades, veto_prob=gate_veto.value).with_columns(pl.lit(_k).alias("period"))
        _scored.append(_sc)
        for _name, _df in (("baseline", _sc), ("gated", _sc.filter(~pl.col("gated")))):
            _s = metrics.summary(_df)
            _rows.append({
                "period": _k, "rules": _name,
                **{c: _s.get(c) for c in ("trades", "win_rate", "profit_factor", "avg_net_bps", "tail_losers", "worst_bps", "max_dd_bps")},
                "vetoed": int(_sc["gated"].sum()) if _name == "gated" else 0,
                "tail_vetoed": int((_sc["gated"] & (_sc["net_bps"] <= LOSER_BPS)).sum()) if _name == "gated" else 0,
            })
    _sc_all = pl.concat(_scored).with_columns(
        pl.when(pl.col("net_bps") <= LOSER_BPS).then(pl.lit("tail loser")).otherwise(pl.lit("other")).alias("result")
    ).select("period", "p_tail", "result")
    _gb = _sc_all.with_columns((pl.col("p_tail") * 20).floor().truediv(20).round(2).alias("bin")).group_by(["period", "result", "bin"]).len()
    _dist = alt.Chart(_gb).mark_bar(opacity=0.8).encode(
        x=alt.X("bin:O", title="Tree P(tail loser), 0.05 bins"), y=alt.Y("len:Q", stack=True, title="Trades"),
        color=alt.Color("result:N", scale=alt.Scale(domain=["other", "tail loser"], range=[GREY, RED])),
        facet=alt.Facet("period:N", columns=3, title=None), tooltip=["period", "result", "bin", "len"],
    ).properties(width=300, height=150, title="Gate score by outcome")
    _imp = alt.Chart(gate.importances(tree)).mark_bar(color=BLUE).encode(
        x=alt.X("importance:Q"), y=alt.Y("feature:N", sort="-x", title=None)).properties(width=300, height=220, title="Feature importance")
    mo.vstack([
        mo.ui.table(pl.DataFrame(_rows), selection=None),
        _dist,
        mo.hstack([_imp, mo.md("**Rules (fit on tune period)**\n```text\n" + gate.rules(tree) + "\n```")], justify="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Decision traces (OpenTelemetry)
    """)
    return


@app.cell(hide_code=True)
def _(runs):
    _days = sorted(set(runs["in-sample"].outcomes["trade_date"].to_list()))
    _tradedays = runs["in-sample"].trades.group_by("trade_date").agg(pl.col("net_bps").min().alias("worst")).sort("worst")
    _default = _tradedays["trade_date"][0] if _tradedays.height else (_days[0] if _days else None)
    trace_day = mo.ui.dropdown(
        options={str(d): d for d in _days}, value=str(_default) if _default else None,
        label="Session (default: worst in-sample day)", searchable=True,
    )
    trace_variant = mo.ui.dropdown(options=["baseline", "bail 100bps/5m", "HOD + 0.25 ATR", "red vol ≥ 1.5x", "VWAP rejection"], value="baseline", label="Rules")
    mo.hstack([trace_day, trace_variant], justify="start")
    return trace_day, trace_variant


@app.cell(hide_code=True)
def _(base, intraday, tiers, trace_day, trace_variant, universe):
    mo.stop(trace_day.value is None, mo.md("No sessions."))
    _v = {
        "baseline": base,
        "bail 100bps/5m": replace(base, mae_bail_bps=100, mae_bail_window_min=5),
        "HOD + 0.25 ATR": replace(base, hod_atr_stop=0.25),
        "red vol ≥ 1.5x": replace(base, red_vol_min_ratio=1.5),
        "VWAP rejection": replace(base, vwap_reject=True),
    }[trace_variant.value]
    _d = trace_day.value
    _r = engine.run(universe, intraday, _v, _d, _d, tiers=tuple(tiers.value), keep_machines=True)
    _spans = tracing.session_spans(_r.machines, _d, _v.label())
    trace_widget = mo.ui.anywidget(TraceView.from_spans(_spans, title=f"Session {_d} · {_v.label()}", height=620))
    _funnel = pl.DataFrame({
        "stage": ["gap-eligible", "hit threshold", "armed (red bar)", "filled", "winners", f"tail ≤ {LOSER_BPS:.0f}"],
        "n": [
            _r.outcomes.height, int(_r.outcomes["triggered"].sum()), int(_r.outcomes["armed"].sum()), _r.trades.height,
            int((_r.trades["net_bps"] > 0).sum()) if _r.trades.height else 0,
            int((_r.trades["net_bps"] <= LOSER_BPS).sum()) if _r.trades.height else 0,
        ],
    })
    _fchart = alt.Chart(_funnel).mark_bar(color=BLUE).encode(y=alt.Y("stage:N", sort=None, title=None), x=alt.X("n:Q", title="Symbols"), tooltip=["stage", "n"]).properties(width=300, height=200, title="Decision funnel")
    _tt = _r.trades.select("symbol", "tier", "entry_ts", "exit_reason", pl.col("net_bps").round(1), pl.col("mae_bps").round(0), pl.col("red_vol_ratio").round(2), pl.col("close_vs_avwap_bps").round(0)) if _r.trades.height else pl.DataFrame()
    _save = tracing.save_spans(_spans, ROOT / "data" / "traces" / f"{_d}_{trace_variant.value.replace(' ', '_').replace('≥', 'ge')}.json")
    mo.vstack([
        mo.md("One trace per session. Each symbol is a span; its children are the decision stages timed in **market time** "
              "(screen+gap → watch → arm → entry → position). Red spans are continuation losers. Click a span for its numbers; drag the scrubber to replay the morning."),
        trace_widget,
        mo.hstack([_fchart, mo.ui.table(_tt, selection=None, page_size=12)], justify="start"),
        mo.md(f"<small>Spans saved to `{_save.relative_to(ROOT)}` (reload with `TraceView(spans=json.load(...))`).</small>"),
    ])
    return (trace_widget,)


@app.cell(hide_code=True)
def _(trace_widget):
    _s = trace_widget.widget.selected_span if hasattr(trace_widget, "widget") else None
    mo.md("") if not _s else mo.callout(mo.md(f"**{_s['name']}**  \n" + "  \n".join(f"`{k}`: {v}" for k, v in _s["attributes"].items())), kind="neutral")
    return


@app.cell(hide_code=True)
def _(summaries):
    _s = summaries["in-sample"]
    _wr = (_s.get("win_rate") or 60) / 100
    math_dag = mo.ui.anywidget(
        FormulaDAG(
            {
                "thr": r"P_{\text{thr}} = O \cdot \left(1 + \tfrac{\tangle{s}}{100}\right)",
                "gap": r"g = \frac{O - C_{t-1}}{\mathrm{ATR}_{14}} \in [0.25,\ 1.0]",
                "arm": r"\text{arm} = \min\{\,i \ge i^{*} : c_i < o_i\,\},\ \ i^{*} = \min\{i : h_i \ge \use{thr}\}",
                "ret": r"r = \frac{o_{\text{arm}+1} - o_{\text{arm}+1+\tangle{H}}}{o_{\text{arm}+1}} - \tangle{k}\cdot 10^{-4}",
                "pf": r"\mathrm{PF} = \frac{\sum r^{+}}{\left|\sum r^{-}\right|}",
                "exp": r"\mathbb{E}[r] = w\,\bar{W} - (1-w)\,\bar{L}",
            },
            parameters={
                "s": {"value": 5, "min_value": 3, "max_value": 8, "step": 0.5, "digits": 1},
                "H": {"value": 20, "min_value": 5, "max_value": 40, "step": 1, "digits": 0},
                "k": {"value": 5, "min_value": 0, "max_value": 30, "step": 0.5, "digits": 1},
            },
            notes={},
        )
    )
    breakeven = mo.ui.anywidget(
        TangleLatex(
            r"\text{day} = \tangle{nw}\times\tangle{W}\ \text{bps} \;-\; \tangle{nl}\times\tangle{L}\ \text{bps}",
            theme="light",
            parameters={
                "nw": {"digits": 0, "value": 8, "min_value": 0, "max_value": 15, "step": 1},
                "W": {"digits": 0, "value": 30, "min_value": 5, "max_value": 80, "step": 1},
                "nl": {"digits": 0, "value": 1, "min_value": 0, "max_value": 5, "step": 1},
                "L": {"digits": 0, "value": 300, "min_value": 50, "max_value": 600, "step": 10},
            },
        )
    )
    return breakeven, math_dag


@app.cell(hide_code=True)
def _(breakeven, math_dag, summaries):
    _v = breakeven.widget.values if hasattr(breakeven, "widget") else breakeven.value.get("values", {})
    _v = _v or {"nw": 8, "W": 30, "nl": 1, "L": 300}
    _day = _v["nw"] * _v["W"] - _v["nl"] * _v["L"]
    _need = _v["nl"] * _v["L"] / max(1, _v["W"])
    _wr_be = _v["L"] / (_v["L"] + _v["W"]) * 100
    _s = summaries["in-sample"]
    math_view = mo.vstack([
        mo.md(r"### The decision math (drag the numbers)"),
        math_dag,
        mo.md(r"### Why one continuation loser flattens the day"),
        breakeven,
        mo.md(
            rf"""
    Day result: **{_day:+.0f} bps**. It takes $\lceil n_\ell L / W \rceil = {int(np.ceil(_need))}$ winners to pay for
    the losers. With every loser at $L$ and every winner at $W$, the break-even win rate is
    $\dfrac{{L}}{{L+W}} = {_wr_be:.1f}\%$. In-sample the baseline shows win rate **{_s.get('win_rate')}%**,
    PF **{_s.get('profit_factor')}**, average winner vs loser in the table below.
    """
        ),
    ])
    return (math_view,)


@app.cell(hide_code=True)
def _(trades_is):
    _wl = trades_is.select(
        pl.col("net_bps").filter(pl.col("net_bps") > 0).mean().round(1).alias("avg_winner_bps"),
        pl.col("net_bps").filter(pl.col("net_bps") <= 0).mean().round(1).alias("avg_loser_bps"),
        pl.col("net_bps").filter(pl.col("net_bps") <= LOSER_BPS).mean().round(1).alias("avg_tail_loser_bps"),
    ) if trades_is.height else pl.DataFrame()
    winloss = mo.ui.table(_wl, selection=None)
    return (winloss,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## PyBroker cross-check
    """)
    return


@app.cell(hide_code=True)
def _():
    run_pyb = mo.ui.run_button(label="Cross-check in PyBroker (in-sample)")
    run_pyb
    return (run_pyb,)


@app.cell(hide_code=True)
def _(base, intraday, is_range, run_pyb, runs, tiers, universe):
    mo.stop(not run_pyb.value, mo.md("_PyBroker recomputes the arm signal independently in Polars, then fills: short at the next open, cover after the hold at the open._"))
    with mo.status.spinner(title="PyBroker backtest"):
        _sig = pbc.arm_signals(universe, intraday, spike_pct=base.spike_pct, arm_window_min=base.arm_window_min,
                               start=is_range.value[0], end=is_range.value[1], tiers=tuple(tiers.value))
        _pb = pbc.run_pybroker(_sig, hold=base.hold_minutes, slippage_bps=base.slippage_bps)
    _rec = pbc.reconcile(runs["in-sample"].trades, _pb)
    _ok = _rec["matched"] == _rec["engine_trades"] == _rec["pybroker_trades"] and _rec["mean_abs_bps_diff"] < 1
    mo.vstack([
        mo.callout(mo.md(("**Engines agree.** " if _ok else "**Engines disagree; investigate.** ") + ", ".join(f"{k}: {v}" for k, v in _rec.items())), kind="success" if _ok else "warn"),
        mo.md("PyBroker rounds fills to the cent, which explains a sub-bp difference."),
    ])
    return


@app.cell(hide_code=True)
def _():
    sql_note = mo.md("## SQL\nDuckDB runs directly over the Polars trade table. Edit the query.")
    sql_note
    return


@app.cell
def _(trades_all):
    sql_result = mo.sql(
        f"""
        SELECT period, tier,
               dayname(trade_date) AS weekday,
               count(*) AS trades,
               round(avg(net_bps), 1) AS avg_net_bps,
               round(100 * avg(CASE WHEN net_bps > 0 THEN 1 ELSE 0 END), 1) AS win_rate,
               sum(CASE WHEN net_bps <= -200 THEN 1 ELSE 0 END) AS tail_losers
        FROM trades_all
        GROUP BY ALL
        ORDER BY period, tier, weekday
        """
    )
    return


@app.cell(hide_code=True)
def _(universe):
    _f = screen_funnel(universe)
    _per_day = universe.filter(pl.col("in_universe")).group_by(["trade_date", "tier"]).len().sort("trade_date")
    _gap = universe.filter(pl.col("gap_ok")).group_by("trade_date").len().sort("trade_date")
    _c1 = alt.Chart(_per_day).mark_area(opacity=0.7).encode(
        x=alt.X("trade_date:T", title=None), y=alt.Y("len:Q", stack=True, title="Symbols in universe"),
        color=alt.Color("tier:N", scale=alt.Scale(range=[BLUE, AMBER])),
    ).properties(height=200, width=560, title="Universe size per day by tier")
    _c2 = alt.Chart(_gap).mark_bar(color=GREEN).encode(x=alt.X("trade_date:T", title=None), y=alt.Y("len:Q", title="Gap 0.25-1.0 ATR")).properties(height=140, width=560, title="Gap-eligible candidates per day")
    universe_view = mo.vstack([
        mo.hstack([mo.vstack([_c1, _c2]), mo.vstack([mo.md("**Screen funnel (avg symbols/day by first failing rule)**"), mo.ui.table(_f, selection=None)])], justify="start"),
        mo.md("Features use prior-day data only (DuckDB window ends at `1 PRECEDING`). ATR is a 14-day simple average of true range; ADDV is the 20-day average of close × volume."),
    ])
    return (universe_view,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Live path (async).** `spike_lab.live` runs the same `SymbolDay.step` behind three asyncio tasks joined by queues:
    bar ingestion → decisions → order intents (logged, never sent). That split is what a Rust (tokio) or C++ (asio)
    port keeps: only the ingestion and gateway tasks change; `rules.py` ports line for line.
    """)
    return


if __name__ == "__main__":
    app.run()
