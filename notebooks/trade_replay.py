# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "duckdb>=2.0.1",
#     "polars>=2.1.0",
#     "pyarrow",
#     "numpy",
#     "altair>=6.3.0",
#     "anywidget",
#     "scikit-learn",
#     "lightgbm",
#     "catboost",
#     "tabpfn",
# ]
# ///
"""Trade replay: one spike-short trade, bar by bar, with the risk models' view.

Paper research only. Nothing here places orders.
Run:  uv run marimo run notebooks/trade_replay.py
Charts: TradingView Lightweight Charts (Apache-2.0) via src/spike_lab/viz/lwchart.py.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Spike Short Trade Replay")

with app.setup:
    import sys
    from dataclasses import replace
    from datetime import date, datetime, timedelta
    from pathlib import Path

    import altair as alt
    alt.theme.enable("dark")
    import marimo as mo
    import numpy as np
    import polars as pl

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))

    from spike_lab.metrics import LOSER_BPS
    from spike_lab.models import features as F
    from spike_lab.models import jev_hook, lab, replay
    from spike_lab.models.zoo import REGISTRY
    from spike_lab.rules import Params
    from spike_lab.universe import build_universe
    from spike_lab.viz.lwchart import ReplayChart

    ALPACA_DIR = ROOT / "data" / "alpaca"
    GREEN, RED, BLUE, GREY, AMBER, PURPLE = "#2f9e6e", "#d64545", "#3b6fd6", "#8a8f98", "#d69a2d", "#8a2be2"
    TUNE = (date(2025, 7, 1), date(2026, 6, 30))
    PERIODS = {"Jul 25 - Jun 26 (tune)": TUNE, "Q3 26 (holdout)": (date(2026, 7, 1), date(2026, 9, 30)),
               "Oct - Nov 26 (holdout)": (date(2026, 10, 1), date(2026, 11, 30))}
    MODEL_CHOICES = ["LightGBM", "CatBoost", "Logistic (L2)", "TabPFN v2", "Chronos-Bolt: q90 path max", "Kronos: q90 path max"]
    FITS: dict = {}

    def epoch(ts) -> int:
        """ET wall-clock as if UTC, so the chart axis reads in ET."""
        return int((ts - datetime(1970, 1, 1)).total_seconds())


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'trade_replay'
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
    # Spike Short Trade Replay
    Pick a trade, then press **▶ Play** to watch the morning one minute at a time: the run to **+5%**, the first red bar
    (**arm**), the **short**, and the **cover** 20 minutes later. The purple pane is the risk models' **P(continuation loser)**
    if the short were entered at the next bar; above the dashed **veto** line the model would skip the trade.

    <small>Alpaca SIP 1-minute bars. Models are trained on Jul 2025 – Jun 2026 only, so Q3 and Oct–Nov trades are scored out of sample.
    Charts: TradingView Lightweight Charts (Apache-2.0). Paper research only.</small>
    """)
    return


@app.cell(hide_code=True)
def _():
    with mo.status.spinner(title="Loading bars, building the universe, running the rule on every period"):
        daily = pl.read_parquet(ALPACA_DIR / "daily.parquet")
        intraday = pl.read_parquet(ALPACA_DIR / "intraday.parquet")
        bench = pl.read_parquet(ALPACA_DIR / "benchmarks.parquet")
        universes = {p: build_universe(daily, min_price=p) for p in (25.0, 50.0)}
        base = Params()
        trades = pl.concat([
            F.build_dataset(universes[50.0], intraday, daily, bench, base, a, b).with_columns(pl.lit(k).alias("period"))
            for k, (a, b) in PERIODS.items()
        ], how="diagonal_relaxed")
        pool = F.pooled_training_set(universes, intraday, daily, bench, base, (0.04, 0.05), (25.0, 50.0), *TUNE)
    strat_tune = trades.filter(pl.col("period") == next(iter(PERIODS)))
    mo.md(f"**{trades.height} trades** (5% run-up, \\$50+, both tiers): " + ", ".join(
        f"{k} {v}" for k, v in trades.group_by("period", maintain_order=True).len().iter_rows()) + f". Training pool: {pool.height} tune trades.")
    return bench, intraday, pool, strat_tune, trades


@app.cell(hide_code=True)
def _():
    models = mo.ui.multiselect(options=MODEL_CHOICES, value=["LightGBM", "CatBoost"], label="Risk model(s), averaged")
    veto_rate = mo.ui.slider(0.05, 0.5, value=0.2, step=0.05, label="Veto the riskiest share of tune trades", show_value=True)
    mo.hstack([models, veto_rate], justify="start", gap=2)
    return models, veto_rate


@app.cell(hide_code=True)
def _(models, pool, strat_tune, trades, veto_rate):
    mo.stop(not models.value, mo.md("_Pick at least one model._"))
    _feats = F.BASE_FEATURES
    _hold = {"all": trades}
    for _m in mo.status.progress_bar([m for m in models.value if m not in FITS], title="Fitting on the tune period (cached after the first time)"):
        FITS[_m] = lab.fit_all([_m], pool, strat_tune, _hold, _feats)[_m]
    fits = {m: FITS[m] for m in models.value}
    # ensemble = mean of calibrated probabilities; threshold from tune out-of-fold scores only
    _oof = np.mean([f.oof for f in fits.values()], axis=0)
    veto = float(np.quantile(_oof, 1 - veto_rate.value))
    scored = trades.with_columns(pl.Series("p_tail", np.mean([f.holdout["all"] for f in fits.values()], axis=0)))
    # tune rows: use out-of-fold scores (fair), not the in-sample final-model score
    _tune_idx = scored.with_row_index("_i").filter(pl.col("period") == next(iter(PERIODS)))["_i"].to_numpy()
    _p = scored["p_tail"].to_numpy().copy()
    _p[_tune_idx] = _oof
    scored = scored.with_columns(pl.Series("p_tail", _p)).with_columns(
        (pl.col("p_tail") >= veto).alias("model_veto"),
        pl.when(pl.col("net_bps") <= LOSER_BPS).then(pl.lit("tail loser")).when(pl.col("net_bps") > 0).then(pl.lit("winner")).otherwise(pl.lit("small loser")).alias("outcome"),
    )
    return fits, scored, veto


@app.cell(hide_code=True)
def _():
    period = mo.ui.dropdown(options=["all", *PERIODS], value="Q3 26 (holdout)", label="Period")
    outcome = mo.ui.dropdown(options=["all", "tail loser", "small loser", "winner"], value="tail loser", label="Outcome")
    verdict = mo.ui.dropdown(options=["all", "model vetoed", "model allowed"], value="all", label="Model")
    mo.hstack([period, outcome, verdict], justify="start", gap=2)
    return outcome, period, verdict


@app.cell(hide_code=True)
def _(outcome, period, scored, verdict):
    _t = scored
    if period.value != "all":
        _t = _t.filter(pl.col("period") == period.value)
    if outcome.value != "all":
        _t = _t.filter(pl.col("outcome") == outcome.value)
    if verdict.value != "all":
        _t = _t.filter(pl.col("model_veto") == (verdict.value == "model vetoed"))
    view = _t.sort("net_bps").select(
        "symbol", "trade_date", "tier", pl.col("entry_ts").dt.strftime("%H:%M").alias("entry"),
        pl.col("net_bps").round(0), pl.col("mae_bps").round(0), pl.col("minutes_to_trigger").round(0),
        pl.col("p_tail").round(3), "model_veto", "outcome", "period", "entry_ts",
    )
    picker = mo.ui.table(view, selection="single", initial_selection=[0] if view.height else None, page_size=8,
                         label="Pick a trade (sorted worst first)")
    picker
    return (picker,)


@app.cell(hide_code=True)
def _(picker, scored):
    mo.stop(picker.value is None or picker.value.height == 0, mo.md("_Select a trade in the table._"))
    _k = picker.value.row(0, named=True)
    trade = scored.filter((pl.col("symbol") == _k["symbol"]) & (pl.col("trade_date") == _k["trade_date"]) & (pl.col("entry_ts") == _k["entry_ts"])).row(0, named=True)
    return (trade,)


@app.cell(hide_code=True)
def _(bench, fits, intraday, trade):
    bars = replay.day_bars(intraday, trade["symbol"], trade["trade_date"])
    _open = bars["ts"][0].replace(hour=9, minute=30)
    trigger_ts = _open + timedelta(minutes=int(trade["minutes_to_trigger"]))
    arm_ts = trade["entry_ts"] - timedelta(minutes=1)
    bars = replay.vwaps(bars, trigger_ts)
    hyp = replay.hypothetical_entries(trade, bars, bench, trigger_ts, trade["exit_ts"])
    _usable = [f for f in fits.values() if all(c in hyp.columns for c in f.features)]  # forecasters only score real entries
    if hyp.height and _usable:
        _p = np.mean([f.calib(f.final.predict_proba(hyp.select([pl.col(c).cast(pl.Float64) for c in f.features]).to_numpy()))
                      for f in _usable], axis=0)
        risk = hyp.select("bar_ts").with_columns(pl.Series("p", _p))
    else:
        risk = pl.DataFrame({"bar_ts": [], "p": []})
    return arm_ts, bars, risk, trigger_ts


@app.cell(hide_code=True)
def _(arm_ts, bars, risk, trade, trigger_ts, veto):
    _thr = trade["day_open"] * 1.05
    _mk = [
        {"time": epoch(bars["ts"][0]), "position": "belowBar", "color": GREY, "shape": "circle", "text": "open"},
        {"time": epoch(trigger_ts), "position": "aboveBar", "color": AMBER, "shape": "circle", "text": "+5%"},
        {"time": epoch(arm_ts), "position": "aboveBar", "color": RED, "shape": "square", "text": "arm (first red)"},
        {"time": epoch(trade["entry_ts"]), "position": "aboveBar", "color": RED, "shape": "arrowDown", "text": f"SHORT {trade['entry_px']:.2f}"},
        {"time": epoch(trade["exit_ts"]), "position": "belowBar", "color": GREEN if trade["net_bps"] > 0 else RED, "shape": "arrowUp",
         "text": f"COVER {trade['net_bps']:+.0f} bps"},
    ]
    _rows = bars.select("ts", "open", "high", "low", "close", "volume").iter_rows(named=True)
    _bars = [{"time": epoch(r["ts"]), **{k: float(r[k]) for k in ("open", "high", "low", "close", "volume")}} for r in _rows]
    _start = int(bars.filter(pl.col("ts") <= trigger_ts).height)
    chart = mo.ui.anywidget(ReplayChart(
        bars=_bars,
        vwap=[{"time": epoch(t), "value": float(v)} for t, v in bars.select("ts", "vwap").iter_rows()],
        avwap=[{"time": epoch(t), "value": float(v)} for t, v in bars.select("ts", "avwap").drop_nulls().iter_rows()],
        risk=[{"time": epoch(t), "value": float(v)} for t, v in risk.iter_rows()],
        markers=sorted(_mk, key=lambda m: m["time"]),
        price_lines=[
            {"price": float(trade["day_open"]), "title": "open", "color": GREY},
            {"price": float(_thr), "title": "+5% trigger", "color": AMBER},
            {"price": float(trade["entry_px"]), "title": "short", "color": RED, "style": 0, "from_time": epoch(trade["entry_ts"])},
            {"price": float(trade["entry_px"] * (1 - LOSER_BPS / 1e4)), "title": "tail loss (−2%)", "color": "#b03030", "from_time": epoch(trade["entry_ts"])},
        ],
        upto=_start, start_at=_start, veto=float(veto), height=580,
        title=f"{trade['symbol']} · {trade['trade_date']} · {trade['tier']} · {trade['period']}",
        legend="Blue: session VWAP · dashed amber: VWAP since the +5% trigger · purple: P(tail) if shorted at the next bar · Lightweight Charts by TradingView",
    ))
    return (chart,)


@app.cell(hide_code=True)
def _(chart, fits, trade, veto):
    _p = trade["p_tail"]
    _cards = mo.hstack([
        mo.stat(f"{trade['net_bps']:+.0f} bps", label="Result (net)", bordered=True,
                caption=f"{trade['outcome']} · cover {trade['exit_reason']}"),
        mo.stat(f"{_p:.2f}", label="Model P(tail) at entry", bordered=True,
                caption=("VETO: model would skip" if _p >= veto else "model would trade") + f" (veto ≥ {veto:.2f})"),
        mo.stat(f"{trade['mae_bps']:.0f} / {trade['mfe_bps']:.0f}", label="Worst / best move (bps)", bordered=True, caption="against / for the short"),
        mo.stat(f"{trade['minutes_to_trigger']:.0f} min", label="Open to +5%", bordered=True, caption="tail losers: median 5, winners 12"),
        mo.stat(f"{trade['red_vol_ratio']:.2f}x", label="Red bar volume", bordered=True, caption="vs run-up average"),
    ], justify="start", gap=1, wrap=True)
    mo.vstack([_cards, chart, mo.md(f"<small>Models: {', '.join(fits)} (mean of calibrated probabilities).</small>")])
    return


@app.cell(hide_code=True)
def _(scored, trade):
    _tune = scored.filter(pl.col("period") == next(iter(PERIODS))).select("outcome", "path_bps").with_row_index("id").explode("path_bps")
    _tune = _tune.with_columns(pl.int_range(pl.len()).over("id").alias("minute"))
    _band = _tune.filter(pl.col("outcome") != "small loser").group_by("outcome", "minute").agg(
        pl.col("path_bps").quantile(0.25).alias("q25"), pl.col("path_bps").median().alias("median"), pl.col("path_bps").quantile(0.75).alias("q75")).sort("minute")
    _me = pl.DataFrame({"minute": list(range(len(trade["path_bps"]))), "path_bps": trade["path_bps"]})
    _col = alt.Scale(domain=["winner", "tail loser"], range=[GREEN, RED])
    _area = alt.Chart(_band).mark_area(opacity=0.18).encode(x=alt.X("minute:Q", title="Minutes after entry"), y=alt.Y("q25:Q", title="Short P&L vs entry (bps)"), y2="q75:Q", color=alt.Color("outcome:N", scale=_col, title="tune trades (median, 25-75%)"))
    _med = alt.Chart(_band).mark_line(strokeDash=[4, 3]).encode(x="minute:Q", y="median:Q", color=alt.Color("outcome:N", scale=_col))
    _line = alt.Chart(_me).mark_line(color="#e5ece8", strokeWidth=2.5, point=True).encode(x="minute:Q", y="path_bps:Q", tooltip=["minute", "path_bps"])
    _zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")
    _tail = alt.Chart(pl.DataFrame({"y": [-200.0]})).mark_rule(color=RED, strokeDash=[6, 4]).encode(y="y:Q")
    mo.vstack([
        mo.md("### This trade's path vs typical winners and tail losers\nBlack: this trade, close-to-entry each minute (positive = the short is making money). "
              "Bands: middle half of tune-period winners (green) and tail losers (red). Tail losers usually show themselves in the first 3–5 minutes."),
        (_area + _med + _zero + _tail + _line).properties(width=760, height=300),
    ])
    return


@app.cell(hide_code=True)
def _(fits, trade):
    _f = next((f for f in fits.values() if f.final.explain(np.zeros((1, len(f.features)))) is not None), None)
    mo.stop(_f is None, mo.md("_The chosen models have no per-trade explanation; add LightGBM, CatBoost or Logistic._"))
    _x = np.array([[trade.get(c) if trade.get(c) is not None else np.nan for c in _f.features]], dtype=float)
    _v = _f.final.explain(_x)[0]
    _d = pl.DataFrame({"feature": _f.features, "shap": _v, "value": _x[0]}).with_columns(pl.col("shap").abs().alias("a")).sort("a", descending=True).head(12)
    alt.Chart(_d).mark_bar().encode(
        x=alt.X("shap:Q", title=f"{_f.name}: push toward tail loser (+) or away (−), log-odds"),
        y=alt.Y("feature:N", sort=_d["feature"].to_list(), title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.condition(alt.datum.shap > 0, alt.value(RED), alt.value(GREEN)), tooltip=["feature", "value", "shap"],
    ).properties(width=560, height=300, title="Why the model scored this trade the way it did (SHAP)")
    return


@app.cell(hide_code=True)
def _(bars, chart, trade):
    _k = max(1, min(chart.value.get("upto", 1) if isinstance(chart.value, dict) else 1, bars.height))
    _bar = bars.row(_k - 1, named=True)
    _in = trade["entry_ts"] <= _bar["ts"] < trade["exit_ts"]
    _mins = int((_bar["ts"] - trade["entry_ts"]).total_seconds() // 60) if _in else None
    _feats = {c: trade.get(c) for c in ("gap_atr", "minutes_to_trigger", "red_vol_ratio", "close_vs_avwap_bps", "entry_vs_hod_bps", "p_tail")}
    _state = jev_hook.bar_state(trade, _bar, _feats, _in, _mins)
    _status = (mo.callout(mo.md("**Jev: connected.** Per-bar decisions will appear here."), kind="success") if jev_hook.connected()
               else mo.callout(mo.md("**Jev per-bar decision: not connected yet.** Add `JEV_API_KEY=...` to `.env` and rerun. "
                                     "The hook (`src/spike_lab/models/jev_hook.py`) will send the state below and get back one of: "
                                     + ", ".join(f"`{a}`" for a in jev_hook.ACTIONS) + "."), kind="neutral"))
    mo.vstack([mo.md("### Jev hook (System One decision per bar)"), _status,
               mo.accordion({f"State sent at the replay cursor ({_bar['ts']:%H:%M} ET)": mo.json(_state)})])
    return


if __name__ == "__main__":
    app.run()
