# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "duckdb>=2.0.1",
#     "polars>=2.1.0",
#     "pyarrow",
#     "numpy",
#     "altair>=6.3.0",
#     "scikit-learn",
#     "lightgbm",
#     "catboost",
#     "shap",
#     "tabpfn",
#     "chronos-forecasting",
# ]
# ///
"""Continuation-loser model lab for the opening spike short.

Paper research only. Nothing here places orders.
Run:  uv run marimo edit notebooks/model_lab.py
Engine: src/spike_lab (rules.py); models: src/spike_lab/models (features, zoo, lab).
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Spike Short Model Lab")

with app.setup:
    import sys
    from dataclasses import replace
    from datetime import date
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
    from spike_lab.models import lab
    from spike_lab.models.zoo import LAYER1, REGISTRY
    from spike_lab.rules import Params
    from spike_lab.universe import build_universe

    ALPACA_DIR = ROOT / "data" / "alpaca"
    GREEN, RED, BLUE, GREY, AMBER = "#2f9e6e", "#d64545", "#3b6fd6", "#8a8f98", "#d69a2d"
    TUNE = (date(2025, 7, 1), date(2026, 6, 30))
    HOLDOUTS = {"validate (Q3 26)": (date(2026, 7, 1), date(2026, 9, 30)),
                "forward (Oct-Nov 26)": (date(2026, 10, 1), date(2026, 11, 30))}
    TUNE_KEY = "tune (OOF)"
    PERIOD_SCALE = alt.Scale(domain=[TUNE_KEY, *HOLDOUTS], range=[BLUE, AMBER, GREY])
    alt.data_transformers.disable_max_rows()
    FIT_CACHE: dict = {}  # (model, data signature, features, calibration) -> lab.Fitted


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'model_lab'
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
    # Spike Short Model Lab
    Which trades will be **continuation losers** (net $\le -200$ bps)? About 15% of baseline trades lose that much and
    they make up 60–68% of all losing bps. Each model below scores $P(\text{tail loser})$ for a trade from facts known
    when the short would be placed (the arm bar has closed, entry is the next open), and a trade is skipped when the
    score crosses a threshold.

    **Protocol.** Models, calibration and thresholds are all set on **Jul 2025 – Jun 2026** using out-of-fold scores
    (blocked time folds with a 5-day purge). **Q3 2026** and **Oct – Nov 2026** are scored once with everything frozen.
    The tune row in the tables is out-of-fold, so it is a fair estimate too. Targets: win rate 60–70%, profit factor $\ge 1.5$.
    """)
    return


@app.cell(hide_code=True)
def _():
    with mo.status.spinner(title="Loading cached Alpaca bars and building the $25 and $50 universes"):
        daily = pl.read_parquet(ALPACA_DIR / "daily.parquet")
        intraday = pl.read_parquet(ALPACA_DIR / "intraday.parquet")
        bench = pl.read_parquet(ALPACA_DIR / "benchmarks.parquet") if (ALPACA_DIR / "benchmarks.parquet").exists() else None
        universes = {p: build_universe(daily, min_price=p) for p in (25.0, 50.0)}
    mo.md(f"Alpaca SIP bars {daily['trade_date'].min()} to {daily['trade_date'].max()}: "
          f"{intraday.height:,} one-minute morning bars, SPY/QQQ benchmarks {'loaded' if bench is not None else 'missing'}.")
    return bench, daily, intraday, universes


@app.cell(hide_code=True)
def _():
    spike = mo.ui.dropdown(options={"4%": 0.04, "5%": 0.05}, value="5%", label="Strategy run-up")
    min_price = mo.ui.dropdown(options={"$25": 25.0, "$50": 50.0}, value="$50", label="Min price")
    tiers = mo.ui.multiselect(options=["100M-1B", "1B+"], value=["100M-1B", "1B+"], label="ADDV tiers")
    hold = mo.ui.slider(5, 40, value=20, step=1, label="Cover after (min)", show_value=True)
    pool_spikes = mo.ui.multiselect(options={"4%": 0.04, "5%": 0.05}, value=["4%", "5%"], label="Train on run-ups")
    pool_prices = mo.ui.multiselect(options={"$25": 25.0, "$50": 50.0}, value=["$25", "$50"], label="Train on price floors")
    feat_groups = mo.ui.multiselect(options=list(F.FEATURE_GROUPS), value=[g for g in F.FEATURE_GROUPS if not g.startswith("forecast")], label="Feature groups")
    mo.vstack([
        mo.md("### Strategy and training data"),
        mo.hstack([spike, min_price, tiers, hold], justify="start", gap=2),
        mo.hstack([pool_spikes, pool_prices, feat_groups], justify="start", gap=2),
        mo.md("_Training pools trades from several run-up and price variants of the same rule (more labelled tail "
              "losers, ~180 instead of 55). Evaluation always uses the strategy trades above._"),
    ])
    return feat_groups, hold, min_price, pool_prices, pool_spikes, spike, tiers


@app.cell(hide_code=True)
def _(bench, daily, hold, intraday, min_price, pool_prices, pool_spikes, spike, tiers, universes):
    base = Params(spike_pct=spike.value, hold_minutes=hold.value)
    _tiers = tuple(tiers.value)
    _spikes = tuple(sorted(set(pool_spikes.value) | {spike.value}))
    _prices = tuple(sorted(set(pool_prices.value) | {min_price.value}))
    with mo.status.spinner(title="Running the rule and building decision-time features"):
        pool = F.pooled_training_set(universes, intraday, daily, bench, base, _spikes, _prices, *TUNE, tiers=_tiers)
        uu = universes[min_price.value]
        strat_tune = F.build_dataset(uu, intraday, daily, bench, base, *TUNE, tiers=_tiers)
        holdouts = {k: F.build_dataset(uu, intraday, daily, bench, base, a, b, tiers=_tiers) for k, (a, b) in HOLDOUTS.items()}
    frames = {TUNE_KEY: strat_tune, **holdouts}
    _fc = [ROOT / "data" / "models" / f"forecast_{m}.parquet" for m in ("chronos", "kronos")]
    data_sig = (base, _tiers, _spikes, _prices, tuple(p.stat().st_mtime if p.exists() else 0 for p in _fc))
    mo.md(
        f"**Training pool:** {pool.height} tune trades, {int(pool['y_tail'].sum())} tail losers. "
        f"**Strategy:** tune {strat_tune.height} ({int(strat_tune['y_tail'].sum())} tail), "
        + ", ".join(f"{k} {v.height} ({int(v['y_tail'].sum())} tail)" for k, v in holdouts.items())
    )
    return data_sig, frames, holdouts, pool, strat_tune


@app.cell(hide_code=True)
def _():
    models = mo.ui.multiselect(options=list(REGISTRY), value=LAYER1, label="Models (order = sequential order)")
    calib = mo.ui.dropdown(options={"Platt (logistic)": "platt", "Isotonic": "isotonic", "None": "none"}, value="Platt (logistic)", label="Calibration")
    thr_mode = mo.ui.dropdown(options={"Veto riskiest share of tune trades": "veto_rate", "Veto if P(tail) ≥": "prob",
                                       "Best tune profit factor": "best_pf"}, value="Veto riskiest share of tune trades", label="Threshold rule")
    thr_value = mo.ui.slider(0.05, 0.6, value=0.2, step=0.01, label="Veto share / probability", show_value=True)
    combo = mo.ui.dropdown(options={"Single models only": "none", "Vote k-of-n": "vote", "Mean probability": "mean",
                                    "Stack (logistic on model scores)": "stack", "Sequential gates": "sequential"},
                           value="Vote k-of-n", label="Combine")
    vote_k = mo.ui.slider(1, 6, value=2, label="k (vote)", show_value=True)
    run = mo.ui.run_button(label="Fit models")
    mo.vstack([
        mo.md("### Models and combination"),
        mo.hstack([models, calib], justify="start", gap=2),
        mo.hstack([thr_mode, thr_value, combo, vote_k, run], justify="start", gap=2),
        mo.md("_TabPFN runs in a separate worker process (its torch runtime crashes LightGBM in one process on macOS); "
              "the first TabPFN fit takes about a minute. Fits are cached, so changing thresholds or the combination is instant._"),
    ])
    return calib, combo, models, run, thr_mode, thr_value, vote_k


@app.cell(hide_code=True)
def _(calib, data_sig, feat_groups, holdouts, models, pool, run, strat_tune):
    feats = [f for g in feat_groups.value for f in F.FEATURE_GROUPS[g]]
    _need = [m for m in models.value if (m, data_sig, tuple(feats), calib.value) not in FIT_CACHE]
    mo.stop(not models.value, mo.md("_Pick at least one model._"))
    mo.stop(bool(_need) and not run.value and mo.app_meta().mode != "script",
            mo.callout(mo.md(f"Press **Fit models** to fit: {', '.join(_need)}."), kind="info"))
    for _m in mo.status.progress_bar(_need, title="Fitting (walk-forward folds, then the final model)"):
        FIT_CACHE[(_m, data_sig, tuple(feats), calib.value)] = lab.fit_all(
            [_m], pool, strat_tune, holdouts, feats, calib.value)[_m]
    fits = {m: FIT_CACHE[(m, data_sig, tuple(feats), calib.value)] for m in models.value}
    return feats, fits


@app.cell(hide_code=True)
def _(combo, fits, frames, strat_tune, thr_mode, thr_value, vote_k):
    _net, _y = strat_tune["net_bps"].to_numpy(), strat_tune["y_tail"].to_numpy()
    thresholds = {n: lab.pick_threshold(f.oof, _net, thr_mode.value, thr_value.value) for n, f in fits.items()}
    decisions = [lab.single(f, thresholds[n], TUNE_KEY) for n, f in fits.items()]
    combo_dec = None
    if combo.value != "none" and len(fits) >= 2:
        combo_dec = lab.combine(list(fits.values()), [thresholds[n] for n in fits], combo.value, _y, _net,
                                k=min(vote_k.value, len(fits)), mode=thr_mode.value, value=thr_value.value, tune_key=TUNE_KEY)
        decisions.append(combo_dec)
    rows = lab.baseline_rows(frames)
    for _d in decisions:
        rows += lab.evaluate(_d, frames)
    comparison = pl.DataFrame(rows)
    return combo_dec, comparison, decisions, thresholds


@app.cell(hide_code=True)
def _(comparison, fits, thresholds):
    _cols = ["strategy", "period", "trades", "vetoed", "win_rate", "profit_factor", "avg_net_bps", "total_bps",
             "losers_avoided", "tail_total", "winners_lost", "winners_total", "bps_saved", "bps_given_up",
             "net_bps_change", "worst_bps", "max_dd_bps", "auc", "meets_targets"]
    _thr = pl.DataFrame([{"model": n, "threshold P(tail)": round(t, 3), "fit seconds": round(fits[n].seconds, 1)} for n, t in thresholds.items()])
    mo.vstack([
        mo.md("## Comparison\nThresholds come from tune out-of-fold scores only. `auc` measures ranking of tail losers "
              "(0.5 = none). `net_bps_change` = bps the vetoes added (losses avoided minus wins given up)."),
        mo.ui.table(comparison.select(_cols), selection=None, page_size=40),
        mo.ui.table(_thr, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(comparison):
    _pts = comparison.filter(pl.col("strategy") != "baseline (no model)").with_columns(
        (100 * pl.col("losers_avoided") / pl.col("tail_total")).alias("losers_avoided_pct"),
        (100 * pl.col("winners_lost") / pl.col("winners_total")).alias("winners_lost_pct"),
    )
    _diag = alt.Chart(pl.DataFrame({"x": [0.0, 100.0], "y": [0.0, 100.0]})).mark_line(strokeDash=[4, 4], color=GREY).encode(x="x", y="y")
    _sc = alt.Chart(_pts).mark_point(filled=True, size=120).encode(
        x=alt.X("losers_avoided_pct:Q", title="Tail losers avoided (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("winners_lost_pct:Q", title="Winners lost (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("period:N", scale=PERIOD_SCALE), shape="strategy:N",
        tooltip=["strategy", "period", "losers_avoided", "tail_total", "winners_lost", "winners_total", "profit_factor", "win_rate"],
    )
    gate_scatter = mo.ui.altair_chart(_sc.properties(width="container", height=400), chart_selection="interval")
    mo.vstack([
        mo.md("### Losers avoided vs winners lost (fixed thresholds)\nA good gate sits bottom-right: it skips a larger share "
              "of tail losers than of winners. Drag a box around any points to compare those strategies below."),
        gate_scatter,
    ])
    return (gate_scatter,)


@app.cell(hide_code=True)
def _(gate_scatter):
    _v = gate_scatter.value
    if _v is None or not len(_v):
        _out = mo.md("_Drag a box on the chart to compare strategies._")
    else:
        _d = pl.DataFrame(_v)
        _out = mo.vstack([mo.md(f"**{_d.height} strategy-period points selected**"),
                          mo.ui.table(_d.select([c for c in ["strategy", "period", "trades", "win_rate", "profit_factor", "avg_net_bps",
                                                              "losers_avoided_pct", "winners_lost_pct", "auc"] if c in _d.columns]), page_size=10)])
    _out
    return


@app.cell(hide_code=True)
def _(decisions, frames):
    _rows = []
    for _d in decisions:
        for _per, _df in frames.items():
            if _df.height and _per in _d.score:
                _rows += lab.tradeoff_curve(_d.score[_per], _df, _d.label, _per)
    curves = pl.DataFrame(_rows)
    _base = alt.Chart(curves).encode(color=alt.Color("strategy:N", legend=alt.Legend(orient="bottom", columns=2)))
    _a = _base.mark_line(point=True).encode(
        x=alt.X("winners_lost_pct:Q", title="Winners lost (%)"), y=alt.Y("losers_avoided_pct:Q", title="Tail losers avoided (%)"),
        tooltip=["strategy", "period", "veto_rate", "losers_avoided_pct", "winners_lost_pct", "profit_factor"],
    ).properties(width=300, height=260).facet(column=alt.Column("period:N", title=None, sort=[TUNE_KEY, *HOLDOUTS]))
    _b = _base.mark_line().encode(
        x=alt.X("veto_rate:Q", title="Share of trades vetoed"), y=alt.Y("profit_factor:Q", title="Profit factor of kept trades"),
        tooltip=["strategy", "period", "veto_rate", "profit_factor"],
    ).properties(width=300, height=200).facet(column=alt.Column("period:N", title=None, sort=[TUNE_KEY, *HOLDOUTS]))
    mo.vstack([
        mo.md("### Trade-off curves (ranking quality)\nEach model's trades ranked by score within the period, vetoing the "
              "riskiest 0–60%. This shows how well a model ranks, not a tradable rule: the table above uses thresholds fixed on tune. "
              "Profit factor 1.5 is the target."),
        _a, _b,
    ])
    return


@app.cell(hide_code=True)
def _(combo_dec, decisions, frames):
    pick = mo.ui.dropdown(options=[d.label for d in decisions], value=(combo_dec or decisions[0]).label, label="Equity curve for")
    pick
    return (pick,)


@app.cell(hide_code=True)
def _(decisions, frames, pick):
    _d = next(d for d in decisions if d.label == pick.value)
    _t = lab.strategy_frames(frames, _d)
    _eq = pl.concat([
        _t.group_by("period", "trade_date").agg(pl.col("net_bps").sum().alias("day")).with_columns(pl.lit("baseline").alias("rule")),
        _t.filter("kept").group_by("period", "trade_date").agg(pl.col("net_bps").sum().alias("day")).with_columns(pl.lit("with model").alias("rule")),
    ]).sort("period", "rule", "trade_date").with_columns(pl.col("day").cum_sum().over("period", "rule").alias("cum_bps"))
    eq_trades = _t
    eq_charts = {
        _per: mo.ui.altair_chart(alt.Chart(_eq.filter(pl.col("period") == _per)).mark_line().encode(
            x=alt.X("trade_date:T", title=None), y=alt.Y("cum_bps:Q", title="Cumulative net bps"),
            color=alt.Color("rule:N", scale=alt.Scale(domain=["baseline", "with model"], range=[GREY, GREEN]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["period", "rule", "trade_date", "cum_bps"],
        ).properties(width="container", height=240, title=_per), chart_selection="interval")
        for _per in [TUNE_KEY, *HOLDOUTS] if _per in _eq["period"].unique().to_list()
    }
    mo.vstack([mo.md("Drag across any equity curve to list the trades in those days, kept and vetoed."),
               mo.hstack(list(eq_charts.values()), widths="equal", wrap=True)])
    return eq_charts, eq_trades


@app.cell(hide_code=True)
def _(eq_charts, eq_trades):
    _parts = []
    for _per, _c in eq_charts.items():
        _v = _c.value
        if _v is not None and len(_v):
            _days = pl.DataFrame(_v)["trade_date"].cast(pl.Date)
            _t = eq_trades.filter((pl.col("period") == _per) & pl.col("trade_date").is_between(_days.min(), _days.max()))
            _k = _t.filter(pl.col("kept"))
            _parts += [mo.md(f"**{_per}, {_days.min()} to {_days.max()}:** {_t.height} trades, {_k.height} kept · "
                             f"vetoed trades summed {_t.filter(~pl.col('kept'))['net_bps'].sum():+,.0f} bps"),
                       mo.ui.table(_t.select([c for c in ["trade_date", "symbol", "net_bps", "kept", "p_tail", "score"] if c in _t.columns]).sort("net_bps"), page_size=8)]
    mo.vstack(_parts) if _parts else mo.md("")
    return


@app.cell(hide_code=True)
def _(combo_dec, frames):
    mo.stop(combo_dec is None or combo_dec.stage_of_veto is None)
    _rows = []
    for _per, _st in combo_dec.stage_of_veto.items():
        _net = frames[_per]["net_bps"].to_numpy()
        for _s in range(int(_st.max()) + 1):
            _m = _st == _s
            _rows.append({"period": _per, "stage": "kept" if _s == 0 else f"vetoed at stage {_s}", "trades": int(_m.sum()),
                          "tail_losers": int((_net[_m] <= LOSER_BPS).sum()), "winners": int((_net[_m] > 0).sum()),
                          "sum_net_bps": round(float(_net[_m].sum()), 0)})
    mo.vstack([mo.md("### Sequential gates: what each stage removed"), mo.ui.table(pl.DataFrame(_rows), selection=None)])
    return


@app.cell(hide_code=True)
def _(fits, frames):
    _rows = []
    for _n, _f in fits.items():
        for _per, _df in frames.items():
            _p = _f.oof if _per == TUNE_KEY else _f.holdout.get(_per, np.array([]))
            if _df.height:
                _rows += lab.reliability(_p, _df["y_tail"].to_numpy(), _n, _per)
    _rel = pl.DataFrame(_rows)
    _diag = alt.Chart().mark_rule(strokeDash=[4, 4], color=GREY).encode(
        x=alt.datum(0.0), y=alt.datum(0.0), x2=alt.datum(0.6), y2=alt.datum(0.6))
    _pts = alt.Chart().mark_line(point=True).encode(
        x=alt.X("pred:Q", title="Calibrated P(tail)"), y=alt.Y("obs:Q", title="Observed tail rate"),
        color="strategy:N", tooltip=["strategy", "period", "pred", "obs", "n"],
    )
    mo.vstack([
        mo.md("### Calibration\nOn the dashed line a predicted 30% means 30% of those trades were tail losers. "
              "Holdout bins hold only 10–20 trades each, so expect noise."),
        alt.layer(_diag, _pts, data=_rel).properties(width=300, height=260).facet(column=alt.Column("period:N", title=None, sort=[TUNE_KEY, *HOLDOUTS])),
    ])
    return


@app.cell(hide_code=True)
def _(fits):
    explain_model = mo.ui.dropdown(options=list(fits), value=next(iter(fits)), label="Explain model")
    explain_on = mo.ui.dropdown(options=[TUNE_KEY, *HOLDOUTS], value=TUNE_KEY, label="on trades from")
    mo.vstack([mo.md("## Why: SHAP and importance"), mo.hstack([explain_model, explain_on], justify="start", gap=2)])
    return explain_model, explain_on


@app.cell(hide_code=True)
def _(explain_model, explain_on, fits, frames):
    _f, _df = fits[explain_model.value], frames[explain_on.value]
    _shap = lab.shap_table(_f, _df)
    if _shap is None:
        _imp = lab.permutation_importance(_f, _df) if _df["y_tail"].sum() > 0 else pl.DataFrame({"feature": [], "importance": []})
        _out = mo.vstack([
            mo.md(f"**{_f.name}** has no fast SHAP; showing permutation importance (drop in AUC when a feature is shuffled)."),
            alt.Chart(_imp.head(20)).mark_bar(color=BLUE).encode(x=alt.X("importance:Q", title="AUC drop"), y=alt.Y("feature:N", sort="-x", title=None)).properties(width=420, height=420),
        ])
    else:
        _mean = _shap.group_by("feature").agg(pl.col("shap").abs().mean().alias("mean_abs_shap")).sort("mean_abs_shap", descending=True)
        _top = _mean.head(15)["feature"].to_list()
        _bar = alt.Chart(_mean.head(15)).mark_bar(color=BLUE).encode(
            x=alt.X("mean_abs_shap:Q", title="mean |SHAP| (log-odds)"), y=alt.Y("feature:N", sort="-x", title=None)).properties(width=320, height=380)
        _bee = alt.Chart(_shap.filter(pl.col("feature").is_in(_top))).mark_circle(size=14, opacity=0.6).encode(
            x=alt.X("shap:Q", title="SHAP value (→ more likely tail loser)"),
            y=alt.Y("feature:N", sort=_top, title=None),
            yOffset=alt.YOffset("jitter:Q"),
            color=alt.Color("value_rank:Q", scale=alt.Scale(scheme="redblue", reverse=True), title="feature value (low→high)"),
            tooltip=["feature", "value", "shap"],
        ).transform_calculate(jitter="random()").properties(width=460, height=380)
        _out = mo.hstack([_bar, _bee], justify="start", gap=2)
    _out
    return


@app.cell(hide_code=True)
def _(frames):
    _rows = []
    for _per, _df in frames.items():
        _y = _df["y_tail"].to_numpy().astype(bool)
        for _c in F.FORECAST_FEATURES:
            _v = _df[_c].to_numpy().astype(float)
            _ok = np.isfinite(_v)
            _rows.append({"forecast stat": _c, "period": _per, "coverage_pct": round(100 * _ok.mean(), 1),
                          "auc_vs_tail": lab._auc(_v[_ok], _y[_ok]) if _ok.sum() > 10 else None,
                          "tail_median": float(np.nanmedian(_v[_y])) if (_y & _ok).any() else None,
                          "others_median": float(np.nanmedian(_v[~_y])) if (~_y & _ok).any() else None})
    _t = pl.DataFrame(_rows)
    mo.vstack([
        mo.md("## Layer 2: forecast upper tail (Chronos-Bolt, Kronos)\n"
              "Both models forecast the 20 minutes after entry from the pre-entry 1-minute window (09:30 to the arm bar; "
              "2–45 bars, no pre-market in the cached data). Higher values mean the forecast sees more room to run up. "
              "`auc_vs_tail` is the raw ranking power of each statistic, no fitting (0.5 = none). Forecasts are cached by "
              "`scripts/precompute_forecasts.py`; coverage below 100% means it has not finished. Use them as single-score "
              "models (model list) or add the forecast feature group to the tabular models."),
        mo.ui.table(_t, selection=None, page_size=30),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Heavier layers (status)
    | layer | model | status |
    |---|---|---|
    | 1 | LightGBM, CatBoost, TabPFN v2 (+ logistic, tree, single-feature rule) | working |
    | 2 | Chronos-Bolt small, Kronos-small: 20-minute forecast from the pre-entry window | working; scores cached in `data/models/` |
    | 3 | CLM-8B as an experimental gate (Q8 GGUF or 8-bit MLX) | optional, not started |
    | 4 | Jev cross-ticker ranking | needs Jero's Jev key |

    Add a model: write a class with `fit`, `predict_proba`, `explain` in `src/spike_lab/models/zoo.py` and register it in `REGISTRY`.
    Torch models run in `torch_worker.py` (separate process).
    """)
    return


if __name__ == "__main__":
    app.run()
