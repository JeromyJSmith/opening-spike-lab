# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "duckdb>=2.0.1",
#     "polars>=2.1.0",
#     "pyarrow",
#     "numpy",
#     "altair>=6.3.0",
#     "lib-pybroker>=2.0.3",
#     "scikit-learn",
# ]
# ///
"""Continuation-loser detector for the opening spike short.

Paper research only. Nothing here places orders.
Run:  uv run marimo edit notebooks/continuation_loser_detector.py
Logic: src/spike_lab/continuation.py (labels, features, rules), src/spike_lab/rules.py (replay).
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Continuation Loser Detector")

with app.setup:
    import sys
    from datetime import date
    from pathlib import Path

    import altair as alt
    alt.theme.enable("dark")
    import marimo as mo
    import numpy as np
    import polars as pl

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))

    from spike_lab import continuation as cl
    from spike_lab import engine
    from spike_lab import pybroker_check as pbc
    from spike_lab.rules import Params
    from spike_lab.universe import build_universe

    DATA = ROOT / "data" / "alpaca"
    TUNE = (date(2025, 7, 1), date(2026, 6, 30))  # prior 12 months: the only data used to choose anything
    HOLDOUT = (date(2026, 7, 1), date(2026, 9, 30))  # last full quarter, untouched
    FORWARD = (date(2026, 10, 1), date(2026, 11, 30))  # Oct-Nov 2026, also untouched
    PERIODS = {"tune": TUNE, "holdout Q3-26": HOLDOUT, "forward Oct-Nov-26": FORWARD}
    LABEL_COLORS = alt.Scale(domain=list(cl.LABELS), range=["#2f9e6e", "#8a8f98", "#d64545"])


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'continuation_loser_detector'
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
    # Continuation Loser Detector

    **Findings (default settings, Alpaca SIP bars Jul 2025 – Nov 2026; tune = Jul 25–Jun 26, holdout = Q3 26):**

    1. **The replay does not show the strategy Chris describes.** At 5%, $50+, cover 20 min, 5 bps: tune 368 trades,
       48.1% win, PF 0.91, −8.2 bps avg; holdout Q3-26 136 trades, 43.4% win, PF 0.75; Oct–Nov-26 71 trades, PF 0.86.
       Winners average +164 bps (median +125), not +30; losers average −169. Losses are spread out, not a 1–2-a-day tail:
       119 of 368 tune trades are continuation losers (≤ −150 bps, or a new high after entry and ≤ −75 bps).
       PyBroker reproduces the holdout replay exactly (136/136 trades, same entry minute, 1.75 bps mean difference).
    2. **Nothing known at entry separates losers.** The best single entry feature has AUC 0.57 (distance from entry to
       high of day; others ≤ 0.57). A depth-3 tree reaches 51% precision / 62% recall in-sample but 33% / 37% on the
       holdout, and every holdout permutation importance is ≤ 0. The best tune skip rule (`entry_vs_hod_bps > 134`:
       PF 0.91 → 1.22) falls to PF 0.73 on the holdout.
    3. **The first minutes do carry signal, but bailing doesn't pay.** Return at minute 5 has AUC 0.19 (0.81 inverted)
       and MAE at minute 3 has 0.73. A minute-3 bail at −75 bps avoids 53 of 119 tune losers but costs 13 winners and the
       spread back to entry: PF 0.98 tune, 0.79 holdout. Small early losses keep mean-reverting often enough that
       cutting them costs as much as the tail.
    4. **No rule tested reaches 60% win / PF ≥ 1.5 on any period.** Before more filter mining, reconcile the replay with
       Chris's own trade list. The gap between +30 bps winners and the +164 bps / 48% win replay suggests a different
       entry (protected limit, 30-second bars, or another arm rule). Use the controls to try 4%, other holds or labels.
        5. **backtest-expert stress test (section 8): REJECT / Abandon.** Across 4%/5% × 10–30 min holds × 5–10 bps
           slippage (90 cells), none reaches 60% win / PF ≥ 1.5. No (trigger, hold) pair is profitable in every period:
           PF stays 0.78–0.98 on the tune year for all ten. Only 5 of 17 months were positive. The only PF ≥ 1.5 cells are 4% with a
           10–15 min hold in Oct–Nov 2026 alone (PF 1.58–1.59, 55–57% win), the same settings ran PF 0.84–0.86 in the tune year,
           so that is one regime, not an edge. The skill's gate returns REJECT for the baseline and the best bail rule
           (negative expectancy; also only ~1.4 years of SIP history, under its 5-year minimum).

    ---
    **Setup replayed**: US stocks $\ge$ \$50, ATR14 $\ge$ 3%, 20-day ADDV $\ge$ \$100M (tiers \$100M–\$1B and >\$1B),
    open 0.25–1.0 ATR above prior close, high reaches open × (1 + 4 or 5%) before 10:00 ET (8:00 MT).
    The first red 1-minute bar arms; the short fills at the next bar's open; cover 20 minutes later; 5 bps slippage.
    Every feature below uses only bars before the fill (plus daily history before the day).
    Thresholds and rules are chosen on the **tune** year only; the holdout and forward periods are scored, never fitted.
    """)
    return


@app.cell(hide_code=True)
def _():
    spike = mo.ui.dropdown(options={"4%": 0.04, "5%": 0.05}, value="5%", label="Run-up trigger")
    min_price = mo.ui.number(25, 200, value=50, step=5, label="Min price $")
    tiers = mo.ui.multiselect(options=["100M-1B", "1B+"], value=["100M-1B", "1B+"], label="ADDV tiers")
    hold = mo.ui.slider(5, 40, value=20, step=1, label="Cover after (min)", show_value=True)
    slip = mo.ui.number(0, 30, value=5, step=0.5, label="Slippage bps (round trip)")
    loser_bps = mo.ui.slider(-400, -50, value=-150, step=10, label="Loser if net ≤ (bps)", show_value=True)
    hod_counts = mo.ui.checkbox(value=True, label="…or new high of day after entry and net ≤")
    hod_loser_bps = mo.ui.slider(-200, 0, value=-75, step=5, label="HOD-loser cutoff (bps)", show_value=True)
    scratch_bps = mo.ui.slider(0, 50, value=10, step=5, label="Clean fade if net ≥ (bps)", show_value=True)
    mo.vstack([
        mo.md("### Replay and labels"),
        mo.hstack([spike, min_price, tiers, hold, slip], justify="start"),
        mo.hstack([loser_bps, hod_counts, hod_loser_bps, scratch_bps], justify="start"),
    ])
    return (
        hod_counts,
        hod_loser_bps,
        hold,
        loser_bps,
        min_price,
        scratch_bps,
        slip,
        spike,
        tiers,
    )


@app.cell(hide_code=True)
def _(min_price):
    with mo.status.spinner(title="Loading cached Alpaca bars and building the universe (DuckDB)"):
        daily = pl.read_parquet(DATA / "daily.parquet")
        intraday = pl.read_parquet(DATA / "intraday.parquet")
        bench = pl.read_parquet(DATA / "benchmarks.parquet") if (DATA / "benchmarks.parquet").exists() else None
        universe = build_universe(daily, min_price=float(min_price.value))
    mo.callout(
        mo.md(
            f"Cached bars: daily {daily['trade_date'].min()} to {daily['trade_date'].max()}, "
            f"{intraday.height:,} one-minute morning bars, SPY/QQQ benchmark bars: "
            f"{'yes' if bench is not None else 'missing - run scripts/fetch_benchmarks.py'}. "
            "Survivorship and borrow caveats: data/alpaca/PROVENANCE.md."
        ),
        kind="info" if bench is not None else "warn",
    )
    return bench, daily, intraday, universe


@app.cell(hide_code=True)
def _(bench, daily, hold, intraday, slip, spike, tiers, universe):
    params = Params(spike_pct=spike.value, hold_minutes=hold.value, slippage_bps=float(slip.value))
    with mo.status.spinner(title="Replaying every setup through the state machine"):
        run = engine.run(universe, intraday, params, TUNE[0], FORWARD[1], tiers=tuple(tiers.value))
        features = cl.in_trade_features(cl.entry_features(run.trades, intraday, daily, bench), intraday)
    period_expr = pl.when(pl.col("trade_date") <= TUNE[1]).then(pl.lit("tune")).when(
        pl.col("trade_date") <= HOLDOUT[1]).then(pl.lit("holdout Q3-26")).otherwise(pl.lit("forward Oct-Nov-26"))
    features = features.with_columns(period_expr.alias("period"))
    return features, params, run


@app.cell(hide_code=True)
def _(features, hod_counts, hod_loser_bps, loser_bps, scratch_bps):
    ds = cl.label_trades(features, float(loser_bps.value), float(scratch_bps.value), hod_counts.value, float(hod_loser_bps.value))
    tune = ds.filter(pl.col("period") == "tune")
    holdout = ds.filter(pl.col("period") == "holdout Q3-26")
    forward = ds.filter(pl.col("period") == "forward Oct-Nov-26")
    return ds, forward, holdout, tune


@app.cell(hide_code=True)
def _(ds):
    _counts = (
        ds.group_by(["period", "label"])
        .agg(pl.len().alias("trades"), pl.col("net_bps").mean().round(1).alias("avg_net_bps"),
             pl.col("net_bps").sum().round(0).alias("total_bps"))
        .sort(["period", "label"])
    )
    _base = pl.DataFrame([{"period": k, **cl.score(ds.filter(pl.col("period") == k)["net_bps"]),
                           "days": ds.filter(pl.col("period") == k)["trade_date"].n_unique()} for k in PERIODS])
    mo.vstack([
        mo.md("## 1. Replay and labels\nBaseline (take every armed trade) and the label mix per period."),
        mo.hstack([mo.ui.table(_base, selection=None, label="Baseline"), mo.ui.table(_counts, selection=None, label="Labels")]),
    ])
    return


@app.cell(hide_code=True)
def _():
    feat_pick = mo.ui.dropdown(options=cl.ENTRY_FEATURES + cl.early_feature_names(3), value="entry_vs_hod_bps", label="Feature")
    clip_q = mo.ui.slider(0.9, 1.0, value=0.98, step=0.01, label="Clip at quantile", show_value=True)
    mo.vstack([mo.md("## 2. Which features separate losers from winners (tune year only)"), mo.hstack([feat_pick, clip_q], justify="start")])
    return clip_q, feat_pick


@app.cell(hide_code=True)
def _(clip_q, feat_pick, tune):
    _f = feat_pick.value
    _hi = tune[_f].quantile(clip_q.value)
    _lo = tune[_f].quantile(1 - clip_q.value)
    _d = tune.select("label", pl.col(_f).clip(_lo, _hi).alias("value")).drop_nulls()
    _box = alt.Chart(_d).mark_boxplot(extent="min-max", size=40).encode(
        x=alt.X("label:N", sort=list(cl.LABELS), title=None), y=alt.Y("value:Q", title=_f),
        color=alt.Color("label:N", scale=LABEL_COLORS, legend=None),
    ).properties(width=320, height=300)
    _hist = alt.Chart(_d).transform_density("value", groupby=["label"], as_=["value", "density"]).mark_line().encode(
        x=alt.X("value:Q", title=_f), y="density:Q", color=alt.Color("label:N", scale=LABEL_COLORS),
    ).properties(width=420, height=300)
    _pts = tune.select(*[c for c in ["symbol", "trade_date", "label", "net_bps"] if c in tune.columns],
                       pl.col(_f).clip(_lo, _hi).alias("value")).drop_nulls("value")
    feat_scatter = mo.ui.altair_chart(alt.Chart(_pts).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X("value:Q", title=_f), y=alt.Y("net_bps:Q", title="Short P&L (bps)"),
        color=alt.Color("label:N", scale=LABEL_COLORS, legend=alt.Legend(orient="top", title=None)),
        tooltip=[c for c in ["symbol", "trade_date", "label", "net_bps", "value"] if c in _pts.columns],
    ).properties(width="container", height=320, title=f"Every tune trade: {_f} vs P&L (drag a box to list them)"),
        chart_selection="interval")
    mo.vstack([mo.hstack([_box, _hist], justify="start"), feat_scatter])
    return (feat_scatter,)


@app.cell(hide_code=True)
def _(feat_scatter):
    _v = feat_scatter.value
    if _v is None or not len(_v):
        _out = mo.md("_Drag a box on the scatter to list those trades._")
    else:
        _d = pl.DataFrame(_v)
        _out = mo.vstack([mo.md(f"**{_d.height} trades** · win rate {(_d['net_bps'] > 0).mean() * 100:.0f}% · avg {_d['net_bps'].mean():+.0f} bps · "
                                + ", ".join(f"{n} {l}" for l, n in _d.group_by("label").len().iter_rows())),
                          mo.ui.table(_d.sort("net_bps"), page_size=10)])
    _out
    return


@app.cell(hide_code=True)
def _(tune):
    from sklearn.metrics import roc_auc_score

    _rows = []
    _y = tune["is_loser"].to_numpy()
    for _f in cl.ENTRY_FEATURES + [c for n in (1, 2, 3, 5) for c in cl.early_feature_names(n)]:
        _x = tune[_f].fill_null(tune[_f].median()).to_numpy().astype(float)
        _auc = roc_auc_score(_y, _x) if 0 < _y.sum() < len(_y) else float("nan")
        _rows.append({
            "feature": _f, "known_at": "entry" if _f in cl.ENTRY_FEATURES else f"minute {_f.split('_')[-1]}",
            "loser_median": round(float(tune.filter(pl.col("is_loser"))[_f].median()), 3),
            "clean_fade_median": round(float(tune.filter(pl.col("label") == "clean_fade")[_f].median()), 3),
            "AUC": round(_auc, 3), "separation": round(abs(_auc - 0.5) * 2, 3),
        })
    separation = pl.DataFrame(_rows).sort("separation", descending=True)
    mo.vstack([
        mo.md("Univariate separation: AUC of each feature alone for flagging a continuation loser "
              "(0.5 = coin flip; `separation` = |AUC − 0.5| × 2)."),
        mo.ui.table(separation, selection=None, page_size=12),
    ])
    return


@app.cell(hide_code=True)
def _():
    model_kind = mo.ui.dropdown(options=["decision tree", "gradient boosting (LightGBM-style)"], value="decision tree", label="Model")
    depth = mo.ui.slider(1, 6, value=3, step=1, label="Tree depth", show_value=True)
    minute_n = mo.ui.slider(0, cl.EARLY_MAX, value=0, step=1, label="Use in-trade features through minute (0 = entry only)", show_value=True)
    flag_at = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, label="Flag as loser if p ≥", show_value=True)
    mo.vstack([mo.md("## 3. Model: can a loser be flagged? (fit on tune, scored on holdout)"),
               mo.hstack([model_kind, depth, minute_n, flag_at], justify="start")])
    return depth, flag_at, minute_n, model_kind


@app.cell(hide_code=True)
def _(depth, flag_at, forward, holdout, minute_n, model_kind, tune):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.tree import DecisionTreeClassifier, export_text

    model_cols = cl.ENTRY_FEATURES + [c for n in range(1, minute_n.value + 1) for c in cl.early_feature_names(n)]

    def _xy(df):
        return df.select([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in model_cols]).to_numpy(), df["is_loser"].to_numpy()

    _Xt, _yt = _xy(tune)
    if model_kind.value == "decision tree":
        model = DecisionTreeClassifier(max_depth=depth.value, min_samples_leaf=15, class_weight="balanced", random_state=0)
    else:
        model = HistGradientBoostingClassifier(max_depth=depth.value, max_iter=150, learning_rate=0.05,
                                               min_samples_leaf=15, class_weight="balanced", random_state=0)
    model.fit(_Xt, _yt)
    _rows = []
    for _name, _df in (("tune (in-sample)", tune), ("holdout Q3-26", holdout), ("forward Oct-Nov-26", forward)):
        _X, _y = _xy(_df)
        _p = model.predict_proba(_X)[:, 1] if len(_y) else np.array([])
        _flag = pl.Series(_p >= flag_at.value)
        _rows.append({"period": _name, "trades": len(_y), "losers": int(_y.sum()),
                      **cl.flag_precision_recall(pl.Series(_y.astype(bool)), _flag)})
    model_pr = pl.DataFrame(_rows)
    _Xh, _yh = _xy(holdout)
    _imp = permutation_importance(model, _Xh, _yh, scoring="roc_auc", n_repeats=10, random_state=0)
    importances = pl.DataFrame({"feature": model_cols, "holdout_perm_importance_auc": _imp.importances_mean.round(4)}).sort(
        "holdout_perm_importance_auc", descending=True)
    _tree_txt = export_text(model, feature_names=model_cols, decimals=2) if model_kind.value == "decision tree" else ""
    mo.vstack([
        mo.md("Precision = share of flagged trades that really were continuation losers. Recall = share of losers flagged. "
              "Importance = drop in holdout AUC when a feature is shuffled (out-of-sample, so it can be negative)."),
        mo.hstack([mo.ui.table(model_pr, selection=None), mo.ui.table(importances.head(12), selection=None)]),
        mo.accordion({"Tree rules (fit on tune)": mo.md(f"```\n{_tree_txt}\n```")}) if _tree_txt else mo.md(""),
    ])
    return


@app.cell(hide_code=True)
def _():
    min_trades = mo.ui.slider(50, 300, value=150, step=10, label="Rule must keep ≥ N tune trades", show_value=True)
    mo.vstack([mo.md("## 4. Candidate rules: losers avoided vs winners lost\n"
                     "Each rule is one condition: **skip** at entry, or **bail at minute N** (cover at that bar's close). "
                     "Skip thresholds are 10th/20th/80th/90th percentiles of the tune year. Ranked on tune; the holdout "
                     "columns are what the same frozen rule did afterwards."), min_trades])
    return (min_trades,)


@app.cell(hide_code=True)
def _(forward, holdout, min_trades, slip, tune):
    _rules = cl.candidate_rules(tune)
    _t = cl.scan_rules(tune, _rules, float(slip.value))
    _h = cl.scan_rules(holdout, _rules, float(slip.value))
    _f = cl.scan_rules(forward, _rules, float(slip.value))
    _keep = ["rule", "trades", "win_rate", "profit_factor", "avg_net_bps", "total_bps", "losers_avoided", "winners_lost"]
    rule_table = (
        _t.select(["rule", "losers", "losers_avoided", "winners", "winners_lost", "trades", "win_rate", "profit_factor",
                   "avg_net_bps", "total_bps", "bps_saved_on_losers", "bps_given_up_on_winners"])
        .join(_h.select(_keep).rename({c: f"hold_{c}" for c in _keep if c != "rule"}), on="rule")
        .join(_f.select(_keep).rename({c: f"fwd_{c}" for c in _keep if c != "rule"}), on="rule")
        .filter((pl.col("trades") >= min_trades.value) | (pl.col("rule").str.starts_with("baseline")))
        .sort("profit_factor", descending=True, nulls_last=True)
    )
    mo.ui.table(rule_table, selection=None, page_size=15, label="Ranked by tune profit factor")
    return


@app.cell(hide_code=True)
def _():
    skip_feat = mo.ui.dropdown(options=["(none)"] + cl.ENTRY_FEATURES, value="entry_vs_hod_bps", label="Skip if")
    skip_op = mo.ui.dropdown(options=[">", "<"], value=">", label="")
    skip_thr = mo.ui.number(-1000, 1000, value=134, step=0.1, label="threshold")
    bail_n = mo.ui.slider(0, cl.EARLY_MAX, value=0, step=1, label="Bail at minute (0 = off)", show_value=True)
    bail_kind = mo.ui.dropdown(options=["MAE ≥ bps", "return ≤ −bps", "new HOD", "new HOD and return < 0", "all bars green"],
                               value="MAE ≥ bps", label="if")
    bail_thr = mo.ui.number(0, 1000, value=75, step=5, label="bps")
    mo.vstack([mo.md("## 5. Try your own rule (live)"),
               mo.hstack([skip_feat, skip_op, skip_thr], justify="start"),
               mo.hstack([bail_n, bail_kind, bail_thr], justify="start")])
    return bail_kind, bail_n, bail_thr, skip_feat, skip_op, skip_thr


@app.cell(hide_code=True)
def _(bail_kind, bail_n, bail_thr, ds, skip_feat, skip_op, skip_thr, slip):
    _skip = None
    if skip_feat.value != "(none)":
        _c = pl.col(skip_feat.value)
        _skip = _c > skip_thr.value if skip_op.value == ">" else _c < skip_thr.value
    _n = bail_n.value or None
    _bail = None
    if _n:
        _bail = {
            "MAE ≥ bps": pl.col(f"mae_{_n}") >= bail_thr.value,
            "return ≤ −bps": pl.col(f"ret_{_n}") <= -bail_thr.value,
            "new HOD": pl.col(f"newhod_{_n}") == 1,
            "new HOD and return < 0": (pl.col(f"newhod_{_n}") == 1) & (pl.col(f"ret_{_n}") < 0),
            "all bars green": pl.col(f"greens_{_n}") >= _n,
        }[bail_kind.value]
    custom_rule = cl.Rule("custom", skip=_skip, bail_n=_n, bail=_bail)
    custom_scores = pl.DataFrame([
        {"period": k, **cl.evaluate_rule(ds.filter(pl.col("period") == k), custom_rule, float(slip.value))} for k in PERIODS
    ]).drop("rule")
    _meets = custom_scores.with_columns(
        ((pl.col("win_rate") >= 60) & (pl.col("profit_factor") >= 1.5)).alias("meets_60pct_PF1.5")
    )
    _applied = cl.apply_rule(ds, custom_rule, float(slip.value))
    _eq = (
        _applied.group_by(["period", "trade_date"])
        .agg(pl.col("net_bps").sum().alias("baseline"), pl.col("rule_net_bps").fill_null(0).sum().alias("with rule"))
        .sort("trade_date")
        .with_columns(pl.col("baseline").cum_sum().over("period"), pl.col("with rule").cum_sum().over("period"))
        .unpivot(index=["period", "trade_date"], variable_name="variant", value_name="cum_bps")
    )
    _chart = alt.Chart(_eq).mark_line().encode(
        x="trade_date:T", y=alt.Y("cum_bps:Q", title="Cumulative net bps (equal notional)"),
        color="variant:N", strokeDash="variant:N", tooltip=["trade_date:T", "variant", alt.Tooltip("cum_bps:Q", format=",.0f")],
    ).properties(width=300, height=220).facet(column=alt.Column("period:N", sort=list(PERIODS)))
    mo.vstack([mo.ui.table(_meets, selection=None), _chart])
    return


@app.cell(hide_code=True)
def _(holdout):
    worst = holdout.sort("net_bps").head(15).select(
        "symbol", "trade_date", "tier", "label", pl.col("net_bps").round(0), "new_hod_in_trade",
        pl.col("mae_3").round(0), pl.col("ret_3").round(0), pl.col("entry_vs_hod_bps").round(0),
        pl.col("runup_pct_at_arm").round(2), pl.col("vwap_z").round(2), pl.col("rvol14").round(1),
        pl.col("spy_move_bps").round(0), "entry_minute",
    )
    mo.vstack([mo.md("## 6. Holdout's 15 worst trades and what they looked like at entry / minute 3"),
               mo.ui.table(worst, selection=None)])
    return


@app.cell(hide_code=True)
def _(intraday, params, run, tiers, universe):
    with mo.status.spinner(title="PyBroker cross-check of the baseline on the holdout quarter"):
        _sig = pbc.arm_signals(universe, intraday, spike_pct=params.spike_pct, start=HOLDOUT[0], end=HOLDOUT[1],
                               tiers=tuple(tiers.value))
        _pb = pbc.run_pybroker(_sig, hold=params.hold_minutes, slippage_bps=params.slippage_bps)
        _eng = run.trades.filter(pl.col("trade_date").is_between(*HOLDOUT))
        pyb = pbc.reconcile(_eng, _pb)
    mo.vstack([mo.md("## 7. Independent replay check (PyBroker)\nPyBroker re-derives the arm signal in Polars and does its own "
                     "fills (next-bar open, cover after the hold). Matching counts and bps mean the labels above rest on a correct replay."),
               mo.ui.table(pl.DataFrame([pyb]), selection=None)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8. Stress tests (backtest-expert skill)
    Following the backtest-expert method: *find what breaks the least*. Grid over trigger (4%/5%) × hold (10–30 min) ×
    slippage (5 / 7.5 / 10 bps, i.e. 1×, 1.5×, 2×), every period, then month-by-month, then the skill's own
    `evaluate_backtest.py` gate (decision + blocking reasons) on the baseline and the best early-bail rule.
    """)
    return


@app.cell(hide_code=True)
def _(intraday, tiers, universe):
    _period = (pl.when(pl.col("trade_date") <= TUNE[1]).then(pl.lit("tune"))
               .when(pl.col("trade_date") <= HOLDOUT[1]).then(pl.lit("holdout Q3-26"))
               .otherwise(pl.lit("forward Oct-Nov-26")))
    _rows = []
    with mo.status.spinner(title="Stress grid: 10 replays x 3 slippage levels"):
        for _sp in (0.04, 0.05):
            for _h in (10, 15, 20, 25, 30):
                _r = engine.run(universe, intraday, Params(spike_pct=_sp, hold_minutes=_h, slippage_bps=0.0),
                                TUNE[0], FORWARD[1], tiers=tuple(tiers.value))
                _t = _r.trades.with_columns(_period.alias("period"))
                for _s in (5.0, 7.5, 10.0):
                    for _p in PERIODS:
                        _rows.append({"trigger": f"{_sp:.0%}", "hold_min": _h, "slippage_bps": _s, "period": _p,
                                      **cl.score(_t.filter(pl.col("period") == _p)["gross_bps"] - _s)})
    stress_grid = pl.DataFrame(_rows)
    return (stress_grid,)


@app.cell(hide_code=True)
def _(stress_grid):
    _heat = alt.Chart(stress_grid).mark_rect().encode(
        x=alt.X("hold_min:O", title="Hold (min)"), y=alt.Y("trigger:N", title=None),
        color=alt.Color("profit_factor:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0.5, 1.5]), title="PF"),
        tooltip=["trigger", "hold_min", "slippage_bps", "period", "trades", "win_rate", "profit_factor", "avg_net_bps"],
    ).properties(width=170, height=70)
    _text = _heat.mark_text(fontSize=11).encode(text=alt.Text("profit_factor:Q", format=".2f"), color=alt.value("black"))
    _meets = stress_grid.filter((pl.col("win_rate") >= 60) & (pl.col("profit_factor") >= 1.5))
    mo.vstack([
        alt.layer(_heat, _text, data=stress_grid).facet(row=alt.Row("slippage_bps:O", title="Slippage bps"),
                                                       column=alt.Column("period:N", sort=list(PERIODS), title=None)),
        mo.md(f"**Cells meeting 60% win / PF ≥ 1.5:** {_meets.height} of {stress_grid.height}. "
              f"Cells with PF > 1 in *every* period at 5 bps: "
              f"{stress_grid.filter(pl.col('slippage_bps') == 5).group_by(['trigger','hold_min']).agg((pl.col('profit_factor') > 1).all().alias('ok')).filter(pl.col('ok')).height} of 10."),
    ])
    return


@app.cell(hide_code=True)
def _(ds):
    monthly = (
        ds.with_columns(pl.col("trade_date").dt.truncate("1mo").alias("month"))
        .group_by("month").agg(pl.len().alias("trades"), pl.col("net_bps").sum().round(0).alias("total_bps"),
                               ((pl.col("net_bps") > 0).mean() * 100).round(1).alias("win_rate"),
                               pl.col("is_loser").sum().alias("losers"))
        .sort("month")
    )
    _bars = alt.Chart(monthly).mark_bar().encode(
        x=alt.X("month:T", title=None), y=alt.Y("total_bps:Q", title="Net bps per month (baseline)"),
        color=alt.condition("datum.total_bps > 0", alt.value("#2f9e6e"), alt.value("#d64545")),
        tooltip=["month:T", "trades", "total_bps", "win_rate", "losers"],
    ).properties(width="container", height=220)
    month_bars = mo.ui.altair_chart(_bars, chart_selection="interval")
    mo.vstack([mo.md(f"**Time robustness:** {int((monthly['total_bps'] > 0).sum())} of {monthly.height} months positive. "
                     "Drag across months to list their trades."), month_bars])
    return (month_bars,)


@app.cell(hide_code=True)
def _(ds, month_bars):
    _v = month_bars.value
    if _v is None or not len(_v):
        _out = mo.md("")
    else:
        _m = pl.DataFrame(_v)["month"].cast(pl.Date)
        _t = ds.filter(pl.col("trade_date").dt.truncate("1mo").is_between(_m.min(), _m.max()))
        _out = mo.vstack([mo.md(f"**{_m.min():%b %Y} to {_m.max():%b %Y}: {_t.height} trades** · win rate {(_t['net_bps'] > 0).mean() * 100:.0f}% · "
                                f"losers {int(_t['is_loser'].sum())}"),
                          mo.ui.table(_t.select([c for c in ["trade_date", "symbol", "period", "net_bps", "label", "is_loser"] if c in _t.columns]).sort("net_bps"), page_size=10)])
    _out
    return


@app.cell(hide_code=True)
def _(ds, holdout, slip):
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("evaluate_backtest", ROOT.parent / ".claude/skills/backtest-expert/scripts/evaluate_backtest.py")
    bt_eval = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(bt_eval)

    def bt_inputs(net: pl.Series, dates: pl.Series, n_params: int) -> dict:
        _w, _l = net.filter(net > 0), net.filter(net <= 0)
        _eq = pl.DataFrame({"d": dates, "n": net}).group_by("d").agg(pl.col("n").sum()).sort("d")["n"].cum_sum()
        return dict(total_trades=net.len(), win_rate=float((net > 0).mean() * 100),
                    avg_win_pct=float(_w.mean()) / 100, avg_loss_pct=-float(_l.mean()) / 100,
                    max_drawdown_pct=-float(min(0.0, (_eq - _eq.cum_max()).min())) / 100,
                    years_tested=1, num_parameters=n_params, slippage_tested=True)

    _bail = cl.apply_rule(ds, cl.Rule("bail m3 if ret <= -75bps", bail_n=3, bail=pl.col("ret_3") <= -75), float(slip.value))
    _cases = {
        "baseline, all periods": (ds["net_bps"], ds["trade_date"], 6),
        "baseline, holdout Q3-26": (holdout["net_bps"], holdout["trade_date"], 6),
        "bail m3 ret<=-75, all periods": (_bail["rule_net_bps"], _bail["trade_date"], 8),
    }
    _out = []
    for _name, (_n, _d, _k) in _cases.items():
        _res = bt_eval.evaluate(**bt_inputs(_n, _d, _k))
        _out.append({"case": _name, "decision": _res["decision"], "verdict": _res["verdict"],
                     "quality_score": _res.get("total_score", _res.get("quality_score")),
                     "blocking_reasons": ", ".join(_res["blocking_reasons"]),
                     **{k: round(v, 2) if isinstance(v, float) else v for k, v in bt_inputs(_n, _d, _k).items()
                        if k in ("total_trades", "win_rate", "avg_win_pct", "avg_loss_pct", "max_drawdown_pct")}})
    backtest_verdicts = pl.DataFrame(_out)
    mo.vstack([mo.md("**backtest-expert verdict** (drawdown = cumulative daily bps at 1× notional per trade; "
                     "years_tested = 1 because the cached SIP history starts May 2025)."),
               mo.ui.table(backtest_verdicts, selection=None)])
    return


if __name__ == "__main__":
    app.run()
