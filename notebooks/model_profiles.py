# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "polars>=2.1.0",
#     "altair>=6.3.0",
#     "numpy",
#     "wigglystuff>=0.6.0",
# ]
# ///
"""Model profiles: what each model is good at and where it could fit in the spike-short strategy.

Run:  uv run marimo run notebooks/model_profiles.py   (dark theme comes from pyproject [tool.marimo.display])
Reads saved results only (data/reports/*.csv, data/models/forecast_*.parquet). It trains nothing.
The strategy is a SHORT: a trade makes money when the opening spike fades. net_bps > 0 = the fade worked.
Paper research only.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Model profiles")

with app.setup:
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from wigglystuff import ParallelCoordinates, TangleSlider

    alt.theme.enable("dark")
    ROOT = Path(__file__).resolve().parents[1]
    REP = ROOT / "data" / "reports"
    MOD = ROOT / "data" / "models"
    import io
    import sys

    IN_BROWSER = sys.platform == "emscripten"  # WASM build: data comes from the public/ snapshot next to the page

    def data_file(local: Path):
        """Local runs read data/ directly; the browser (WASM) build fetches the snapshot copied into public/."""
        if not IN_BROWSER:
            return local
        import urllib.request
        url = str(mo.notebook_location() / "public" / "model_profiles" / local.name)
        return io.BytesIO(urllib.request.urlopen(url).read())

    def read_table(local: Path) -> pl.DataFrame:
        """Parquet locally; in the browser build parquet needs pyarrow, so a CSV snapshot is shipped instead."""
        if local.suffix == ".parquet" and not IN_BROWSER:
            return pl.read_parquet(local)
        return pl.read_csv(data_file(local.with_suffix(".csv")), try_parse_dates=True)
    FADE, RUN, GREY, ACC = "#5cc2a2", "#ef7b66", "#7d8b86", "#e3a948"
    TAIL_BPS = -200

    def card(title: str, body: str, chart) -> mo.Html:
        return mo.vstack([mo.md(f"## {title}"), mo.hstack([mo.md(body), chart], widths=[1, 1.15], gap=2)], gap=0.6)


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'model_profiles'
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
    # Model profiles for the spike short

    **The trade.** A liquid stock gaps up 0.25–1.0 ATR, runs **+5%** from the open, and we **short** on the first
    red 1-minute bar, covering 20 minutes later. We make money when the spike **fades**. We lose big when it
    **keeps running** (a *continuation loser*, here any trade at or below **−200 bps**).

    **The job for a model** is almost always the same question: *is this one of the 1-2 shorts a day that will keep
    running?* Different models answer it from different inputs:

    | Model | Reads | Answers | Speed | Status here |
    |---|---|---|---|---|
    | LightGBM | numbers about the trade (51 features) | P(tail loser) | µs | tested |
    | CatBoost | same, handles categories natively | P(tail loser) | µs | tested |
    | TabPFN v2 | same, no training loop (in-context) | P(tail loser) | ms–s | tested |
    | Chronos-Bolt | the price path so far | future price paths | ~10 ms GPU | tested |
    | Kronos | OHLCV candles | sampled candle paths | ~0.3 s CPU | tested |
    | CLM-8B | text (news, filings) + a list of options | nearest option | ~100 ms | not run yet |
    | Jev | any structured state + typed questions | a pick, a score, a yes/no | ~0.5–1 s API | demo'd (Jev notebook) |

    Every chart below uses saved results from this lab unless it says **illustration**.
    """)
    return


@app.cell(hide_code=True)
def _():
    CMP = pl.read_csv(data_file(REP / "model_lab_comparison_layer1_2.csv"))
    TRADES = pl.read_csv(data_file(REP / "loop_round1_trades.csv"), try_parse_dates=True).with_columns(
        outcome=pl.when(pl.col("net_bps") <= TAIL_BPS).then(pl.lit("continuation loser (≤ −200)"))
        .when(pl.col("net_bps") < 0).then(pl.lit("small loss"))
        .otherwise(pl.lit("faded (win)"))
    )
    KEYS = ["symbol", "trade_date", "entry_ts"]
    FC = TRADES.join(read_table(MOD / "forecast_chronos.parquet"), on=KEYS).join(
        read_table(MOD / "forecast_kronos.parquet"), on=KEYS)
    OUT_SCALE = alt.Scale(domain=["faded (win)", "small loss", "continuation loser (≤ −200)"], range=[FADE, GREY, RUN])
    PERIODS = ["tune (OOF)", "validate (Q3 26)", "forward (Oct-Nov 26)"]
    return CMP, FC, OUT_SCALE, PERIODS


@app.cell(hide_code=True)
def _():
    SCORES = {
        "Kronos: share of 16 paths that run +200 bps": ("kronos_p_up_tail", 1),
        "Kronos: q90 highest price ahead": ("kronos_q90_max_bps", 1),
        "Chronos-Bolt: q90 highest price ahead": ("chronos_q90_max_bps", 1),
        "Run speed (faster to +5% = riskier)": ("minutes_to_trigger", -1),
        "Entry distance below the high of day": ("entry_vs_hod_bps", 1),
        "Red-bar volume (lighter = riskier)": ("red_vol_ratio", -1),
    }
    score_pick = mo.ui.dropdown(list(SCORES), value=list(SCORES)[0], label="Risk score")
    veto = mo.ui.anywidget(TangleSlider(amount=20, min_value=0, max_value=60, step=5, suffix="%", digits=0))
    return SCORES, score_pick, veto


@app.cell(hide_code=True)
def _(score_pick, veto):
    mo.vstack([
        mo.md("""## Try it: veto lab

Pick a risk score, then drag the number to change how many of the riskiest shorts we skip. The cut-off is set
on the **tune year only** and then applied unchanged to Q3 2026 and Oct-Nov 2026, the same discipline Mark uses.
Everything below recomputes from the 575 real trades."""),
        score_pick,
        mo.md(f"### Skip the riskiest {veto} of shorts"),
    ])
    return


@app.cell(hide_code=True)
def _(FC, SCORES, score_pick, veto):
    _col, _sign = SCORES[score_pick.value]
    _pct = float(veto.value.get("amount") or 0)
    _risk = FC.with_columns(risk=pl.col(_col) * _sign)
    _thr = _risk.filter(pl.col("period") == "tune")["risk"].quantile(1 - _pct / 100) if _pct > 0 else float("inf")
    LAB = _risk.with_columns(kept=pl.col("risk").is_null() | (pl.col("risk") <= _thr))

    def _stats(df):
        _w, _l = df.filter(pl.col("net_bps") > 0)["net_bps"].sum(), -df.filter(pl.col("net_bps") < 0)["net_bps"].sum()
        return {"trades": df.height, "win rate %": round(100 * (df["net_bps"] > 0).mean(), 1) if df.height else None,
                "profit factor": round(_w / _l, 2) if _l else None, "avg bps": round(df["net_bps"].mean(), 1) if df.height else None,
                "continuation losers": int((df["net_bps"] <= TAIL_BPS).sum())}

    _rows = []
    for _p in ["tune", "Q3 2026", "Oct-Nov 2026"]:
        _d = LAB.filter(pl.col("period") == _p)
        _rows.append({"period": _p, "rule": "every short", **_stats(_d)})
        _rows.append({"period": _p, "rule": f"skip riskiest {_pct:.0f}%", **_stats(_d.filter(pl.col("kept")))})
    LAB_STATS = pl.DataFrame(_rows)
    _eq = pl.concat([
        LAB.sort("entry_ts").with_columns(rule=pl.lit("every short"), cum=pl.col("net_bps").cum_sum().over("period")),
        LAB.filter(pl.col("kept")).sort("entry_ts").with_columns(rule=pl.lit("after veto"), cum=pl.col("net_bps").cum_sum().over("period")),
    ]).select("entry_ts", "period", "rule", "cum", "symbol", "net_bps")
    def _curve(p):
        return alt.Chart(_eq.filter(pl.col("period") == p)).mark_line(strokeWidth=2.2, interpolate="step-after").encode(
            x=alt.X("entry_ts:T", title=None), y=alt.Y("cum:Q", title="cumulative bps"),
            color=alt.Color("rule:N", scale=alt.Scale(domain=["every short", "after veto"], range=[GREY, FADE]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["symbol", "entry_ts", "net_bps", "cum"],
        ).properties(height=240, width="container", title=p)

    mo.vstack([
        mo.ui.table(LAB_STATS, selection=None, page_size=6, label="Live scorecard (target: win rate 60-70%, profit factor 1.5+)"),
        mo.hstack([_curve(p) for p in ["tune", "Q3 2026", "Oct-Nov 2026"]], widths="equal", wrap=True, gap=1),
    ])
    return (LAB,)


@app.cell(hide_code=True)
def _(LAB, OUT_SCALE, score_pick):
    scatter = mo.ui.altair_chart(
        alt.Chart(LAB.with_columns(status=pl.when(pl.col("kept")).then(pl.lit("kept")).otherwise(pl.lit("skipped")))).mark_point(filled=True, size=60).encode(
            x=alt.X("risk:Q", title=f"{score_pick.value} (higher = riskier)"),
            y=alt.Y("net_bps:Q", title="short P&L (bps)"),
            color=alt.Color("outcome:N", scale=OUT_SCALE, legend=alt.Legend(orient="top", title=None)),
            shape=alt.Shape("status:N", scale=alt.Scale(domain=["kept", "skipped"], range=["circle", "cross"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["symbol", "trade_date", "period", alt.Tooltip("net_bps:Q", format=".0f"), alt.Tooltip("risk:Q", format=".2f")],
        ).properties(height=380, width="container"),
        chart_selection="interval",
    )
    mo.vstack([mo.md("### Drag a box on the chart to pull those trades into the table"), scatter])
    return (scatter,)


@app.cell(hide_code=True)
def _(LAB, scatter):
    _sel = scatter.value if scatter.value is not None and len(scatter.value) else LAB.head(0)
    _sel = pl.DataFrame(_sel) if not isinstance(_sel, pl.DataFrame) else _sel
    _n = _sel.height
    _msg = (f"**{_n} trades selected** · win rate {100 * (_sel['net_bps'] > 0).mean():.0f}% · avg {_sel['net_bps'].mean():+.0f} bps · "
            f"{int((_sel['net_bps'] <= TAIL_BPS).sum())} continuation losers") if _n else "_No trades selected yet._"
    mo.vstack([mo.md(_msg), mo.ui.table(_sel.select([c for c in ["symbol", "trade_date", "period", "net_bps", "risk", "minutes_to_trigger", "entry_vs_hod_bps", "red_vol_ratio", "kept"] if c in _sel.columns]), page_size=10)]) if _n else mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(FC):
    PC_DATA = FC.select(
        "symbol", "period", "outcome", "net_bps", "minutes_to_trigger", "entry_vs_hod_bps", "red_vol_ratio",
        "kronos_p_up_tail", "chronos_q90_max_bps",
    ).with_columns(pl.col(pl.Float64).round(2))
    pc = mo.ui.anywidget(ParallelCoordinates(
        PC_DATA.drop("symbol"), color_by="outcome", height=520,
        color_map={"faded (win)": FADE, "small loss": GREY, "continuation loser (≤ −200)": RUN},
    ))
    mo.vstack([mo.md("""## Build a filter by hand

Each line is one short. Drag along any axis to keep only part of its range (for example slow run-ups, or entries
near the high); drag axis titles to reorder. The scorecard under the chart recomputes for whatever survives,
so you can see straight away whether a filter helps on the holdout periods as well as on the tune year."""), pc])
    return PC_DATA, pc


@app.cell(hide_code=True)
def _(PC_DATA, pc):
    _uids = (pc.value or {}).get("filtered_uids") or []
    _idx = sorted({int(u) for u in _uids if str(u).isdigit()})
    _kept = PC_DATA.with_row_index("uid").filter(pl.col("uid").is_in(_idx)) if _idx else PC_DATA.with_row_index("uid")

    def _row(df, p):
        _d = df.filter(pl.col("period") == p)
        _w, _l = _d.filter(pl.col("net_bps") > 0)["net_bps"].sum(), -_d.filter(pl.col("net_bps") < 0)["net_bps"].sum()
        return {"period": p, "trades": _d.height, "win rate %": round(100 * (_d["net_bps"] > 0).mean(), 1) if _d.height else None,
                "profit factor": round(_w / _l, 2) if _l else None, "continuation losers": int((_d["net_bps"] <= TAIL_BPS).sum())}

    _tbl = pl.DataFrame([_row(_kept, p) for p in ["tune", "Q3 2026", "Oct-Nov 2026"]])
    mo.vstack([mo.md(f"**{_kept.height} of {PC_DATA.height} shorts pass the brushed filter**"), mo.ui.table(_tbl, selection=None)])
    return


@app.cell(hide_code=True)
def _():
    _tail = pl.read_csv(data_file(REP / "tail_anatomy.csv")).with_columns(
        edge=(pl.col("AUC_tail_vs_rest") - 0.5),
        direction=pl.when(pl.col("AUC_tail_vs_rest") < 0.5).then(pl.lit("lower in tail losers")).otherwise(pl.lit("higher in tail losers")),
    )
    _c = (alt.Chart(_tail).mark_bar().encode(
        y=alt.Y("feature:N", sort=alt.EncodingSortField("edge", op="min", order="ascending"), title=None),
        x=alt.X("edge:Q", title="AUC − 0.5 (0 = no signal)", scale=alt.Scale(domain=[-0.2, 0.2])),
        color=alt.Color("direction:N", scale=alt.Scale(range=[RUN, FADE]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature", "winners_median", "tail_median", alt.Tooltip("AUC_tail_vs_rest:Q", format=".2f")],
    ).properties(width="container", height=260, title="Single features that separate continuation losers (tune period)"))
    card("LightGBM: the workhorse", """
**What it is.** Gradient-boosted decision trees. Fast to train, fast to score, strong on tables of numbers.

**Good at.** Mixing many weak signals (run speed, distance from the high, red-bar volume, SPY/QQQ) into one
risk score, and finding interactions a human rule misses.

**Where it fits.** At the arm: score each short before entry and veto (or size down) the riskiest ones.
Microsecond scoring means it can sit inside a Rust runner.

**What we found.** Out-of-sample AUC **0.63–0.71** for tail losers, the steadiest of the three tabular models.
Vetoing the riskiest 20% still left profit factor at **0.72–0.91**. It ranks; it doesn't rescue.

**Chart.** The raw ingredients it combines. Fast runs (low `minutes_to_trigger`) and entries far below the
high are the strongest tells.
    """, _c)
    return


@app.cell(hide_code=True)
def _(CMP, PERIODS):
    _d = CMP.filter(pl.col("strategy").is_in(["baseline (no model)", "CatBoost"])).with_columns(
        period=pl.col("period").replace({"tune (OOF)": "1 tune", "validate (Q3 26)": "2 Q3 2026", "forward (Oct-Nov 26)": "3 Oct-Nov"}))
    _bars = alt.Chart(_d).mark_bar().encode(
        x=alt.X("period:N", title=None), xOffset="strategy:N",
        y=alt.Y("profit_factor:Q", title="Profit factor", scale=alt.Scale(domain=[0, 1.6])),
        color=alt.Color("strategy:N", scale=alt.Scale(domain=["baseline (no model)", "CatBoost"], range=[GREY, ACC]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["strategy", "period", "profit_factor", "trades", "tail_losers_left"])
    _target = alt.Chart(pl.DataFrame({"y": [1.5]})).mark_rule(color=FADE, strokeDash=[5, 4]).encode(y="y:Q")
    _c = (_bars + _target).properties(width="container", height=260, title="CatBoost veto vs no model (target 1.5 dashed)")
    card("CatBoost: categories and stability checks", """
**What it is.** Another boosted-tree model, built to handle categorical inputs (sector, catalyst type,
weekday) without one-hot encoding, with ordered boosting that resists target leakage.

**Good at.** Mixed tables where labels matter: *catalyst = strong positive earnings* next to price features.
That makes it the natural home for Chris's news taxonomy once it's frozen.

**Where it fits.** Same gate as LightGBM, but trained with the red/yellow/green catalyst label as a feature.

**What we found.** A warning about overfitting: profit factor **1.01** on the tune folds fell to **0.58** on Q3
2026, then **0.83** in Oct-Nov. Without the catalyst column it has little to add over LightGBM.
    """, _c)
    return


@app.cell(hide_code=True)
def _(CMP, PERIODS):
    _d = CMP.filter(pl.col("strategy").is_in(["LightGBM", "CatBoost", "TabPFN v2"])).with_columns(
        period=pl.col("period").replace({"tune (OOF)": "1 tune", "validate (Q3 26)": "2 Q3 2026", "forward (Oct-Nov 26)": "3 Oct-Nov"}))
    _line = alt.Chart(_d).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("period:N", title=None), y=alt.Y("auc:Q", title="AUC for tail losers", scale=alt.Scale(domain=[0.45, 0.8])),
        color=alt.Color("strategy:N", scale=alt.Scale(domain=["TabPFN v2", "LightGBM", "CatBoost"], range=[FADE, GREY, ACC]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["strategy", "period", alt.Tooltip("auc:Q", format=".3f"), "profit_factor"])
    _coin = alt.Chart(pl.DataFrame({"y": [0.5]})).mark_rule(color="#888", strokeDash=[3, 3]).encode(y="y:Q")
    _c = (_line + _coin).properties(width="container", height=260, title="Ranking skill by period (0.5 = coin flip)")
    card("TabPFN v2: a model that doesn't train", """
**What it is.** A transformer pre-trained on millions of synthetic tables. You hand it your labelled trades
as context and it predicts the new ones in one forward pass, no tuning loop.

**Good at.** Small samples. We have ~1,200 training trades and ~180 tail losers, which is exactly the
regime where boosted trees overfit and TabPFN tends to shine.

**Where it fits.** A quick second opinion for any new edge Mark proposes: a baseline in minutes before anyone
tunes anything. Also useful for rare categories like *strong positive earnings shock* (5 cases so far).

**What we found.** Best tune AUC (**0.67**) but it slid to **0.65** forward and profit factor **0.64**.
Runs in a separate worker process because torch and LightGBM clash on macOS.
    """, _c)
    return


@app.cell(hide_code=True)
def _(FC, OUT_SCALE):
    _c = (alt.Chart(FC).mark_circle(size=34, opacity=0.75).encode(
        x=alt.X("chronos_q90_max_bps:Q", title="Chronos q90 forecast of the highest price ahead (bps)", scale=alt.Scale(type="symlog")),
        y=alt.Y("net_bps:Q", title="Actual short P&L (bps)"),
        color=alt.Color("outcome:N", scale=OUT_SCALE, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["symbol", "trade_date", alt.Tooltip("net_bps:Q", format=".0f"), alt.Tooltip("chronos_q90_max_bps:Q", format=".0f")],
    ).properties(width="container", height=280, title=f"{FC.height} real trades: forecast run-up vs what happened"))
    card("Chronos-Bolt: a price-path forecaster", """
**What it is.** Amazon's time-series foundation model (T5-style), trained on many kinds of series. Give it the
1-minute closes so far and it returns forecast quantiles for the next 20 minutes.

**Good at.** Reading the *shape* of a run without any feature engineering, and giving a range, not a point.

**Where it fits.** As one more input to the gate: "how high could this still go?" (the q90 path max). If the
upper band is far above the arm price, the spike may not be done.

**What we found.** Weak. AUC **0.68** on tune, **0.54** forward. The pre-entry window is only 2-45 bars
(our data starts at 09:30), which is thin context. Worth retrying with 30-second bars and pre-market.

**Chart.** Red dots should sit to the right if the forecast worked. They're mixed in.
    """, _c)
    return


@app.cell(hide_code=True)
def _(FC, OUT_SCALE):
    _k = FC.group_by("outcome").agg(
        mean_p=pl.col("kronos_p_up_tail").mean(),
        share_flagged=(pl.col("kronos_p_up_tail") >= 0.25).mean(),
        n=pl.len(),
    )
    _c = (alt.Chart(_k).mark_bar().encode(
        x=alt.X("outcome:N", title=None, sort=["faded (win)", "small loss", "continuation loser (≤ −200)"], axis=alt.Axis(labelAngle=0, labelLimit=140)),
        y=alt.Y("share_flagged:Q", title="share of trades Kronos flags (≥ 4 of 16 paths run +200 bps)", axis=alt.Axis(format="%")),
        color=alt.Color("outcome:N", scale=OUT_SCALE, legend=None),
        tooltip=["outcome", "n", alt.Tooltip("mean_p:Q", format=".2f"), alt.Tooltip("share_flagged:Q", format=".0%")],
    ).properties(width="container", height=280, title="Does Kronos see the continuation coming?"))
    card("Kronos: a candlestick language model", """
**What it is.** A model that tokenises OHLCV candles the way an LLM tokenises words, trained on billions of
candles from many exchanges, then samples plausible next candles.

**Good at.** Sampling *many* futures. The share of paths that keep running is a direct, explainable risk
number: "12 of 16 simulated futures went another 2% higher".

**Where it fits.** Pre-entry risk score, or a size multiplier, and a nice visual for the replay notebook
(draw the 16 paths on the chart).

**What we found.** No reliable edge: AUC **0.51-0.56** on the holdouts for P(path reaches +200 bps).
CPU speed ~0.3 s per trade with 16 paths. Same thin-context problem as Chronos.

**Chart.** On all 575 lab trades Kronos flags about 15% of continuation losers and 5% of the others: a real but small tell that misses most of the damage.
    """, _c)
    return


@app.cell(hide_code=True)
def _():
    _rng = np.random.default_rng(7)
    _groups = {
        "strong positive earnings shock": (2.2, 1.6), "positive guidance shock": (1.2, 2.3),
        "moderate earnings beat": (0.2, 1.2), "analyst action": (-1.4, 0.6),
        "routine disclosure": (-2.0, -1.4), "no identifiable catalyst": (-0.6, -2.2),
    }
    _rows = []
    for _k, (_x, _y) in _groups.items():
        for _ in range(9):
            _rows.append({"category": _k, "x": _x + _rng.normal(0, .28), "y": _y + _rng.normal(0, .28), "kind": "past headline"})
    _pts = pl.DataFrame(_rows)
    _q = pl.DataFrame({"category": ["new headline"], "x": [1.9], "y": [1.35], "kind": ["new headline"]})
    _cent = _pts.group_by("category").agg(pl.col("x").mean(), pl.col("y").mean())
    _c = (alt.Chart(_pts).mark_circle(size=40, opacity=0.6).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            color=alt.Color("category:N", legend=alt.Legend(orient="bottom", columns=2, title=None)))
        + alt.Chart(_cent).mark_text(fontSize=11, dy=-14, color="#ddd").encode(x="x:Q", y="y:Q", text="category:N")
        + alt.Chart(_q).mark_point(shape="diamond", size=260, filled=True, color="#fff").encode(x="x:Q", y="y:Q")
    ).properties(width="container", height=300, title="ILLUSTRATION: headlines placed in embedding space")
    card("CLM-8B: pick the nearest answer", """
**What it is.** A contrastive language model (Qwen3-8B base) that embeds a question and a list of options in the
same space and picks the closest. No text generation, so it's fast and its choice is easy to audit.

**Good at.** Sorting text into a fixed menu: *which of Chris's 11 catalyst categories is this headline?*,
*which past event looks most like this 8-K?*

**Where it fits.** Pre-open, on every headline and filing for the 400-name universe. Its pick becomes the
catalyst label that LightGBM/CatBoost use, and the "nearest past events" become Mark's *rank by expected move*:
average the 20-minute fade of the 20 nearest past headlines.

**Status.** Not run yet (~8-9 GB at Q8; we'll ask before downloading).

**Chart (made-up points, for illustration).** The white diamond is a new headline; it lands nearest the
*strong positive earnings shock* cluster, so the short would be flagged red.
    """, _c)
    return


@app.cell(hide_code=True)
def _():
    _d = pl.DataFrame({"decision": ["took", "watch", "rejected", "abstained"], "signals": [3, 6, 324, 100]})
    _f = pl.DataFrame({"cause": ["false breakout", "exit gave back profit", "unclear"], "losers": [271, 10, 54]})
    _a = alt.Chart(_d).mark_bar().encode(
        x=alt.X("signals:Q", title="breakout signals (Feb 2026, 30 tickers)", scale=alt.Scale(type="symlog")),
        y=alt.Y("decision:N", sort=["took", "watch", "rejected", "abstained"], title=None),
        color=alt.Color("decision:N", scale=alt.Scale(domain=["took", "watch", "rejected", "abstained"], range=[FADE, ACC, RUN, GREY]), legend=None),
        tooltip=["decision", "signals"]).properties(width="container", height=120, title="Jev take / watch / reject / abstain (First-Hour Desk)")
    _b = alt.Chart(_f).mark_bar().encode(
        x=alt.X("losers:Q", title="losing trades"), y=alt.Y("cause:N", sort="-x", title=None),
        color=alt.value(GREY), tooltip=["cause", "losers"]).properties(width="container", height=100, title="Jev's cause for each loss")
    card("Jev: fast, typed decisions", """
**What it is.** A decision model (TypeSafe's System One) that never writes prose. You give it a state and typed
questions (yes/no, pick one, score 0-1) and it answers in under a second, cheaply, with a confidence.

**Good at.** Judgement calls that rules can't express but that must be fast and auditable: *is this
headline about the ticker?*, *grade this catalyst*, *should this order pause?*

**Where it fits.** Everywhere a human would glance: the pre-open red/yellow/green screen, a confidence gate
at the arm, a news kill switch during the 20 minutes, and tagging the cause of each loss afterwards.
See the **Jev notebook** (`notebooks/jev_lab.py`) for 466 recorded scenarios.

**What we have.** The First-Hour Desk demo (stand-in breakout strategy, not the short): Jev took 3 trades,
watched 6, rejected 324 and abstained on 100, then tagged 271 of 335 losers as false breakouts.
    """, mo.vstack([_a, _b]))
    return


@app.cell(hide_code=True)
def _(CMP):
    _d = CMP.filter(pl.col("period") != "tune (OOF)").with_columns(
        family=pl.when(pl.col("strategy").str.contains("vote|mean prob|stack|sequential")).then(pl.lit("combination"))
        .when(pl.col("strategy").str.starts_with("baseline")).then(pl.lit("no model"))
        .otherwise(pl.lit("single model")),
        name=pl.col("strategy").str.slice(0, 22),
    )
    _pts = alt.Chart(_d).mark_point(size=110, filled=True).encode(
        x=alt.X("auc:Q", title="AUC for tail losers (holdout)", scale=alt.Scale(domain=[0.45, 0.75])),
        y=alt.Y("profit_factor:Q", title="Profit factor after veto", scale=alt.Scale(domain=[0.4, 1.6])),
        color=alt.Color("family:N", scale=alt.Scale(range=[ACC, GREY, FADE]), legend=alt.Legend(orient="bottom", title=None)),
        shape="period:N", tooltip=["strategy", "period", "auc", "profit_factor", "trades"])
    _goal = alt.Chart(pl.DataFrame({"y": [1.5]})).mark_rule(color=FADE, strokeDash=[5, 4]).encode(y="y:Q")
    mo.vstack([
        mo.md("""
## Putting it together: ranking skill is not profit

Every model and combination on the two holdout periods. Moving right (better ranking) barely moves anything up.
Tail losers and winners look too alike *at the moment of entry*, so the next gains probably come from new
**inputs** (the protected-limit entry, 30-second bars, catalyst labels from CLM-8B / Jev) rather than new models.
        """),
        (_pts + _goal).properties(width="container", height=320),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.vstack([mo.md("## Where each model sits on the trading morning"), mo.mermaid("""
%%{init: {'theme': 'dark', 'themeVariables': {'fontSize': '22px'}, 'flowchart': {'nodeSpacing': 40, 'rankSpacing': 55, 'useMaxWidth': true}}}%%
flowchart TB
  subgraph PRE[Before 09:30]
    N[News and filings] --> CLM[CLM-8B: nearest catalyst category]
    N --> J1[Jev: red / yellow / green]
  end
  subgraph OPEN[09:30-10:00]
    U[Universe + gap filter] --> R[Run to +5%]
    R --> A[Arm on first red bar]
  end
  subgraph GATE[At the arm]
    A --> T[LightGBM / CatBoost / TabPFN: P tail loser]
    A --> F[Kronos / Chronos: paths still running?]
    T --> G{Jev confidence gate}
    F --> G
    CLM --> G
    J1 --> G
  end
  G -->|take| S[Short with protected limit]
  G -->|size down| S
  G -->|skip| X[No trade]
  S --> C[Cover after 20 min]
  C --> REV[Review: Jev tags the cause of each loss]
""")])
    return


if __name__ == "__main__":
    app.run()
