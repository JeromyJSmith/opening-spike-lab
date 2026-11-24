# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "polars>=2.1.0",
#     "altair>=6.3.0",
# ]
# ///
"""Slide deck: spike-short results, what they mean, and what to try next.

Run:  uv run marimo run notebooks/results_deck.py   (opens as slides)
Reads the saved tables in data/reports/ and research/loop/; it re-runs nothing.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Spike short: results and next steps",
                 layout_file="layouts/results_deck.slides.json")

with app.setup:
    import json
    from pathlib import Path

    import altair as alt
    alt.theme.enable("dark")
    import marimo as mo
    import polars as pl

    ROOT = Path(__file__).resolve().parents[1]
    REP = ROOT / "data" / "reports"
    LOOP = ROOT / "research" / "loop"
    GREEN, RED, BLUE, GREY, AMBER = "#2f9e6e", "#d64545", "#3b6fd6", "#8a8f98", "#d69a2d"
    PERIODS = ["tune (Jul25-Jun26)", "validate (Q3 26)", "forward (Oct-Nov 26)"]
    SHORT = {"tune (Jul25-Jun26)": "Jul 25 - Jun 26 (tune)", "validate (Q3 26)": "Q3 26 (holdout)",
             "forward (Oct-Nov 26)": "Oct-Nov 26 (holdout)", "tune (OOF)": "Jul 25 - Jun 26 (tune)"}
    PSCALE = alt.Scale(domain=list(dict.fromkeys(SHORT.values())), range=[BLUE, AMBER, GREY])

    def side(chart, text: str):
        """Chart on the left, one block of text on the right (wrapped so it stays one column)."""
        return mo.hstack([chart, mo.Html(f"<div style='max-width:460px;font-size:0.92em'>{mo.md(text).text}</div>")],
                         justify="start", align="start", gap=2, widths=[1.1, 1])


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'results_deck'
    _nxt = _ORDER[(_ORDER.index(_me) + 1) % len(_ORDER)]
    _mode = "wasm" if _sys.platform == "emscripten" else _os.environ.get("SPIKE_HUB_MODE", "live")
    if _mode == "wasm":
        _hub, _next = "../../index.html", (f"../{_nxt}/" if _nxt in _WASM else f"../../notebooks/{_nxt}.html")
    elif _mode == "static":
        _hub, _next = "https://jeromyjsmith.github.io/opening-spike-lab/", f"https://jeromyjsmith.github.io/opening-spike-lab/notebooks/{_nxt}.html"
    else:
        _hub, _next = "http://127.0.0.1:2750/", f"http://localhost:{_PORTS[_nxt]}/"
    _molab = "https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/results_deck.py"
    _btn = ("display:inline-block;padding:.45rem .9rem;border:1px solid #2a3631;border-radius:999px;"
            "color:#e5ece8;text-decoration:none;font:500 14px system-ui;background:#161f1c")
    NAV = _mo.Html(f'<div style="display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;max-width:720px">'
             f'<a href="{_hub}" target="_top" style="{_btn}">← Back to hub</a>'
             f'<span style="color:#9aaba4;font:13px system-ui">Notebook {_ORDER.index(_me) + 1} of {len(_ORDER)}</span>'
             f'<a href="{_molab}" target="_blank" rel="noopener" style="{_btn};border-color:#e3a948;color:#e3a948">Open in molab ↗</a>'
             f'<a href="{_next}" target="_top" style="{_btn};border-color:#5cc2a2;color:#5cc2a2">Next notebook →</a></div>')
    _mo.vstack([NAV, _mo.Html(
        '<div style="margin:.4rem 0 .2rem;padding:.8rem 1rem;border:2px solid #e3a948;border-radius:12px;background:#1f1a10;'
        'color:#f3e2bf;font:600 16px system-ui;max-width:720px">'
        '<span>This is a slide deck. Use the arrows at the bottom right, or the ← → keys, to step through the slides.</span>'
        '<span style="font-size:24px"> ↘</span></div>'), _mo.md(r"""
    # Opening spike short: what the data says

    **Real Alpaca 1-minute data, May 2025 to Nov 2026. Paper research only.**

    1. The rule and the question
    2. Results: the rule as written loses money
    3. Why: a few continuation losers do the damage
    4. What we tried: filters, exits, machine-learning models, forecasters
    5. A first keep-or-undo research loop
    6. What to change next

    <small>24 Nov 2026 · sources: docs/trading/reports/ and research/loop/</small>
    """)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The rule, in plain words

    - **Universe:** liquid US stocks, price ≥ \$50, daily range (ATR) ≥ 3%, \$100M+ traded a day, in two tiers (\$100M–1B and \$1B+).
    - **Setup:** the stock opens 0.25–1 ATR above yesterday's close, then runs **+5%** above its open before 10:00 ET (8:00 MT).
    - **Entry:** short at the bar after the **first red 1-minute bar**.
    - **Exit:** cover **20 minutes** later. 5 bps round-trip slippage.

    **Chris's problem:** most trades make about +30 bps, but one or two **continuation losers** a day (the stock keeps squeezing up 2–5%) wipe out the day. Stops tried so far also cut the winners.

    **Targets:** win rate 60–70%, profit factor (gross wins ÷ gross losses) ≥ 1.5.

    **How we test:** tune on Jul 2025 – Jun 2026. Q3 2026 and Oct–Nov 2026 are **held out**: scored once, never used to choose anything.
    """)
    return


@app.cell(hide_code=True)
def _():
    _sc = pl.read_csv(REP / "baseline_scorecard.csv").with_columns(pl.col("period").replace(SHORT).alias("when"))
    _bars = alt.Chart(_sc).mark_bar().encode(
        x=alt.X("when:N", title=None, sort=list(SHORT.values())), y=alt.Y("profit_factor:Q", title="Profit factor", scale=alt.Scale(domain=[0, 1.7])),
        color=alt.Color("when:N", scale=PSCALE, legend=None), tooltip=["when", "trades", "win_rate", "profit_factor", "avg_net_bps"])
    _target = alt.Chart(pl.DataFrame({"y": [1.5]})).mark_rule(color=GREEN, strokeDash=[6, 4], size=2).encode(y="y:Q")
    _even = alt.Chart(pl.DataFrame({"y": [1.0]})).mark_rule(color=GREY).encode(y="y:Q")
    mo.vstack([
        mo.md("## Result 1: the rule as coded loses money in every period"),
        side(
            (_bars + _even + _target).properties(width=340, height=260),
            f"""
| | trades | win rate | profit factor | avg per trade |
|---|---|---|---|---|
""" + "\n".join(f"| {r['when']} | {r['trades']} | {r['win_rate']}% | **{r['profit_factor']}** | {r['avg_net_bps']} bps |" for r in _sc.iter_rows(named=True)) + """

Green dashed line = target (1.5). Grey = break-even (1.0).

**In plain words:** for every \\$1.00 the winners make, the losers take back \\$1.10 to \\$1.33. Win rate is 43–51%, not 60–70%.

PyBroker, an independent backtester, reproduces the same 136 Q3 trades at the same minutes, so this is not a coding error.
""",
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    _sc = pl.read_csv(REP / "baseline_scorecard.csv").with_columns(pl.col("period").replace(SHORT).alias("when"))
    _d = _sc.select("when", pl.col("tail_share_of_losses_pct").alias("tail losers' share of all losing bps"),
                    (pl.col("tail_losers") / pl.col("trades") * 100).round(1).alias("tail losers' share of trades")).unpivot(index="when")
    _c = alt.Chart(_d).mark_bar().encode(
        y=alt.Y("variable:N", title=None), x=alt.X("value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("variable:N", scale=alt.Scale(range=[RED, GREY]), legend=None), row=alt.Row("when:N", title=None, sort=list(SHORT.values())),
        tooltip=["when", "variable", "value"]).properties(width=300, height=45)
    mo.vstack([
        mo.md("## Result 2: Chris is right about where the damage comes from"),
        side(_c, """
A **tail loser** here is a trade that loses 2% (200 bps) or more.

- They are **about 15% of trades**.
- They cause **60–68% of all the losing**.
- Worst single trade: **−10.2%** (CAR, 21 Apr 2026).

Remove the tail losers and the rule makes money in every period. The whole problem is identifying them **before** entering, without also throwing away the winners.
"""),
    ])
    return


@app.cell(hide_code=True)
def _():
    _a = pl.read_csv(REP / "tail_anatomy.csv").with_columns((pl.col("AUC_tail_vs_rest") - 0.5).abs().alias("strength")).sort("strength", descending=True)
    _c = alt.Chart(_a).mark_bar().encode(
        x=alt.X("AUC_tail_vs_rest:Q", title="AUC (0.5 = no signal; left = lower in tail losers)", scale=alt.Scale(domain=[0.25, 0.75])),
        x2=alt.datum(0.5),
        y=alt.Y("feature:N", sort=_a["feature"].to_list(), title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.condition(alt.datum.strength >= 0.1, alt.value(RED), alt.value(GREY)),
        tooltip=["feature", "winners_median", "tail_median", "AUC_tail_vs_rest"]).properties(width=330, height=260)
    _mid = alt.Chart(pl.DataFrame({"x": [0.5]})).mark_rule(color="#e5ece8").encode(x="x:Q")
    mo.vstack([
        mo.md("## What tells a tail loser apart (tune period)"),
        side(_c + _mid, """
Bars far from 0.5 carry a signal (red = usable).

- **Speed of the run-up is the clearest clue.** Tail losers hit +5% a median **5 minutes** after the open, winners **12 minutes**. A stock that rips straight up tends to keep going.
- **Entry far below the high so far** (the stock had already pulled back before our red bar) is a late, worse entry.
- **Red bar closing below the VWAP since the trigger** looks like a reversal but often bounces.

Each clue is real, but each one is weak on its own: plenty of winners share it too.
"""),
    ])
    return


@app.cell(hide_code=True)
def _():
    _f = pl.read_csv(REP / "filter_ablations.csv").filter(pl.col("period") != "tune (Jul25-Jun26)").with_columns(pl.col("period").replace(SHORT).alias("when"))
    _c = alt.Chart(_f).mark_circle(size=110).encode(
        x=alt.X("profit_factor:Q", title="Profit factor on held-out periods", scale=alt.Scale(domain=[0.6, 1.6])),
        y=alt.Y("variant:N", title=None, sort="-x"), color=alt.Color("when:N", scale=PSCALE, legend=alt.Legend(orient="bottom")),
        tooltip=["variant", "when", "trades", "win_rate", "profit_factor", "tail_losers"]).properties(width=320, height=280)
    _t = alt.Chart(pl.DataFrame({"x": [1.5]})).mark_rule(color=GREEN, strokeDash=[6, 4], size=2).encode(x="x:Q")
    mo.vstack([
        mo.md("## What we tried, part 1: simple filters and stops"),
        side(_c + _t, """
Each filter added alone to the rule, scored on the held-out quarters.

- **Early bailouts** (cover if down 1% in the first 5 min) cut tail losers from 25 to 8, but win rate drops to 37%: **the stop cuts winners too**, exactly what Chris found.
- **Stop above the high of day**: same trade-off.
- **Heavy red bar** (red bar volume ≥ 1.5× the run-up average) helps a little but leaves only a third of the trades.
- A **decision-tree gate** looked great on the data it was fit on (PF 1.77) and failed out of sample (0.77–0.88): **overfit**.

Nothing reaches 1.5 on either held-out period.
"""),
    ])
    return


@app.cell(hide_code=True)
def _():
    _m = pl.read_csv(REP / "model_lab_comparison_layer1_2.csv").filter(
        ~pl.col("strategy").str.starts_with("vote") & ~pl.col("strategy").str.starts_with("mean")
        & ~pl.col("strategy").str.starts_with("sequential")).with_columns(
        pl.col("strategy").str.replace(r"^stack \(logistic\).*", "stack of all six").str.replace(r"^Chronos-Bolt.*", "Chronos-Bolt")
        .str.replace(r"^Kronos: q90.*", "Kronos (q90 path)").str.replace(r"^Kronos: P.*", "Kronos (P +2%)"),
        pl.col("period").replace(SHORT).alias("when"))
    _auc = alt.Chart(_m.filter(pl.col("strategy") != "baseline (no model)")).mark_circle(size=100).encode(
        x=alt.X("auc:Q", title="How well it ranks losers (AUC, 0.5 = chance)", scale=alt.Scale(domain=[0.45, 0.75])),
        y=alt.Y("strategy:N", title=None), color=alt.Color("when:N", scale=PSCALE, legend=alt.Legend(orient="bottom")),
        tooltip=["strategy", "when", "auc", "profit_factor", "win_rate", "losers_avoided", "winners_lost"]).properties(width=330, height=240)
    _pf = alt.Chart(_m).mark_circle(size=100).encode(
        x=alt.X("profit_factor:Q", title="Profit factor (skip riskiest 20%)", scale=alt.Scale(domain=[0.5, 1.6])),
        y=alt.Y("strategy:N", title=None), color=alt.Color("when:N", scale=PSCALE, legend=None),
        tooltip=["strategy", "when", "profit_factor", "win_rate", "trades"]).properties(width=330, height=240)
    _t = alt.Chart(pl.DataFrame({"x": [1.5]})).mark_rule(color=GREEN, strokeDash=[6, 4], size=2).encode(x="x:Q")
    mo.vstack([
        mo.md("## What we tried, part 2: machine-learning models and price forecasters"),
        mo.hstack([_auc, _pf + _t], justify="start", gap=2),
        mo.md("""
**LightGBM, CatBoost, TabPFN** read about 50 facts known at entry time. **Chronos-Bolt and Kronos** forecast the next 20 minutes of price.
The tree models **do rank losers better than chance on unseen months** (AUC 0.6–0.7, the old tree did not). The forecasters barely do (0.51–0.65).
But skipping the riskiest 20% of trades still leaves profit factor at **0.57–0.99**: the signal is too weak to separate losers from winners cleanly.
"""),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why the models can't fix it (yet)

    **The losers and the winners look alike at the moment of entry.**

    - When a model skips the riskiest 20% of trades it catches 30–40% of the tail losers, and also about 15% of the winners.
    - There are only about **55 tail losers** in the 12-month tune set at 5% / \$50 (about 180 if we pool the 4% and \$25 variants). That is very little to learn from. More model complexity does not help: the simple models did as well as the fancy ones.
    - The forecasters see only 2–45 one-minute bars (the data starts at 09:30, no pre-market), and they were trained on much longer histories.
    - Kronos runs at about 0.3 seconds per trade on this Mac's processor, not the 10–30 ms quoted for a GPU.

    **What this means:** the edge, if there is one, is more likely in **how we enter** and **which news days we trade** than in a smarter veto on the same entries.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack([
        mo.md("## The keep-or-undo loop (autoresearch pattern), now set up"),
        mo.mermaid("""
flowchart LR
    A[Idea] --> B[Edit filters.py]
    B --> C[Score on tune]
    C --> D{Better by 0.03?}
    D -- yes --> E[Keep]
    D -- no --> F[Undo]
    E --> A
    F --> A
    E -.once, at the end.-> H[Sealed holdout]
"""),
        mo.md("""
- The agent (Claude Code) may only change `filters.py`: a skip rule on facts known at entry, plus rule settings such as an early bailout.
- The scorer, the data and the holdout are fixed and off-limits. Holdout is opened **once** per final candidate; a second look is refused.
- Instructions for an unattended run: `research/loop/program.md`. Run with `uv run python research/loop/loop.py try <candidate> "<hypothesis>"`.
"""),
    ])
    return


@app.cell(hide_code=True)
def _():
    _r = pl.read_csv(LOOP / "results.tsv", separator="\t").with_row_index("step")
    _c = alt.Chart(_r).mark_circle(size=120).encode(
        x=alt.X("step:O", title="attempt"), y=alt.Y("score:Q", title="Tune score (median quarterly PF)"),
        color=alt.Color("decision:N", scale=alt.Scale(domain=["baseline", "keep", "undo"], range=[GREY, GREEN, RED])),
        tooltip=["file", "decision", "score", "trades", "win_rate", "profit_factor", "tail_losers", "hypothesis"]).properties(width=320, height=230)
    _line = alt.Chart(_r.filter(pl.col("decision") != "undo")).mark_line(color=GREEN).encode(x="step:O", y="score:Q")
    _rev = json.loads((LOOP / "reveals.tsv").read_text().splitlines()[-1].split("\t")[3])
    _rows = [{"period": k.rsplit(" ", 1)[0], "rule": k.rsplit(" ", 1)[1], **v} for k, v in _rev.items()]
    _h = pl.DataFrame(_rows)
    _tune = _r.filter(pl.col("decision") == "keep").tail(1).row(0, named=True)
    mo.vstack([
        mo.md("## Loop round 1: 10 ideas, 4 kept, then the holdout"),
        side(_c + _line, f"""
**Kept:** skip if the +5% came in under 10 minutes; skip if entry is more than 1.5% below the high; skip if the red bar's volume is under 0.6× the run-up average; bail out if down 1.25% in the first 5 minutes.

**On the tune period:** profit factor 0.91 → **{_tune['profit_factor']}**, tail losers 55 → **{_tune['tail_losers']}**, win rate {_tune['win_rate']}%.
"""),
        mo.md("**Holdout, opened once:**"),
        mo.ui.table(_h.select("period", "rule", "trades", "win_rate", "profit_factor", "avg_net_bps", "tail_losers"), selection=None),
        mo.md("Tail losers drop by about three quarters on both held-out periods (25 → 6, 11 → 3), but profit factor only reaches **0.95–0.97** "
              "and win rate stays ~45%. The 1.49 on tune was partly fit to the tune period. The loop is honest about that because the holdout was sealed."),
    ])
    return


@app.cell(hide_code=True)
def _():
    _t = pl.read_csv(REP / "loop_round1_trades.csv", try_parse_dates=True)
    _eq = pl.concat([
        _t.group_by("period", "trade_date").agg(pl.col("net_bps").sum().alias("day")).with_columns(pl.lit("baseline rule").alias("rule")),
        _t.group_by("period", "trade_date").agg(pl.col("filtered_bps").sum().alias("day")).with_columns(pl.lit("with round-1 rules").alias("rule")),
    ]).sort("period", "rule", "trade_date").with_columns(pl.col("day").cum_sum().over("period", "rule").alias("cum_bps"))
    _c = alt.Chart(_eq).mark_line(interpolate="step-after").encode(
        x=alt.X("trade_date:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=0, tickCount=4)),
        y=alt.Y("cum_bps:Q", title="Cumulative net bps"),
        color=alt.Color("rule:N", scale=alt.Scale(domain=["baseline rule", "with round-1 rules"], range=[GREY, GREEN]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["period", "rule", "trade_date", "cum_bps"],
    ).properties(width=230, height=230).facet(column=alt.Column("period:N", title=None, sort=["tune", "Q3 2026", "Oct-Nov 2026"])).resolve_scale(x="independent", y="independent")
    mo.vstack([
        mo.md("## Round 1, before and after: equity curves"),
        _c,
        mo.md("Equal size per trade, summed in bps. On the tune months the new rules turn a steady loss into a steady gain. "
              "On the two held-out periods they mostly **flatten the losses** (Q3: −3,121 → −143 bps; Oct–Nov: −721 → −92 bps) rather than make money."),
    ])
    return


@app.cell(hide_code=True)
def _():
    _s = pl.read_csv(REP / "loop_round1_steps.csv")
    _order = _s["step"].unique(maintain_order=True).to_list()
    _per = alt.Scale(domain=["tune", "Q3 2026", "Oct-Nov 2026"], range=[BLUE, AMBER, GREY])
    _pf = alt.Chart(_s).mark_line(point=alt.OverlayMarkDef(size=70)).encode(
        x=alt.X("step:N", sort=_order, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=150)),
        y=alt.Y("profit_factor:Q", title="Profit factor", scale=alt.Scale(domain=[0.5, 1.6])),
        color=alt.Color("period:N", scale=_per, legend=alt.Legend(orient="top", title=None)),
        tooltip=["period", "step", "trades", "win_rate", "profit_factor", "tail_losers"]).properties(width=380, height=280)
    _tgt = alt.Chart(pl.DataFrame({"y": [1.5]})).mark_rule(color=GREEN, strokeDash=[6, 4], size=2).encode(y="y:Q")
    _one = alt.Chart(pl.DataFrame({"y": [1.0]})).mark_rule(color="#e5ece8", opacity=0.4).encode(y="y:Q")
    _tl = alt.Chart(_s).mark_bar().encode(
        x=alt.X("step:N", sort=_order, title=None, axis=alt.Axis(labels=False, ticks=False)),
        xOffset=alt.XOffset("period:N", sort=["tune", "Q3 2026", "Oct-Nov 2026"]),
        y=alt.Y("tail_losers:Q", title="Tail losers left"), color=alt.Color("period:N", scale=_per, legend=None),
        tooltip=["period", "step", "tail_losers", "trades"]).properties(width=380, height=110)
    mo.vstack([
        mo.md("## What each kept rule did, one at a time"),
        mo.hstack([_pf + _tgt + _one, mo.Html("<div style='max-width:330px;font-size:0.9em'>" + mo.md("""
Each step adds one rule to the ones before it.

- **Slow run-up (≥ 10 min)** was the big win on tune, and it **hurt Oct–Nov** (0.86 → 0.60). The strongest clue is not stable.
- **Heavy red bar** helped in all three periods.
- **Early bailout** lifts every period and removes most remaining tail losers.

Tail losers left (tune / Q3 / Oct–Nov): 55 / 25 / 11 → 5 / 6 / 3, falling at every step. Profit factor does not follow, because the skipped trades include winners.

<small>Holdout lines are post-hoc attribution after the search closed. They must not be used to pick round 2.</small>
""").text + "</div>")], justify="start", align="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def _():
    _t = pl.read_csv(REP / "loop_round1_trades.csv", try_parse_dates=True).with_columns(
        pl.when(pl.col("net_bps") <= -200).then(pl.lit("tail loser (≤ −2%)"))
        .when(pl.col("net_bps") <= 0).then(pl.lit("small loser")).otherwise(pl.lit("winner")).alias("outcome"),
        pl.col("skipped_by").fill_null("kept (traded)").alias("fate"))
    _g = _t.group_by("period", "fate", "outcome").len()
    _fates = ["kept (traded)", "1 slow run-up (>= 10 min)", "2 entry near high (<= 150 bps below)", "3 heavy red bar (>= 0.6x)"]
    _c = alt.Chart(_g).mark_bar().encode(
        y=alt.Y("fate:N", sort=_fates, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("len:Q", title="Trades", stack="normalize", axis=alt.Axis(format="%")),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["winner", "small loser", "tail loser (≤ −2%)"], range=[GREEN, "#e0b25a", RED]), legend=alt.Legend(orient="top", title=None)),
        row=alt.Row("period:N", title=None, sort=["tune", "Q3 2026", "Oct-Nov 2026"]),
        tooltip=["period", "fate", "outcome", "len"]).properties(width=330, height=80)
    _h = _t.with_columns(pl.col("net_bps").clip(-800, 600).alias("net"))
    _hist = alt.Chart(_h).transform_filter(alt.datum.period != "tune").mark_bar(opacity=0.75).encode(
        x=alt.X("net:Q", bin=alt.Bin(step=50), title="Net bps per trade (holdout, clipped at −800 / +600)"),
        y=alt.Y("count():Q", title="Trades"),
        color=alt.Color("traded:N", scale=alt.Scale(domain=["skipped by a rule", "still traded"], range=[RED, BLUE]), legend=alt.Legend(orient="top", title=None)),
    ).transform_calculate(traded="datum.skipped_by == null ? 'still traded' : 'skipped by a rule'").properties(width=330, height=260)
    mo.vstack([
        mo.md("## Which trades the rules removed"),
        mo.hstack([_c, _hist], justify="start", align="start", gap=2),
        mo.md("Left: the mix of outcomes in each group (share of trades). The skipped groups hold more red than the kept group, so the rules aim at the right trades, "
              "but 40–50% of what they skip is still winners. Right: on the held-out months the skipped trades sit in the left tail and in the middle, not only in the tail."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why Chris may see about +30 bps and we don't

    Most likely causes, in order:

    1. **Entry.** We short at the next bar's open after the red bar. Chris's spec places a **protected limit order at the armed price** and does not chase. That skips trades that keep running (the tail losers) and fills better on the ones that turn. This alone could flip the result.
    2. **Feed and the open price.** His +5% is measured from IBKR's official open; ours from Alpaca's consolidated opening print.
    3. **Different trades.** His good days have 8–10 trades; this universe gives 2–4 a day at \$50+.
    4. **No news screen here.** His red-flag list removes strong positive-news days, the worst category in his data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What to change next (ranked)

    | # | change | why | how we'd judge it |
    |---|---|---|---|
    | 1 | **Protected-limit entry** at the red bar's close; fill only if price trades back up to it within N minutes | The most likely gap vs Chris; naturally skips runaway squeezes | Same scorer and holdout; count trades skipped and fill quality |
    | 2 | **Catalyst exclusion** using Alpaca news headlines (from Apr 2026) | Positive news shocks continue; Chris's own red-flag list | Tail losers on news vs no-news days |
    | 3 | **Compare 5–10 of Chris's logged trades** minute by minute against this engine | Settles feed, open and entry differences directly | Same entries and P&L within a few bps |
    | 4 | **Keep the rules that held up**: heavy red bar + early bailout; **drop** the 10-minute rule (it hurt Oct–Nov) | Round 1 attribution | Start round 2 from these, on top of #1 |
    | 5 | **Pre-market data** for the forecasters and run-up features | Today they see 2–45 bars | Forecaster AUC on holdout above 0.65 |
    | 6 | **Jev cross-ticker ranking** on busy mornings | Pick the cleanest 2–3 of many candidates | Needs your Jev key |
    | 7 | **CLM-8B gate** | Experimental, text model | Only after 1–3; ~8–9 GB download, needs your OK |

    Next loop round: start the search from the protected-limit entry (#1) and let the loop test #2 and #4 on top.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Did we use autoresearch or the meta-harness?

    **Before today, no.** Every test so far was run by hand.

    - **Autoresearch-style loop: now yes**, a small first version in `research/loop/` (one editable file, fixed scorer, keep or undo, sealed holdout, ledger). Round 1 above was run by Claude following `program.md`. It can now run unattended for 20+ ideas per round.
    - **EC1 meta-harness: not yet.** It adds a Pareto frontier, a falsifiable change manifest per candidate and a tamper-evident ledger, and it routes judgement calls (which idea to try, whether evidence supports a change) to **Jev**. It refuses to run without a live Jev key by design. With your key, the loop above becomes its scorer and candidate surface.

    **Asks:** your Jev key (for the meta-harness and the ranking test), and a few of Chris's logged trades with times (for #3).
    """)
    return


if __name__ == "__main__":
    app.run()
