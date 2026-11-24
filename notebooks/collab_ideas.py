# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "marimo>=0.25.2",
#     "polars>=2.1.0",
#     "altair>=6.3.0",
#     "marimo-studio==0.4.1"
# ]
#
# [tool.marimo-studio]
# default = "board"
#
# [tool.marimo-studio.cells]
# cell-2 = {ref = "cell:v1:3863d588251ae2c57be73feb81242395f8498399aab884709b3482bc8d633353:dc1fde9b5a009e81339efba8dc321996830c8ecffd5fc97b3a422afe0036fe2c:0"}
# cell-3 = {ref = "cell:v1:96d91afb3f813259b99823147d8f002bcbab1aac8606bfc3a381bac9bd4a4f01:adb9d700269121a5596efecc3d00b4f5c4517fad1635e085919bb43437d99047:0"}
# cell-4 = {ref = "cell:v1:4bc1eae35090401529a8f658e1c4553018db16b790cb37a373643af0ea9d5ad3:4bc1eae35090401529a8f658e1c4553018db16b790cb37a373643af0ea9d5ad3:0"}
# cell-5 = {ref = "cell:v1:e1915a0a4eb10b7a76431a03087097f67b9685ab8a0e225792ff9ae7a3648932:e1915a0a4eb10b7a76431a03087097f67b9685ab8a0e225792ff9ae7a3648932:0"}
# cell-6 = {ref = "cell:v1:fa3eeaa9b1140d1c6ec4b1dc4f790cef2b37cba1e256bd8e65a55fd4aa066679:5706cead5bce5818b503df9318fefa36c44d4c3fa73a305248776c8b5c9a7e1f:0"}
# cell-7 = {ref = "cell:v1:1ab1a3158ef3c755036949c9eb9cc6cfcd4944f604b17a12ea3110a063694923:0905b6eadad64b500e412ffc67f1016e9f3711afcdb23ca5ae7333a9ef8aa909:0"}
# cell-8 = {ref = "cell:v1:0f7276fe8ef17f538df9786225da8161c20eb49836989550dd0ec7372898b989:0a2b5cda7e748d1e5edcbcb992eda0f3e69c41555a5776abcf409573007af01f:0"}
# cell-9 = {ref = "cell:v1:f4a02f0cd3fa2409d06a04839d605c3a18f5c4e1078c2544f60aaaa58910d3fb:1a038f5ea0163ad6557300cb79848a21658e2c1e1495a600c2409d285bf37548:0"}
# ///
"""Collaboration board for the spike-short group: why, results so far, idea board, InfraNodus gap map.

Run:  uv run marimo run notebooks/collab_ideas.py
Reads docs/trading/collab/infranodus_gaps_2026-11-24.json and the saved report numbers; it re-runs nothing.
Paper research only. Nothing here places orders.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Spike short: collaboration board")

with app.setup:
    import json
    from pathlib import Path

    import altair as alt
    alt.theme.enable("dark")
    import marimo as mo
    import polars as pl

    ROOT = Path(__file__).resolve().parents[1]
    GAPS = json.loads((ROOT / "docs/trading/collab/infranodus_gaps_2026-11-24.json").read_text())


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'collab_ideas'
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
    # Spike short: what we're building and what to try next

    **Why.** Chris and Mark have watched liquid, higher-priced stocks spike in the first 15-30 minutes and
    come back down. Mark's tests find a *small* edge with a lot of month-to-month noise. The aim of this
    group is not one magic rule: it is **5-10 explainable edges** that hold out of sample and can be traded
    with our own money first.

    **How.** A shared research lab where every idea is a small, testable change: marimo notebooks
    (the agent and we see the same live state), Polars + DuckDB data, PyBroker parity checks,
    OpenTelemetry traces of every trade decision, and fast decision models (Jev, CLM-8B) where a rule
    needs judgement. Anything that survives the holdout gets compiled from the Python lab to Rust/C++ for live use.
    """)
    return


@app.cell(hide_code=True)
def _():
    _kpi = lambda label, val, note: mo.stat(value=val, label=label, caption=note, bordered=True)
    mo.vstack([
        mo.md("## Results so far (Alpaca SIP 1-minute bars, May 2025 - Nov 2026)"),
        mo.hstack([
            _kpi("Baseline profit factor", "0.75-0.91", "target 1.5+"),
            _kpi("Baseline win rate", "43-51%", "target 60-70%"),
            _kpi("Tail losers", "~15% of trades", "60-68% of losing bps"),
            _kpi("Best holdout after vetoes", "PF 0.95-0.97", "tail losers cut ~75%"),
        ], widths="equal"),
        mo.md("""
- **Chris is right about the damage:** 1-2 continuation losers a day (<= -200 bps) do most of the harm.
- **Speed is the clearest tell:** tail losers hit +5% a median **5 min** after the open, winners in **12**.
- **ML ranks but doesn't rescue:** LightGBM / CatBoost / TabPFN reach AUC 0.6-0.7 on tail losers, PF only ~0.9-1.0.
  Kronos and Chronos add nothing reliable on so few pre-entry bars.
- **Keep-or-undo loop:** 4 of 10 ideas kept; tune PF 1.49, holdout 0.95-0.97. Part of the gain was fit to the tune period.
- **Matches Mark's read:** a small, noisy edge. The biggest open question is the *entry*: Chris's protected
  limit at the armed price and 30-second bars, which this lab hasn't modelled yet.

Details: `docs/trading/reports/` and the slide deck `notebooks/results_deck.py`.
        """),
    ])
    return


@app.cell(hide_code=True)
def _():
    IDEAS = pl.DataFrame(
        [
            ("Protected-limit entry at the armed price", "Entry", "Re-run the baseline with Chris's rule: stage a limit at the armed price, never chase. Then rerun every filter on those trades.", "spike_lab + PyBroker", "next", "Explains the gap vs Chris's +30 bps?"),
            ("30-second bars for the arm", "Data", "Pull Massive 30 s bars (or rebuild from Alpaca trades) for a sample month; compare arm minute and fill price vs 1-min.", "Massive / Alpaca trades", "next", "Mark's edge is measured on 30 s"),
            ("Run-speed gate", "Entry", "Skip if +5% came < 10 min after the open (kept in loop round 1). Re-test on the new entry.", "research/loop", "lead", "Fast runs = tail"),
            ("Red / yellow / green pre-open gate as size", "News", "Red = no trade, yellow = half size, no news = full. Freeze the taxonomy first, apply forward.", "Jev L2 classifier, L13 kill switch", "idea", "Explainable by design"),
            ("Earnings shock strength split", "News", "Keep ordinary earnings spikes (67% win, PF 2.02 in Chris's data), exclude strong positive shocks (TXN, TGT, BBY).", "Jev L3 weighted score", "idea", "Small sample: 5 strong shocks"),
            ("Relative dollar volume at the spike", "Price", "Thin relative volume = short covering (fade it); heavy = news continuation (skip). From the InfraNodus gap map.", "Polars feature + loop", "idea", "Bridges continuation <-> market edge"),
            ("8-K / 10-K keyword veto", "Filings", "Scan new filings for fraud, delayed financials, auditor change before the open; veto shorts on flagged names.", "EDGAR full-text + Jev L8 blind read", "idea", "Mark's idea"),
            ("Language as a signal", "News", "Embed each headline/filing, find the nearest past events, rank by their average next-30-min move.", "CLM-8B, InfraNodus", "idea", "Rank by expected move (Mark)"),
            ("Form 4 insider edge as edge #2", "Filings", "Mark and Chris already turn Form 4 filings into orders. Port it into the same loop + scorecard.", "existing code + this lab", "idea", "Second independent edge"),
            ("Review every tail loser on a chart", "Review", "Step through each loser bar by bar, tag a cause, look for the pattern a rule could catch.", "trade_replay.py (Lightweight Charts)", "built", "Human eyes on the tail"),
            ("Massive for data, IB only for orders", "Ops", "Stop asking IB for 400 opening prices; precompute the universe + gates pre-open, send only orders.", "Massive + IB TWS API", "idea", "Fixes the 354/300 errors"),
            ("Edge scoreboard", "Process", "One table, one fixed scorer, keep-or-undo loop per edge; holdout opened once.", "research/loop", "built", "5-10 edges, same discipline"),
        ],
        schema=["idea", "theme", "test", "tools", "status", "why"],
        orient="row",
    )
    theme = mo.ui.multiselect(options=sorted(IDEAS["theme"].unique().to_list()), label="Theme")
    status = mo.ui.multiselect(options=["built", "lead", "next", "idea"], label="Status")
    mo.vstack([mo.md("## Idea board"), mo.hstack([theme, status], justify="start")])
    return IDEAS, status, theme


@app.cell(hide_code=True)
def _(IDEAS, status, theme):
    _df = IDEAS
    if theme.value:
        _df = _df.filter(pl.col("theme").is_in(theme.value))
    if status.value:
        _df = _df.filter(pl.col("status").is_in(status.value))
    mo.ui.table(_df, selection=None, page_size=12)
    return


@app.cell(hide_code=True)
def _():
    _url = GAPS["graph_url"]
    mo.vstack([
        mo.md(f"""
## Gap map (InfraNodus)

The call transcript, Nov 14 memo, Chris's emails and today's reports were condensed into
{GAPS['stats']['nodes']} concepts and {GAPS['stats']['clusters']} topic clusters. InfraNodus calls the
structure **{GAPS['stats']['diversity']}**: lots of threads, few connections between them. The gaps
below are where the conversation hasn't joined two topics yet, which is where new tests come from.

The live graph is below. If it shows a login page, the graph is still private: open it in
[InfraNodus]({_url}) or switch on its public share link first.
        """),
        mo.Html(f'<iframe src="{_url}" style="width:100%;height:620px;border:1px solid #ddd;border-radius:8px" loading="lazy"></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def _():
    _cl = pl.DataFrame(GAPS["clusters"])
    _chart = (
        alt.Chart(_cl)
        .mark_bar()
        .encode(
            y=alt.Y("name:N", sort="-x", title=None),
            x=alt.X("influence:Q", title="Influence in the discussion (%)"),
            color=alt.condition(
                alt.FieldOneOfPredicate("name", ["Broker errors", "Continuation issues", "Market edge", "Entry modifier", "Trade losses"]),
                alt.value("#d9534f"), alt.value("#8aa1b4")),
            tooltip=["name", "words", "influence", "share_nodes"],
        )
        .properties(height=300, width=420)
    )
    _gaps = "\n".join(f"- {g}" for g in GAPS["gaps"])
    mo.hstack([
        mo.ui.altair_chart(_chart),
        mo.md(f"**Structural gaps (red = clusters on a gap):**\n\n{_gaps}\n\n"
              f"**Gateway words:** {', '.join(GAPS['gateways'])}"),
    ], widths=[1.2, 1])
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({
        f"Bridge {i + 1}: {b['title']}": mo.md(f"*{b['gap']}*\n\n{b['idea']}\n\n**Test:** {b['test']}")
        for i, b in enumerate(GAPS["bridging_ideas"])
    })
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tools in the kit

    | Tool | What it does here | Link |
    |---|---|---|
    | marimo | Reactive notebooks the agent can drive live; dashboards, slides, static exports | [marimo.io](https://marimo.io) |
    | PyBroker | Independent backtest engine; reproduced 136/136 trades as a parity check | [pybroker](https://www.pybroker.com) |
    | Alpaca | Free SIP 1-minute bars back to 2016, news from 2026 | [alpaca.markets](https://alpaca.markets/docs/market-data/) |
    | LightGBM / CatBoost / TabPFN | Tabular models for "will this short become a tail loser?" | [LightGBM](https://lightgbm.readthedocs.io) · [CatBoost](https://catboost.ai) · [TabPFN](https://github.com/PriorLabs/TabPFN) |
    | Kronos / Chronos-Bolt | Price-path foundation models (candles in, paths out) | [Kronos](https://github.com/shiyu-coder/Kronos) · [Chronos](https://github.com/amazon-science/chronos-forecasting) |
    | CLM-8B | Picks the option closest to the question in embedding space; reranker for news/filings | [Hugging Face](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) |
    | Jev | Millisecond decision model: yes/no gates, classifiers, scores, kill switches | [Jev Trading Lab (13 levels)](https://jeromyjsmith.github.io/jev-trading-lab/) · [First-Hour Desk](https://jeromyjsmith.github.io/jev-first-hour-desk/) |
    | Lightweight Charts | TradingView-style candles for bar-by-bar trade replay | [tradingview.github.io/lightweight-charts](https://tradingview.github.io/lightweight-charts/) |
    | InfraNodus | Text-network gap analysis of notes, news and filings | [infranodus.com](https://infranodus.com) |
    | OpenTelemetry | A trace span for every rule decision, so a losing trade can be read step by step | [opentelemetry.io](https://opentelemetry.io) |

    **Notebooks in this lab:** `spike_short_dashboard.py` (engine + ablations), `continuation_loser_detector.py`,
    `model_lab.py`, `trade_replay.py` (Lightweight Charts replay), `results_deck.py` (slides).
    """)
    return


if __name__ == "__main__":
    app.run()
