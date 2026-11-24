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
# ///
"""Brainstorm board for the spike-short group: what we found, open ideas with examples, gap analysis, tools.

Run:  uvx --with "marimo-studio[deno]" marimo edit notebooks/brainstorm_board.py --sandbox --watch --no-token
Reads saved results only (data/reports, research/loop, docs/trading/collab); it re-runs no backtest.
Paper research only. Nothing here places orders.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Spike short: brainstorm board")

with app.setup:
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    ROOT = Path(__file__).resolve().parents[1]
    REPORTS = ROOT / "data" / "reports"
    LOOP = ROOT / "research" / "loop"
    COLLAB = ROOT / "docs" / "trading" / "collab"
    INTRADAY = ROOT / "data" / "alpaca" / "intraday.parquet"
    alt.theme.enable("dark")

    LWC = "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"

    def candle_page(bars: list[dict], vols: list[dict], markers: list[dict], lines: list[dict], title: str) -> str:
        """A full-width TradingView Lightweight Charts page (dark) for one trade morning."""
        return f"""<!doctype html><html><head><meta charset='utf-8'>
    <style>html,body{{margin:0;height:820px;background:#0f1513;color:#e5ece8;font:13px system-ui,sans-serif}}
    #t{{padding:8px 12px;font-weight:600}}#c{{position:relative;width:100%;height:780px}}</style>
    <script src='{LWC}'></script></head><body><div id='t'>{title}</div><div id='c'></div><script>
    const el=document.getElementById('c');
    const chart=LightweightCharts.createChart(el,{{autoSize:true,
      layout:{{background:{{type:'solid',color:'#0f1513'}},textColor:'#c9d4cf'}},
      grid:{{vertLines:{{color:'#1e2a26'}},horzLines:{{color:'#1e2a26'}}}},
      rightPriceScale:{{borderColor:'#2a3631'}},timeScale:{{borderColor:'#2a3631',timeVisible:true,secondsVisible:false}},
      crosshair:{{mode:0}}}});
    const s=chart.addCandlestickSeries({{upColor:'#ef7b66',downColor:'#5cc2a2',borderVisible:false,wickUpColor:'#ef7b66',wickDownColor:'#5cc2a2'}});
    s.setData({json.dumps(bars)});
    const v=chart.addHistogramSeries({{priceFormat:{{type:'volume'}},priceScaleId:'vol',color:'#3a4a44'}});
    chart.priceScale('vol').applyOptions({{scaleMargins:{{top:0.82,bottom:0}}}});
    v.setData({json.dumps(vols)});
    s.setMarkers({json.dumps(markers)});
    for(const l of {json.dumps(lines)}) s.createPriceLine(l);
    chart.timeScale().fitContent();
    </script></body></html>"""


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'brainstorm_board'
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
def purpose():
    mo.md(r"""
    # Spike short: brainstorm board

    **What we trade.** Liquid US stocks ($50+, ATR ≥ 3%, $100M+ daily dollar volume) that open 0.25–1.0 ATR up,
    then run another 4–5% before 10:00 ET. Short after the first red 1-minute bar, cover 20 minutes later.

    **The problem.** Most trades fade, but one or two a day keep running 2–5% and erase the winners. Stops tried
    so far cut winners as often as losers. Mark measures a small edge on 30-second bars and says it may be noise.

    **What this group wants.** Five to ten edges we can explain, test out of sample, and trade with our own
    money first. This page lists what has been tested, what failed, and what to try next, with a concrete
    example for each idea. Every number comes from a saved run; links point to the code and tools.
    """)
    return


@app.cell(hide_code=True)
def results():
    baseline = pl.read_csv(REPORTS / "baseline_scorecard.csv").select(
        "period", "trades", "win_rate", "profit_factor", "avg_net_bps", "tail_losers", "pct_days_positive"
    )
    _rev = json.loads((LOOP / "reveals.tsv").read_text().splitlines()[1].split("\t", 3)[3])
    loop_reveal = pl.DataFrame([{"run": k, **v} for k, v in _rev.items()])
    ledger = pl.read_csv(LOOP / "results.tsv", separator="\t").select(
        "decision", "hypothesis", "trades", "win_rate", "profit_factor", "tail_losers"
    )
    return baseline, ledger, loop_reveal


@app.cell(hide_code=True)
def results_view(baseline, ledger, loop_reveal):
    _best = loop_reveal.filter(pl.col("run").str.contains("filtered"))
    mo.vstack([
        mo.md("## 1. Where we are\nAlpaca SIP 1-minute bars, May 2025 – Nov 2026. Targets: win rate 60–70%, profit factor ≥ 1.5."),
        mo.hstack([
            mo.stat(value=f"{baseline['profit_factor'].min():.2f}–{baseline['profit_factor'].max():.2f}",
                    label="Baseline profit factor", caption="tune, Q3 2026, Oct–Nov 2026", bordered=True),
            mo.stat(value=f"{baseline['win_rate'].min():.0f}–{baseline['win_rate'].max():.0f}%",
                    label="Baseline win rate", caption="target 60–70%", bordered=True),
            mo.stat(value=f"{_best['profit_factor'].min():.2f}–{_best['profit_factor'].max():.2f}",
                    label="Best filter set, untouched data", caption="research loop round 1", bordered=True),
            mo.stat(value=f"{int(baseline['tail_losers'].sum())}", label="Tail losers (≤ −2%)",
                    caption="all periods, baseline", bordered=True),
        ], widths="equal"),
        mo.ui.table(baseline, selection=None, label="Baseline by period"),
        mo.accordion({
            "Research loop round 1: every hypothesis tried (keep / undo)": mo.ui.table(ledger, selection=None),
            "Round 1 filters on untouched data": mo.ui.table(loop_reveal, selection=None),
        }),
    ])
    return


@app.cell(hide_code=True)
def idea_list():
    ideas = pl.DataFrame([
        {
            "idea": "Protected-limit entry",
            "source": "Chris's spec",
            "example": "Replay fills at the next bar's open and its winners average +164 bps; Chris reports ~+30 bps. A limit at the armed price that never chases would skip the fastest fades and may also skip some continuations.",
            "test": "Rebuild fills in PyBroker; report fill rate, win rate and PF for tune, Q3 and Oct–Nov.",
            "data": "Have it (Alpaca 1-min)",
            "status": "next",
        },
        {
            "idea": "30-second bars",
            "source": "Mark",
            "example": "Mark's edge is measured on Massive 30-second bars. On 1-minute bars the first red bar can hide a 30-second bounce that already happened.",
            "test": "Pull 30-s bars for a sample month, rerun the arm, compare entries trade by trade.",
            "data": "Massive or Alpaca trades",
            "status": "needs data",
        },
        {
            "idea": "Participation decay",
            "source": "InfraNodus gap analysis",
            "example": "CAR, 2026-04-21: armed at +6%, red bar on 2.5x volume, kept going to −10.2%. Was volume per minute still flat after the arm, i.e. a buyer still working an order?",
            "test": "Feature: volume slope over the 3 bars after the arm. Exit on decay instead of on price.",
            "data": "Have it",
            "status": "idea",
        },
        {
            "idea": "Catalyst type",
            "source": "Emails, call",
            "example": "Strong positive earnings shocks: 5 trades, −109.5 bps, PF 0.28. No-news spikes reverted best. Earnings spikes overall PF 2.02.",
            "test": "Tag each spike with Alpaca news (since Apr 2026); compare continuation rate by tag.",
            "data": "Alpaca news",
            "status": "next",
        },
        {
            "idea": "SEC filing scan",
            "source": "Mark",
            "example": "8-K or 10-K language such as 'delay in filing', 'auditor', 'material weakness' can appear before the headline. Mark and Chris already turn Form 4 insider filings into orders.",
            "test": "Keyword or CLM-8B score on filings from the prior 2 days; veto shorts above a threshold.",
            "data": "EDGAR full-text search (free)",
            "status": "idea",
        },
        {
            "idea": "News ranked by past move",
            "source": "Mark",
            "example": "Rank each headline by the average move after similar past headlines, not by sentiment.",
            "test": "Embed headlines, nearest-neighbour past events, average 20-min move; use as a veto score.",
            "data": "Alpaca news + bars",
            "status": "idea",
        },
        {
            "idea": "Market opening drive",
            "source": "Loop round 2 list",
            "example": "If SPY and QQQ are both up strongly in the first 15 minutes, single-name spikes may have more buyers behind them.",
            "test": "Skip when SPY open-to-arm move > X bps; tune X on Jul 2025 – Jun 2026 only.",
            "data": "Have it (SPY/QQQ cached)",
            "status": "next",
        },
        {
            "idea": "Break of the prior green bar's low",
            "source": "Loop round 2 list",
            "example": "A stricter arm: enter only when price trades below the low of the last green bar, not on any red close.",
            "test": "Alternative arm rule in the state machine; compare tail-loser count.",
            "data": "Have it",
            "status": "next",
        },
        {
            "idea": "Tier-specific thresholds",
            "source": "Loop round 2 list",
            "example": ">$1B names may need a smaller run-up (4%) and $100M–1B names a larger one (5%+).",
            "test": "Separate trigger per tier; check both tiers keep 100+ trades.",
            "data": "Have it",
            "status": "next",
        },
        {
            "idea": "Fast decision model as a veto",
            "source": "Call",
            "example": "Jev returns a yes/no veto in milliseconds from a fixed schema (gap, run-up, red-bar volume, catalyst tag). Rules stay explainable because the inputs and options are fixed.",
            "test": "Wire src/spike_lab/models/jev_hook.py; score against the same tune/holdout split.",
            "data": "Jev API key",
            "status": "needs key",
        },
        {
            "idea": "Fair value gap / 5-bar box breakdown arm",
            "source": "Casper Trading and Soheil PKO videos (tested 2026-11-24)",
            "example": "Instead of the first red bar, arm only on a bearish 3-bar fair value gap, a close below the last 5-bar box, or a close below the last green bar's low. FVG at 5%: tune PF 1.19 (272 trades) but Q3 0.65 and Oct-Nov 0.54. A stop above the pattern plus a 2R target cut tail losers by about 60% but dropped win rate to 30-44%.",
            "test": "Done: scripts/entry_variants.py, data/reports/entry_variants_2026-11-24.csv. Does not hold on untouched data; keep only as a confirmation feature for the veto model.",
            "data": "Have it",
            "status": "tested",
        },
    ])
    return (ideas,)


@app.cell(hide_code=True)
def idea_filter(ideas):
    status_pick = mo.ui.multiselect(options=sorted(ideas["status"].unique().to_list()),
                                    value=sorted(ideas["status"].unique().to_list()), label="Status")
    mo.vstack([mo.md("## 2. Ideas to test\nEach has a concrete example and the test that would settle it."), status_pick])
    return (status_pick,)


@app.cell(hide_code=True)
def idea_cards(ideas, status_pick):
    _cards = [
        mo.callout(
            mo.md(f"**{r['idea']}** · *{r['source']}* · `{r['status']}`\n\n"
                  f"**Example.** {r['example']}\n\n**Test.** {r['test']}\n\n**Data.** {r['data']}"),
            kind={"next": "success", "idea": "neutral", "needs data": "warn", "needs key": "warn", "tested": "danger"}.get(r["status"], "neutral"),
        )
        for r in ideas.filter(pl.col("status").is_in(status_pick.value)).iter_rows(named=True)
    ]
    mo.vstack(_cards)
    return


@app.cell(hide_code=True)
def gap_load():
    gaps = json.loads((COLLAB / "brainstorm_infranodus_2026-11-24.json").read_text())
    _names = [c.split(":")[0] for c in gaps["clusters"]]
    _links = [tuple(g.split(" <-> ")) for g in gaps["gaps"]]
    gap_edges = pl.DataFrame({"from": [a for a, _ in _links], "to": [b for _, b in _links]})
    gap_clusters = pl.DataFrame({"cluster": _names, "terms": [c.split(": ", 1)[1] for c in gaps["clusters"]],
                                 "in_gap": [any(n in l for l in _links) for n in _names]})
    return gap_clusters, gap_edges, gaps


@app.cell(hide_code=True)
def gap_view(gap_clusters, gaps):
    gap_pick = mo.ui.table(gap_clusters, selection="multi", label="Topic clusters (amber = sits on a gap). Select rows to see their gaps.")
    _url = "https://infranodus.com/ojeromyo/spike-short-strategy-gaps"
    mo.vstack([
        mo.md(f"""## 3. Gap analysis (InfraNodus)
The call, Chris's brief, the emails and our results as a text network: [strategy gap graph]({_url}). The full news
and gap analysis, with both graphs, answers from the graphs and the headline-by-outcome analysis, is in
**News and gap analysis** (notebook 4). Clusters on either side of a **gap** are topics we discuss a lot but never connect.
The biggest one: we talk about *decision gates* (vetoes, kill switches) and about *timing* (minutes after entry)
but have no rule that uses the first minutes to make a gate decision with an explanation attached."""),
        mo.Html(f'<iframe src="{_url}" style="width:100%;height:92vh;min-height:880px;display:block;border:1px solid #2a3631;'
                'border-radius:8px;background:#0f1513" loading="lazy" allowfullscreen></iframe>'),
        gap_pick,
        mo.md("**Questions the gaps raise**\n\n" + "\n".join(f"- {q}" for q in gaps["research_questions"])),
        mo.md("**Hypotheses that bridge them** (rewritten as tests)\n\n" + "\n".join(f"- {b}" for b in gaps["bridges_rewritten"])),
    ])
    return (gap_pick,)


@app.cell(hide_code=True)
def gap_pick_view(gap_edges, gap_pick):
    _sel = gap_pick.value
    if _sel is None or not len(_sel):
        _out = mo.ui.table(gap_edges, selection=None, label="All gaps between clusters")
    else:
        _names = list(pl.DataFrame(_sel)["cluster"])
        _e = gap_edges.filter(pl.col("from").is_in(_names) | pl.col("to").is_in(_names))
        _out = mo.vstack([mo.md(f"**Gaps touching {', '.join(_names)}:** {_e.height}"), mo.ui.table(_e, selection=None)])
    _out
    return


@app.cell(hide_code=True)
def tail_example():
    worst = pl.read_csv(REPORTS / "worst_15_tune.csv").select(
        "symbol", "trade_date", "tier", "net_bps", "mae_bps", "red_vol_ratio", "runup_pct_at_arm", "close_vs_avwap_bps"
    )
    mo.vstack([mo.md("## 4. What a continuation loser looks like\nThe 15 worst trades in the tune year. "
                     "Open any of them in `notebooks/trade_replay.py` to step through it bar by bar."),
               mo.ui.table(worst, selection=None)])
    return


@app.cell(hide_code=True)
def chart_pick():
    _worst = pl.read_csv(REPORTS / "worst_15_tune.csv", try_parse_dates=True)
    _best = (pl.read_csv(REPORTS / "loop_round1_trades.csv", try_parse_dates=True)
             .filter(pl.col("period") == "tune").sort("net_bps", descending=True).head(5))
    chart_trades = pl.concat([
        _worst.select("symbol", "trade_date", "entry_ts", "net_bps").with_columns(kind=pl.lit("continuation loser")),
        _best.select("symbol", "trade_date", "entry_ts", "net_bps").with_columns(kind=pl.lit("clean fade")),
    ])
    _labels = {f"{r['symbol']} {r['trade_date']} · {r['kind']} · {r['net_bps']:+.0f} bps": i
               for i, r in enumerate(chart_trades.iter_rows(named=True))}
    chart_choice = mo.ui.dropdown(_labels, value=next(iter(_labels)), label="Trade", full_width=True)
    mo.vstack([mo.md("## 5. On the chart\nTradingView Lightweight Charts from the cached Alpaca 1-minute bars. "
                     "Red candles are up bars (bad for the short), green are down bars. Arrows mark the short entry and "
                     "the 20-minute cover; dashed lines show the official open and the +5% arm level. "
                     "Scroll to zoom, drag to pan."), chart_choice])
    return chart_choice, chart_trades


@app.cell(hide_code=True)
def chart_view(chart_choice, chart_trades):
    import datetime as _dt
    _t = chart_trades.row(chart_choice.value, named=True)
    _day = (pl.scan_parquet(INTRADAY).filter((pl.col("symbol") == _t["symbol"]) & (pl.col("trade_date") == _t["trade_date"]))
            .sort("ts").collect())
    _epoch = lambda ts: int(ts.replace(tzinfo=_dt.timezone.utc).timestamp())
    _bars = [{"time": _epoch(r["ts"]), "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"]}
             for r in _day.iter_rows(named=True)]
    _vols = [{"time": _epoch(r["ts"]), "value": r["volume"],
              "color": "#5a3b35" if r["close"] >= r["open"] else "#2d4d43"} for r in _day.iter_rows(named=True)]
    _entry = _t["entry_ts"]
    _cover = _entry + _dt.timedelta(minutes=20)
    _markers = [
        {"time": _epoch(_entry), "position": "aboveBar", "color": "#e3a948", "shape": "arrowDown", "text": "short"},
        {"time": _epoch(_cover), "position": "belowBar", "color": "#5cc2a2" if _t["net_bps"] > 0 else "#ef7b66",
         "shape": "arrowUp", "text": f"cover {_t['net_bps']:+.0f} bps"},
    ]
    _open = _day["open"][0] if _day.height else None
    _lines = [] if _open is None else [
        {"price": _open, "color": "#7d8b86", "lineStyle": 2, "lineWidth": 1, "title": "open"},
        {"price": round(_open * 1.05, 4), "color": "#e3a948", "lineStyle": 2, "lineWidth": 1, "title": "+5% arm"},
    ]
    _title = f"{_t['symbol']} · {_t['trade_date']} · {_t['kind']} · {_t['net_bps']:+.0f} bps"
    mo.iframe(candle_page(_bars, _vols, _markers, _lines, _title), width="100%", height="830px") if _day.height else mo.md("_No cached bars for this trade._")
    return


@app.cell(hide_code=True)
def next_build():
    mo.md(r"""
    ## 6. What to build next (round 2)

    | # | Work | Why | Output |
    |---|---|---|---|
    | 1 | Protected-limit entry in PyBroker | Closest to how Chris actually trades; likely explains +164 vs +30 bps | Fill rate, win rate, PF for tune / Q3 / Oct–Nov |
    | 2 | Research loop round 2 | Keep the heavy-red-bar skip and the early bail; drop the 10-minute rule; try ~10 ideas from section 2 | Keep/undo ledger; holdout opened once at the end |
    | 3 | Data-quality pass | Gaps, splits and halts in cached bars can create fake tail losers | List of trades that depend on bad bars |
    | 4 | Async live prototype (`src/live/`) | Alpaca websocket → screen → arm/entry state machine → paper-order stub, each with OpenTelemetry spans | Pure decision functions ready to port to Rust |

    Tuning uses Jul 2025 – Jun 2026 only. Q3 2026 and Oct–Nov 2026 are opened once, at the end.
    """)
    return


@app.cell(hide_code=True)
def tool_list():
    tools = pl.DataFrame([
        {"tool": "Jev Trading Lab", "use": "Where a millisecond decision model fits in the trade (13 levels)", "url": "https://jeromyjsmith.github.io/jev-trading-lab/"},
        {"tool": "Jev First-Hour Desk", "use": "Jev applied to the first trading hour", "url": "https://jeromyjsmith.github.io/jev-first-hour-desk/"},
        {"tool": "InfraNodus", "use": "Text-network gap analysis of notes, news and filings", "url": "https://infranodus.com"},
        {"tool": "CLM-8B", "use": "Embedding reranker for news or filing language", "url": "https://huggingface.co/Contrastive-LM/CLM-v0.1-8B"},
        {"tool": "marimo", "use": "Reactive notebooks; the agent edits the same live session we look at", "url": "https://marimo.io"},
        {"tool": "marimo Studio", "use": "Turns a notebook into a shareable page or slides", "url": "https://marimo-team.github.io/marimo-studio/"},
        {"tool": "PyBroker", "use": "Independent backtester; reproduced the replay 136/136 trades", "url": "https://www.pybroker.com"},
        {"tool": "Alpaca market data", "use": "SIP 1-minute bars and news", "url": "https://docs.alpaca.markets/docs/about-market-data-api"},
        {"tool": "Massive", "use": "Mark's data source: 30-second bars, long history", "url": "https://massive.com"},
        {"tool": "SEC EDGAR full-text search", "use": "Filing language before the headline", "url": "https://efts.sec.gov/LATEST/search-index?q=%22material%20weakness%22"},
        {"tool": "Lightweight Charts", "use": "Candles for bar-by-bar trade replay", "url": "https://tradingview.github.io/lightweight-charts/"},
        {"tool": "OpenTelemetry", "use": "A trace span for each rule decision", "url": "https://opentelemetry.io"},
    ])
    return (tools,)


@app.cell(hide_code=True)
def tools_view(tools):
    mo.md("## 7. Tools\n" + "\n".join(f"- **[{r['tool']}]({r['url']})**: {r['use']}" for r in tools.iter_rows(named=True)))
    return


@app.cell(hide_code=True)
def jev_sites():
    _frame = ('<iframe src="{u}" style="width:100%;height:92vh;min-height:900px;border:1px solid #2a3631;'
              'border-radius:8px;background:#0f1513;display:block" loading="lazy"></iframe>')
    mo.vstack([
        mo.md("## 8. Jev, live\nThe two Jev sites, full size. Scroll inside each one, or open it in its own tab: "
              "[Jev Trading Lab](https://jeromyjsmith.github.io/jev-trading-lab/) · "
              "[Jev First-Hour Desk](https://jeromyjsmith.github.io/jev-first-hour-desk/)."),
        mo.md("### Jev Trading Lab (13 decision levels)"),
        mo.Html(_frame.format(u="https://jeromyjsmith.github.io/jev-trading-lab/")),
        mo.md("### Jev First-Hour Desk"),
        mo.Html(_frame.format(u="https://jeromyjsmith.github.io/jev-first-hour-desk/")),
    ], gap=1)
    return


if __name__ == "__main__":
    app.run()
