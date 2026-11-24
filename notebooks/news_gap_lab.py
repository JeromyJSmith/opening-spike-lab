# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "polars>=2.1.0",
#     "altair>=6.3.0",
# ]
# ///
"""News and gap analysis: two InfraNodus graphs (news before each trade, strategy gaps), recorded answers from
the graphs, and a live headline-by-outcome analysis you can filter, select and search.

Run:  uv run marimo run notebooks/news_gap_lab.py
Data: data/alpaca/trade_news.parquet (scripts/fetch_trade_news.py), data/reports/loop_round1_trades.csv,
docs/trading/collab/infranodus_*.json. The browser (WASM) build reads CSV snapshots from public/news_gap_lab/.
Paper research only.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="News and gap analysis")

with app.setup:
    import io
    import json
    import re
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    alt.theme.enable("dark")
    ROOT = Path(__file__).resolve().parents[1]
    IN_BROWSER = sys.platform == "emscripten"
    FADE, RUN, GREY, ACC = "#5cc2a2", "#ef7b66", "#7d8b86", "#e3a948"
    OUT_SCALE = alt.Scale(domain=["faded", "small loss", "continuation loser"], range=[FADE, GREY, RUN])

    def fetch(local: Path):
        if not IN_BROWSER:
            return local
        import urllib.request
        return io.BytesIO(urllib.request.urlopen(str(mo.notebook_location() / "public" / "news_gap_lab" / local.name)).read())

    def read_csv_or_parquet(local: Path) -> pl.DataFrame:
        if local.suffix == ".parquet" and not IN_BROWSER:
            return pl.read_parquet(local)
        return pl.read_csv(fetch(local.with_suffix(".csv")), try_parse_dates=True)

    def read_json(local: Path):
        return json.loads(fetch(local).read().decode() if IN_BROWSER else local.read_text())

    CATEGORIES = {
        "earnings / results": r"(?i)results|earnings|reports (first|second|third|fourth|q\d)|quarter",
        "guidance / outlook": r"(?i)guidance|outlook",
        "analyst target / rating": r"(?i)price target|upgrade|downgrade|initiates coverage|maintains .*rating|rating",
        "insider / holder selling": r"(?i)insider|sells|share sale|form 144|stock sale",
        "financing / offering": r"(?i)offering|senior notes|convertible|bond sale|financing|private placement",
        "deal / partnership / contract": r"(?i)acquire|acquisition|partnership|partners|agreement|contract|order|deal",
        "legal / investigation": r"(?i)lawsuit|class action|investigat|fraud",
        "index / ETF": r"(?i)index|etf|nasdaq-100|russell",
    }

    def headline_category(h: str) -> str:
        for k, rx in CATEGORIES.items():
            if re.search(rx, h or ""):
                return k
        return "other"


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'news_gap_lab'
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
    # News and gap analysis

    Two InfraNodus text networks built from our own material, plus a live analysis of the actual headlines.

    1. **News and semantic graph.** 322 Alpaca headlines published in the 3 days before each short we took since
       29 Apr 2026, each tagged with what the trade did: *faded*, *small loss* or *continuation loser*. The tags are
       nodes, so you can see which words sit next to the losers.
    2. **Strategy gap graph.** 46 statements from the Nov 24 call, Chris's Nov 14 brief, the email thread, our
       results and the idea board. Gaps between clusters are where new tests come from.

    Each graph is embedded full size below. Click a node to see its statements; the graph's own AI panel answers
    questions about it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack([mo.md(r"""
## How InfraNodus works

InfraNodus turns text into a **network of concepts** and then reads the shape of that network.

1. **Words become nodes.** Each statement is cleaned (stop words removed, words lemmatised), and every remaining
   word is a node. Our outcome tags (*faded*, *small loss*, *continuation loser*) are added as nodes too.
2. **Co-occurrence becomes edges.** Two words that appear close together in a statement are linked; the more often,
   the heavier the link. "price ↔ target" is the heaviest edge in the news graph because analysts publish so many
   target changes.
3. **Betweenness centrality finds the hinges.** A word that sits on many shortest paths between other words connects
   topics. In the news graph *price*, *result* and *report* are the hinges; in the gap graph it is *percent*, *minute*
   and *open*.
4. **Community detection finds the topics.** Groups of words that link to each other more than to the rest form
   clusters (Louvain modularity). The news graph splits into AI/data-center deals, analyst price targets, earnings
   results and insider selling.
5. **Structural gaps are the ideas.** Two big clusters with few links between them are topics people discuss
   separately but never connect. Bridging a gap means asking how the two relate, and that question is usually a new
   hypothesis. Example: *earnings results* and *analyst price targets* sit in separate clusters, yet both precede our
   shorts; does a results headline that is followed by target raises behave differently?

**Why this helps a trading group.** It is a disciplined way to brainstorm: instead of asking "what else could work?",
it points at the specific pair of topics we have not connected yet. Every gap it finds still gets tested on the data
before it counts.
"""), mo.mermaid("""
%%{init: {'theme': 'dark', 'themeVariables': {'fontSize': '20px'}}}%%
flowchart TB
  T[Statements: headlines, call notes, results] --> N[Words and tags become nodes]
  N --> E[Words close together become weighted edges]
  E --> B[Betweenness: which words bridge topics]
  E --> C[Communities: topic clusters]
  C --> G[Structural gaps: big clusters with few links]
  G --> H[Bridge question = new hypothesis]
  H --> X[Test it on our trades in section 4]
""")], gap=1)
    return


@app.cell(hide_code=True)
def _():
    mini_text = mo.ui.text_area(
        value="Twist Bioscience reports fiscal third quarter results and raises guidance\n"
              "Morgan Stanley raises price target on Western Digital\n"
              "Bloom Energy reports record results and raises full-year guidance\n"
              "Barclays maintains buy rating and raises price target on Amphenol",
        label="Try it: paste a few headlines, one per line", full_width=True, rows=5)
    mini_text
    return (mini_text,)


@app.cell(hide_code=True)
def _(mini_text):
    _stop = set("a an the and or of on in to for with by at from as is are was were be its it this that than into over after before up down new".split())
    _lines = [l for l in mini_text.value.splitlines() if l.strip()]
    _toks = [[w for w in re.findall(r"[a-z][a-z0-9-]+", l.lower()) if w not in _stop] for l in _lines]
    _edges = {}
    for _ws in _toks:
        for _i, _a in enumerate(_ws):
            for _b in _ws[_i + 1:_i + 4]:
                if _a != _b:
                    _k = tuple(sorted((_a, _b)))
                    _edges[_k] = _edges.get(_k, 0) + 1
    _E = pl.DataFrame([{"word_a": a, "word_b": b, "weight": w} for (a, b), w in _edges.items()]) if _edges else pl.DataFrame({"word_a": [], "word_b": [], "weight": []})
    _deg = (pl.concat([_E.select(pl.col("word_a").alias("word"), "weight"), _E.select(pl.col("word_b").alias("word"), "weight")])
            .group_by("word").agg(pl.col("weight").sum().alias("weighted_degree"), pl.len().alias("links")).sort("weighted_degree", descending=True)) if _E.height else _E
    mo.hstack([
        mo.vstack([mo.md("**Edges (words within 3 positions of each other)**"), mo.ui.table(_E.sort("weight", descending=True) if _E.height else _E, page_size=8)]),
        mo.vstack([mo.md("**Most connected words** (InfraNodus would use betweenness here)"), mo.ui.table(_deg, page_size=8)]),
    ], widths="equal")
    return


@app.cell(hide_code=True)
def _():
    GRAPHS = read_json(ROOT / "docs/trading/collab/infranodus_graphs_2026-11-24.json")["graphs"]
    ANSWERS = read_json(ROOT / "docs/trading/collab/infranodus_answers_2026-11-24.json")
    return ANSWERS, GRAPHS


@app.cell(hide_code=True)
def _(GRAPHS):
    def _frame(g):
        return mo.vstack([
            mo.md(f"## {'1. News and semantic graph' if g['name'].endswith('semantic') else '2. Strategy gap graph'}\n"
                  f"{g['what']} · [open full screen]({g['public_url']})\n\n"
                  "**Gaps InfraNodus found:**\n" + "\n".join(f"- {x}" for x in g["gaps"])),
            mo.Html(f'<iframe src="{g["public_url"]}" style="width:100%;height:92vh;min-height:880px;display:block;'
                    'border:1px solid #2a3631;border-radius:8px;background:#0f1513" loading="lazy" allowfullscreen></iframe>'),
        ], gap=0.8)
    mo.vstack([_frame(GRAPHS["news"]), _frame(GRAPHS["gaps"])], gap=2.5)
    return


@app.cell(hide_code=True)
def _(ANSWERS):
    question = mo.ui.dropdown({f"[{a['graph']} graph] {a['question']}": i for i, a in enumerate(ANSWERS)},
                              value=f"[{ANSWERS[0]['graph']} graph] {ANSWERS[0]['question']}", label="Question", full_width=True)
    mo.vstack([mo.md("## 3. Ask the graphs\nAnswers generated by InfraNodus from each graph (GraphRAG over the nodes and "
                     "statements). To ask something new, use the AI panel inside the graph above, or ask Claude to add "
                     "it here. Check each answer against the data in section 4 before trusting it."), question])
    return (question,)


@app.cell(hide_code=True)
def _(ANSWERS, question):
    _a = ANSWERS[question.value]
    mo.callout(mo.md(f"**{_a['question']}**\n\n{_a['answer']}\n\n<small>{_a['model']} · {_a['asked']}</small>"), kind="info")
    return


@app.cell(hide_code=True)
def _():
    NEWS = read_csv_or_parquet(ROOT / "data/alpaca/trade_news.parquet").with_columns(
        category=pl.col("headline").map_elements(headline_category, return_dtype=pl.String))
    TRADES = pl.read_csv(fetch(ROOT / "data/reports/loop_round1_trades.csv"), try_parse_dates=True).filter(
        pl.col("trade_date") >= pl.date(2026, 4, 29)).with_columns(
        outcome=pl.when(pl.col("net_bps") <= -200).then(pl.lit("continuation loser"))
        .when(pl.col("net_bps") < 0).then(pl.lit("small loss")).otherwise(pl.lit("faded")))
    _cats = NEWS.group_by("symbol", "trade_date").agg(pl.col("category").unique().sort().str.join(", ").alias("news_types"),
                                                       pl.len().alias("headlines"))
    TRADE_NEWS = TRADES.join(_cats, on=["symbol", "trade_date"], how="left").with_columns(
        pl.col("news_types").fill_null("no news found"), pl.col("headlines").fill_null(0))
    return NEWS, TRADES, TRADE_NEWS


@app.cell(hide_code=True)
def _():
    outcome_pick = mo.ui.multiselect(["faded", "small loss", "continuation loser"],
                                     value=["faded", "small loss", "continuation loser"], label="Outcomes")
    search = mo.ui.text(placeholder="e.g. guidance, price target, insider", label="Search headlines", full_width=True)
    mo.vstack([mo.md("""## 4. What the headlines actually say about the outcome

Each short since 29 Apr 2026, grouped by the kind of news it had before the open. Click a bar (shift-click for
several) to pull those headlines into the table; filter by outcome or search the text. Everything recomputes from
the data."""), mo.hstack([outcome_pick, search], widths=[1, 2])])
    return outcome_pick, search


@app.cell(hide_code=True)
def _(NEWS, TRADES):
    _per_trade = NEWS.unique(subset=["symbol", "trade_date", "category"]).select("symbol", "trade_date", "category", "net_bps")
    _base = TRADES.with_columns(category=pl.lit("all shorts (baseline)")).select("symbol", "trade_date", "category", "net_bps")
    _no = TRADES.join(NEWS.select("symbol", "trade_date").unique(), on=["symbol", "trade_date"], how="anti").with_columns(
        category=pl.lit("no news found")).select("symbol", "trade_date", "category", "net_bps")
    CAT_STATS = pl.concat([_per_trade, _no, _base]).group_by("category").agg(
        trades=pl.len(),
        loser_rate=((pl.col("net_bps") <= -200).mean() * 100).round(1),
        win_rate=((pl.col("net_bps") > 0).mean() * 100).round(1),
        avg_bps=pl.col("net_bps").mean().round(1),
    ).sort("loser_rate", descending=True)
    _base_rate = CAT_STATS.filter(pl.col("category") == "all shorts (baseline)")["loser_rate"][0]
    cat_chart = mo.ui.altair_chart(
        alt.Chart(CAT_STATS).mark_bar().encode(
            y=alt.Y("category:N", sort="-x", title=None),
            x=alt.X("loser_rate:Q", title="% of shorts that became continuation losers (≤ −200 bps)"),
            color=alt.condition(f"datum.loser_rate > {_base_rate}", alt.value(RUN), alt.value(FADE)),
            tooltip=["category", "trades", "loser_rate", "win_rate", "avg_bps"],
        ).properties(height=340, width="container",
                     title=f"Red = worse than the {_base_rate:.1f}% baseline for all shorts in this window"),
        chart_selection="point",
    )
    mo.vstack([cat_chart, mo.ui.table(CAT_STATS, selection=None, label="Same numbers as a table")])
    return (cat_chart,)


@app.cell(hide_code=True)
def _(NEWS, cat_chart, outcome_pick, search):
    _sel = cat_chart.value
    _cats = [] if _sel is None or len(_sel) == 0 else list(pl.DataFrame(_sel)["category"].unique())
    _df = NEWS.filter(pl.col("outcome").is_in(outcome_pick.value))
    if _cats and not set(_cats) & {"all shorts (baseline)"}:
        _df = _df.filter(pl.col("category").is_in(_cats))
    if search.value.strip():
        _df = _df.filter(pl.col("headline").str.contains("(?i)" + re.escape(search.value.strip())))
    _df = _df.select("trade_date", "symbol", "outcome", pl.col("net_bps").round(0), "category", "headline", "source", "url").sort("trade_date", descending=True)
    _tr = _df.unique(subset=["symbol", "trade_date"])
    _msg = (f"**{_df.height} headlines across {_tr.height} shorts** · continuation losers "
            f"{(_tr['net_bps'] <= -200).mean() * 100:.0f}% · win rate {(_tr['net_bps'] > 0).mean() * 100:.0f}% · "
            f"avg {_tr['net_bps'].mean():+.0f} bps" if _tr.height else "**No headlines match.**")
    mo.vstack([mo.md(_msg + (f"  \nSelected: {', '.join(_cats)}" if _cats else "")), mo.ui.table(_df, page_size=12)])
    return


@app.cell(hide_code=True)
def _(TRADE_NEWS):
    trade_scatter = mo.ui.altair_chart(
        alt.Chart(TRADE_NEWS).mark_circle(size=70, opacity=0.85).encode(
            x=alt.X("entry_ts:T", title=None), y=alt.Y("net_bps:Q", title="short P&L (bps)"),
            color=alt.Color("outcome:N", scale=OUT_SCALE, legend=alt.Legend(orient="top", title=None)),
            size=alt.Size("headlines:Q", title="headlines", scale=alt.Scale(range=[40, 260])),
            tooltip=["symbol", "trade_date", alt.Tooltip("net_bps:Q", format=".0f"), "news_types", "headlines"],
        ).properties(height=380, width="container"),
        chart_selection="interval",
    )
    mo.vstack([mo.md("## 5. Trade by trade\nEvery short since 29 Apr. Bigger dots had more headlines. Drag a box "
                     "(for example around the red dots at the bottom) to list those trades and their news."), trade_scatter])
    return (trade_scatter,)


@app.cell(hide_code=True)
def _(NEWS, trade_scatter):
    _sel = trade_scatter.value
    if _sel is None or len(_sel) == 0:
        _out = mo.md("_Drag a box on the chart to list trades._")
    else:
        _sel = pl.DataFrame(_sel).select("symbol", "trade_date", "net_bps", "outcome", "news_types")
        _heads = _sel.join(NEWS.select("symbol", "trade_date", "headline"), on=["symbol", "trade_date"], how="left")
        _out = mo.vstack([
            mo.md(f"**{_sel.height} shorts selected** · win rate {(_sel['net_bps'] > 0).mean() * 100:.0f}% · "
                  f"avg {_sel['net_bps'].mean():+.0f} bps · {int((_sel['net_bps'] <= -200).sum())} continuation losers"),
            mo.ui.table(_heads.sort("net_bps"), page_size=12),
        ])
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. What this suggests testing

    - **Earnings and guidance headlines** carry the highest continuation-loser rates in this window, which matches
      Chris's finding that strong positive earnings shocks are the dangerous subset. Candidate veto: skip shorts with a
      same-day results or guidance headline, then check it on the tune year and the holdout separately.
    - **Analyst target headlines** look no different from the baseline here, despite what the graph's answer suggests.
      That is why every graph answer gets checked against the table above.
    - **No-news spikes** are the cleanest test of the short-covering thesis. Their row shows how they behave so far.
    - Only seven months of news (Alpaca's archive for these names starts around Apr 2026), so treat every rate here as
      a lead, not a rule. Massive or EDGAR history would extend it.
    """)
    return


if __name__ == "__main__":
    app.run()
