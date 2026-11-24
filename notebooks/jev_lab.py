# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.2",
#     "polars>=2.1.0",
#     "altair>=6.3.0",
#     "httpx",
# ]
# ///
"""Jev lab: how Jev makes decisions, and where it fits in the spike short, using the recorded scenarios
published on the Jev Trading Lab GitHub Pages site.

Run:  uv run marimo run notebooks/jev_lab.py
Data: https://jeromyjsmith.github.io/jev-trading-lab/data/level-N.json (466 real inputs with recorded live Jev
answers, 2026-11-04, jev-1.14.0). Downloaded once into data/jev_pages/; "Refresh" re-pulls. No Jev calls are made.
Inputs are real but unlabelled: a recorded answer is what Jev said, not a measure of accuracy.
Paper research only.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Jev lab")

with app.setup:
    import json
    from pathlib import Path

    import altair as alt
    import httpx
    import marimo as mo
    import polars as pl

    alt.theme.enable("dark")
    ROOT = Path(__file__).resolve().parents[1]
    CACHE = ROOT / "data" / "jev_pages"
    PAGES = "https://jeromyjsmith.github.io/jev-trading-lab"
    LOCAL_REPO = ROOT.parent / "Trading-workbench" / "jev-trading-lab-main" / "apps/ten-levels/web/public/data"
    LEVEL_NAMES = {
        1: "Smart If (Noul gate)", 2: "Choice classifier", 3: "Weighted score", 4: "Confidence gate",
        5: "Cheap router", 6: "Hook guardrail", 7: "Context judge", 8: "Blind read", 9: "Parallel triage",
        10: "Agent-called decision", 11: "Signal review", 12: "Residual classifier", 13: "News filter and kill switch",
    }
    FADE, RUN, GREY, ACC = "#5cc2a2", "#ef7b66", "#7d8b86", "#e3a948"

    def load_levels(refresh: bool = False) -> tuple[dict[int, dict], str]:
        """Pull level JSON from GitHub Pages into data/jev_pages/, falling back to the local repo copy."""
        CACHE.mkdir(parents=True, exist_ok=True)
        out, source = {}, "cache"
        for n in LEVEL_NAMES:
            f = CACHE / f"level-{n}.json"
            if refresh or not f.exists():
                try:
                    r = httpx.get(f"{PAGES}/data/level-{n}.json", timeout=30, follow_redirects=True)
                    r.raise_for_status()
                    f.write_bytes(r.content)
                    source = "GitHub Pages"
                except Exception:
                    lf = LOCAL_REPO / f"level-{n}.json"
                    if lf.exists():
                        f.write_bytes(lf.read_bytes())
                        source = "local repo copy (Pages unreachable)"
            out[n] = json.loads(f.read_text())
        return out, source

    def short(s, n=140):
        s = "" if s is None else str(s)
        return s if len(s) <= n else s[: n - 1] + "…"


@app.cell(hide_code=True)
def hub_nav():
    import os as _os
    import sys as _sys

    import marimo as _mo
    _ORDER = ['brainstorm_board', 'collab_ideas', 'results_deck', 'news_gap_lab', 'model_profiles', 'jev_lab', 'model_lab', 'spike_short_dashboard', 'continuation_loser_detector', 'trade_replay', 'trading_workbench']
    _PORTS = {n: 2761 + i for i, n in enumerate(_ORDER)}
    _WASM = ['model_profiles', 'news_gap_lab']
    _me = 'jev_lab'
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
    # Jev lab

    **Jev** is a decision model: it never writes prose. You hand it a **state** (the facts) and typed
    **questions** (yes/no, pick one, score 0-1). It returns an answer with a confidence in about a second.
    Ordinary code then turns those answers into an **outcome** with a fixed, versioned policy, so every decision
    can be replayed and audited. That split (fuzzy judgement by Jev, hard rules in code) is what makes it usable
    for Mark's "explainable edge" bar.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.mermaid("""
%%{init: {'theme': 'dark', 'themeVariables': {'fontSize': '22px'}, 'flowchart': {'nodeSpacing': 40, 'rankSpacing': 55, 'useMaxWidth': true}}}%%
flowchart TB
  I[Inputs<br/>headline, filing, bars] --> B[Build<br/>versioned decision spec]
  B --> G{Guards in code<br/>venue, future data, size}
  G -->|fail| RJ[rejected_input<br/>no Jev call]
  G -->|pass| J[[Jev<br/>typed questions]]
  J --> A[Answers + confidence<br/>noul / choice / score]
  A --> P{Policy in code<br/>thresholds, rules}
  P --> O[Outcome<br/>attach, flag, pause, skip...]
  O --> L[(Recorded trace<br/>replayable)]
""")
    return


@app.cell(hide_code=True)
def _():
    refresh = mo.ui.run_button(label="Refresh from GitHub Pages")
    refresh
    return (refresh,)


@app.cell(hide_code=True)
def _(refresh):
    LEVELS, SOURCE = load_levels(refresh=refresh.value)
    rows = []
    for _n, _lv in LEVELS.items():
        for _o in _lv["options"]:
            for _i, _st in enumerate(_o.get("static") or []):
                if _st.get("source") != "recorded":
                    continue
                _ev = {e["event"]: e["data"] for e in _st.get("stream", [])}
                _out = (_ev.get("decision") or {}).get("output") or {}
                _inp = (_ev.get("decision") or {}).get("input") or {}
                _res = (_ev.get("response") or {}).get("result") or {}
                _meta = _res.get("meta") or {}
                _use = _res.get("usage") or {}
                _sets = _o.get("sets") or []
                rows.append({
                    "level": _n, "level_name": LEVEL_NAMES[_n], "key": _o["key"], "option": _o["name"],
                    "catalog_id": _o.get("catalog_id"), "set": _i, "set_id": _sets[_i]["set_id"] if _i < len(_sets) else None,
                    "symbol": _inp.get("symbol"), "excerpt": short(_inp.get(_o.get("visual", {}).get("excerpt") or "headline") or _inp.get("headline") or json.dumps({k: v for k, v in _inp.items() if k not in ("data_cutoff", "symbol")}, default=str)),
                    "pick": _out.get("pick"), "confidence": _out.get("confidence"), "outcome": _out.get("outcome"),
                    "status": _out.get("status"), "rule": _out.get("rule"),
                    "ms": _meta.get("elapsedMs"), "in_tokens": _use.get("input_tokens"), "out_tokens": _use.get("output_tokens"),
                })
    RUNS = pl.DataFrame(rows, infer_schema_length=None)
    mo.md(f"Loaded **{RUNS.height} recorded decisions** across **{RUNS['catalog_id'].n_unique()}** decision types "
          f"and 13 levels from *{SOURCE}* ([Jev Trading Lab]({PAGES}/)).")
    return LEVELS, RUNS


@app.cell(hide_code=True)
def _(RUNS):
    _lat = RUNS.filter(pl.col("ms").is_not_null())
    _a = alt.Chart(_lat).mark_circle(size=50, opacity=0.6, color=FADE).encode(
        x=alt.X("ms:Q", title="Jev response time (ms)"), y=alt.Y("in_tokens:Q", title="tokens in"),
        tooltip=["option", "symbol", "pick", "ms", "in_tokens"]).properties(width="container", height=300, title="Speed vs input size (drag a box)")
    _b = alt.Chart(RUNS.filter(pl.col("confidence").is_not_null())).mark_tick(thickness=2, size=14, opacity=0.6).encode(
        x=alt.X("confidence:Q", title="confidence of the pick", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("level_name:N", sort=list(LEVEL_NAMES.values()), title=None),
        color=alt.condition("datum.confidence >= 0.7", alt.value(FADE), alt.value(ACC)),
        tooltip=["option", "symbol", "pick", "confidence", "outcome"]).properties(width="container", height=300, title="Confidence by level (drag across to select)")
    jev_speed = mo.ui.altair_chart(_a, chart_selection="interval")
    jev_conf = mo.ui.altair_chart(_b, chart_selection="interval")
    mo.vstack([
        mo.md(f"""
## What the recordings show

Median response **{_lat['ms'].median():.0f} ms**, median **{RUNS['in_tokens'].median():.0f}** tokens in and
**{RUNS['out_tokens'].median():.0f}** out. A full pre-open screen of 400 names x 3 questions is ~1,200 calls,
which runs in parallel in well under a minute.
        """),
        mo.hstack([jev_speed, jev_conf], widths="equal"),
    ])
    return jev_conf, jev_speed


@app.cell(hide_code=True)
def _(jev_conf, jev_speed):
    _parts = []
    for _name, _c in (("speed chart", jev_speed), ("confidence chart", jev_conf)):
        _v = _c.value
        if _v is not None and len(_v):
            _d = pl.DataFrame(_v)
            _parts += [mo.md(f"**{_d.height} decisions selected on the {_name}** · " + ", ".join(f"{n} {o}" for o, n in _d.group_by("outcome").len().sort("len", descending=True).head(6).iter_rows())),
                       mo.ui.table(_d.select([c for c in ["level_name", "option", "symbol", "excerpt", "pick", "confidence", "outcome", "ms"] if c in _d.columns]), page_size=8)]
    mo.vstack(_parts) if _parts else mo.md("_Drag on either chart to list those decisions._")
    return


@app.cell(hide_code=True)
def _(RUNS):
    level = mo.ui.dropdown({f"{n}. {nm}": n for n, nm in LEVEL_NAMES.items()}, value="13. News filter and kill switch", label="Level")
    level
    return (level,)


@app.cell(hide_code=True)
def _(RUNS, level):
    _opts = RUNS.filter(pl.col("level") == level.value).select("catalog_id", "option").unique(maintain_order=True)
    option = mo.ui.dropdown({f"{r['catalog_id']} · {r['option']}": r["catalog_id"] for r in _opts.iter_rows(named=True)},
                            value=f"{_opts.row(0, named=True)['catalog_id']} · {_opts.row(0, named=True)['option']}", label="Decision")
    option
    return (option,)


@app.cell(hide_code=True)
def _(LEVELS, RUNS, level, option):
    OPT = next(o for o in LEVELS[level.value]["options"] if o.get("catalog_id") == option.value)
    _vis = OPT.get("visual") or {}
    _qs = _vis.get("questions") or []
    _outs = _vis.get("outcomes") or []

    def _esc(s):
        return short(str(s), 60).replace('"', "'").replace("`", "").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")

    _lines = ["%%{init: {'theme': 'dark', 'themeVariables': {'fontSize': '22px'}, 'flowchart': {'nodeSpacing': 40, 'rankSpacing': 55, 'useMaxWidth': true}}}%%", "flowchart TB", f'  IN["{_esc(", ".join(_vis.get("fed", [])))}"] --> J[[Jev]]']
    for _k, _q in enumerate(_qs):
        _lines.append(f'  J --> Q{_k}["{_esc(_q.get("short", _q.get("id")))}: {_esc(_q.get("text", ""))}"]')
        _lines.append(f"  Q{_k} --> P{{policy}}")
    if not _qs:
        _lines.append("  J --> P{policy}")
    for _k, _o in enumerate(_outs):
        _when = _esc("; ".join(_o.get("when") or [])) or "default"
        _lines.append(f'  P -->|"{_when}"| O{_k}["{_esc(_o["id"])}<br/>{_esc(_o.get("meaning", ""))}"]')
    _sub = RUNS.filter(pl.col("catalog_id") == option.value)
    mo.vstack([
        mo.md(f"## {OPT['name']}  \n{OPT.get('blurb') or ''}  \n" + " · ".join(
            f"*{_lbl}: {OPT.get(_k)}*" for _lbl, _k in [("cost", "cost_class"), ("calls", "jev_calls"), ("runs in", "runs_in"), ("stage", "stage_hint")] if OPT.get(_k))),
        mo.mermaid("\n".join(_lines)),
        mo.md(f"**{_sub.height} recorded scenarios.** Pick one:"),
    ])
    return (OPT,)


@app.cell(hide_code=True)
def _(RUNS, option):
    _sub = RUNS.filter(pl.col("catalog_id") == option.value)
    table = mo.ui.table(_sub.select("set", "symbol", "excerpt", "pick", "confidence", "outcome", "ms"),
                        selection="single", initial_selection=[0], page_size=8)
    table
    return (table,)


@app.cell(hide_code=True)
def _(OPT, table):
    _sel = table.value
    _i = int(_sel["set"][0]) if _sel is not None and _sel.height else 0
    _st = OPT["static"][_i]
    _ev = {e["event"]: e["data"] for e in _st.get("stream", [])}
    _dec = _ev.get("decision") or {}
    _pol = _ev.get("policy") or {}
    _steps = pl.DataFrame(_pol.get("steps") or [], infer_schema_length=None)
    _ans = ((_ev.get("response") or {}).get("result") or {}).get("answers") or {}
    _prov = ((OPT.get("sets") or [{}])[_i] if _i < len(OPT.get("sets") or []) else {}).get("provenance") or []
    _links = " · ".join(f"[{p.get('source')}]({p.get('url')})" for p in _prov if p.get("url"))
    _inp = {k: short(v, 600) for k, v in (_dec.get("input") or {}).items()}
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("**Input (real, unlabelled)**"), mo.json(_inp)]),
            mo.vstack([mo.md("**Jev's raw answers**"), mo.json(_ans),
                       mo.md(f"**Outcome:** `{(_dec.get('output') or {}).get('outcome')}` "
                             f"via rule `{(_dec.get('output') or {}).get('rule')}`  \nSources: {_links or 'n/a'}")]),
        ], widths="equal"),
        mo.md("**Policy steps (deterministic code over Jev's answers)**"),
        mo.ui.table(_steps, selection=None) if _steps.height else mo.md("_no steps recorded_"),
        mo.md(f"<small>{_st.get('label', '')}</small>"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.vstack([mo.md(r"""
## Where Jev fits in the spike short

The short makes money when the opening spike fades and loses when it keeps running. Jev's job is never to
predict price. It answers the judgement questions around the price rule, each one a small, versioned decision
we can test on its own and switch off if it doesn't help. The L-numbers match the levels you can browse above.
    """), mo.mermaid("""
%%{init: {'theme': 'dark', 'themeVariables': {'fontSize': '22px'}, 'flowchart': {'nodeSpacing': 40, 'rankSpacing': 55, 'useMaxWidth': true}}}%%
flowchart TB
  subgraph PRE[Night before / pre-open]
    H[Every headline + 8-K for 400 names] --> L1[L1 is it about this ticker?]
    L1 --> L2[L2 which catalyst category?<br/>Chris's 11-category taxonomy]
    L2 --> L3[L3 catalyst strength score<br/>strong positive shock = red]
    H --> L8[L8 blind read of filings<br/>going concern, auditor change, delay]
  end
  subgraph LIVE[09:30-10:00]
    ARM[Spike +5% then first red bar] --> L4{L4 confidence gate<br/>take / half size / skip}
    L3 --> L4
    L8 --> L4
    L4 -->|take| SH[Short, protected limit]
    SH --> L13[L13 news kill switch<br/>new headline during the 20 min]
    L13 --> CV[Cover at +20 min]
  end
  subgraph POST[After the close]
    CV --> L10[L10 journal review<br/>tag the cause of each loss]
    L10 --> L11[L11 signal review<br/>should the rule change?]
  end
"""), mo.md(f"""
**Concrete starting points from the recorded set:** level 2 *Headline triage* and *Which 8-K item is this?*
(catalyst labels), level 3 *Catalyst quality grade* and *Earnings quality* (red/yellow/green), level 8
*Auditor change in the 8-K* and *going-concern doubt* (Mark's filing-keyword idea), level 13 *Earnings gaps*
(kill switch). The [First-Hour Desk](https://jeromyjsmith.github.io/jev-first-hour-desk/) shows the same idea
end to end on a stand-in breakout strategy.

**To test any of these on the short:** run the decision on the pre-open news for each of the ~575 lab trades,
then compare profit factor and tail losers with and without the veto, on tune and holdout separately.
That needs a live Jev key; nothing here calls Jev.
""")])
    return


if __name__ == "__main__":
    app.run()
