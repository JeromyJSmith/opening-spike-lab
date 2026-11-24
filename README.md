# Opening Spike Lab

Research notebooks for the opening-spike short: liquid US stocks that gap up 0.25–1.0 ATR, run +5% from the open,
and are shorted on the first red 1-minute bar with a 20-minute cover. **Paper research only. Nothing here places
orders or is investment advice.**

**Hub:** https://jeromyjsmith.github.io/opening-spike-lab/

Each notebook can be opened three ways:
- **Open**: a pre-computed page (`notebooks/<name>.html`).
- **In your browser**: a WebAssembly build that runs the whole notebook in the page (`live/model_profiles/`, `live/news_gap_lab/`).
- **Open in molab**: the notebook source (`notebooks/<name>.py`) in your own marimo account.

| # | Notebook | Open in molab |
|---|---|---|
| 1 | Brainstorm board | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/brainstorm_board.py) |
| 2 | Collaboration board | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/collab_ideas.py) |
| 3 | Results deck | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/results_deck.py) |
| 4 | News and gap analysis | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/news_gap_lab.py) |
| 5 | Model profiles | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/model_profiles.py) |
| 6 | Jev lab | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/jev_lab.py) |
| 7 | Model lab | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/model_lab.py) |
| 8 | Spike Short Lab | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/spike_short_dashboard.py) |
| 9 | Continuation-loser detector | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/continuation_loser_detector.py) |
| 10 | Trade replay | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/trade_replay.py) |
| 11 | Trading workbench (Nov 14) | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/JeromyJSmith/opening-spike-lab/blob/main/notebooks/trading_workbench.py) |

Model profiles and News and gap analysis run from the small CSV/JSON snapshots in `notebooks/public/`. The other
notebooks need the full research workbench (the `spike_lab` package and cached market data, which are not published
here): in molab you can read the code and the saved outputs, but re-running them needs that data.

Built with [marimo](https://marimo.io). Market data: Alpaca SIP 1-minute bars (May 2025 – Nov 2026).
