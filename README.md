# Release Radar

A daily forecast of the next Claude model launch, per model line (Opus, Sonnet, Haiku, Fable / Mythos).

Live page: https://stackcalw.github.io/release-radar/

This is an independent project and is not affiliated with Anthropic.

## How it works

The model has one row per model line per day since March 2024. A ridge logistic regression gives each day a launch probability from the line's own clock (days since its last release against its usual gap) and a set of signals: leak-blog posts, Hacker News and news headlines that name an unreleased version, rival launches, the weekday, and tripwires on official feeds. Chaining the days gives each curve. Polymarket "released by" prices are blended in at up to half weight. The page lists all 20 signals with their weights.

There are only 22 launches to learn from, so treat the numbers as rough.

## How it updates

A GitHub Action (`.github/workflows/daily.yml`) runs `radar.py run` every day at 07:15 UTC, commits the day's data to `data/` and publishes `site/index.html` to GitHub Pages. To run it by hand, open the Actions tab and choose "Run workflow".

## Run it yourself

Python 3.9 or later, no packages.

| Command | What it does |
|---|---|
| `python3 radar.py run` | Fetch all 18 feeds, refit, forecast, write `site/index.html`. |
| `python3 radar.py test` | Self-check. No network. |
| `python3 radar.py backtest` | Walk-forward test against a constant-rate forecast. No network. |
| `python3 radar.py backfill` | One-off pull of leak-blog and Hacker News history. |

The hosted copy owns `data/`. Pull before running locally, or the two histories diverge.

## Files

- `model.py` holds the hazard model. Every weight, window and prior is a constant at the top.
- `sources.py` holds the fetchers, one per feed.
- `radar.py` runs the pipeline and holds the 20-row signal table.
- `template.html` is the page; `run` injects the data into it.
- `data/releases.json` is the launch book. New launches are added automatically when OpenRouter, the release notes or the lifecycle table list them.
- `data/mentions.json` holds every leak-blog, Hacker News and news headline seen so far.
- `data/history.json` logs the 30-day odds once a day.

## Limits

- A new family name (as Fable and Mythos were) is invisible until it is added to `NEW` in `model.py` and `LINE_WORDS` in `sources.py`.
- A feed that fails falls back to its last good copy for three days, then its signal switches off. The page footer names any feed that is not fresh.
- LMArena, Metaculus, Reddit and the Vertex catalog cannot be read by a script, so they are not used.
