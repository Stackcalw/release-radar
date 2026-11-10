#!/usr/bin/python3
"""Release Radar: a daily forecast of the next Anthropic model launch.

  radar.py run        fetch signals, refit, forecast, write site/index.html  (launchd runs this daily)
  radar.py backfill   one-off: pull TestingCatalog and Hacker News history used to fit the leak weights
  radar.py backtest   walk-forward check against a constant-rate forecast
  radar.py test       self-check
"""
import datetime as dt
import json
import math
import os
import random
import sys

import model as M
import sources as S

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZONS = (7, 14, 30, 60, 90, 180)
CURVE_DAYS = 240
LABEL = {"opus": "Opus", "sonnet": "Sonnet", "haiku": "Haiku", "fable": "Fable / Mythos"}
GROUPS = {  # tripwire groups: which feeds can name a model before it ships
    "docs": ("docs_llms", "docs_overview", "sitemap", "relnotes"),
    "tooling": ("sdk_py", "sdk_ts", "litellm", "modelsdev"),
    "cloud": ("aws_news", "bedrock_cards"),
    "changelog": ("cc_changelog",),
}
# The 20 signals: the 10 first proposed (rechecked) and 10 added. kind: fit | prior | blend | dropped.
SIGNALS = [
    ("clock", "Days since the line's last release", "Anthropic release notes", "fit", ""),
    ("cloud", "Cloud catalog names an unreleased model", "AWS What's New feed, Bedrock model cards", "prior",
     "Tripwire. AWS posts on launch day, so this has never fired early. Vertex's catalog is not machine readable."),
    ("lmarena", "LMArena anonymous codenames", "lmarena.ai", "dropped",
     "The site blocks scripts (HTTP 403). Codename talk still reaches the model through the leak blog and Hacker News."),
    ("evals", "AISI and METR evaluations", "metr.org, aisi.gov.uk", "dropped",
     "They publish on launch day (METR's Opus 5.5 summary is dated 22 Sep 2026, the day it shipped), so there is no lead time."),
    ("policy", "Policy filings and RSP updates", "anthropic.com", "dropped", "No dated history to test against past launches."),
    ("compute", "Compute and funding news", "press", "dropped", "Moves over quarters and says nothing about which week."),
    ("exec", "Executive statements", "press", "dropped", "Cannot be measured apart from general news; covered by the news headline signal."),
    ("rival", "A rival shipped a new version in the last 14 days", "models.dev (OpenAI, Google, xAI)", "fit", ""),
    ("market", "Polymarket 'released by' prices", "Polymarket", "blend",
     "Blended in log-odds. A market gets up to half the weight, less when its volume is under $20k."),
    ("gn", "News headline names an unreleased version", "Google News", "prior",
     "Borrows the Hacker News weight, the closest signal that has a history, and counts only while Hacker News is quiet so the same chatter is not counted twice."),
    ("tc", "Leak blog names an unreleased version (21 days)", "TestingCatalog", "fit", ""),
    ("hn", "Hacker News title names an unreleased version (14 days)", "Hacker News", "fit", ""),
    ("train", "Another Claude line shipped in the last 45 days", "Anthropic release notes", "fit", ""),
    ("offday", "Friday, Saturday or Sunday", "calendar", "fit", ""),
    ("holiday", "20 December to 2 January", "calendar", "fit", ""),
    ("cc", "Claude Code release burst (7 days)", "npm registry", "fit", ""),
    ("tooling", "SDK or model catalog names an unreleased model", "Anthropic SDKs, LiteLLM, models.dev", "prior",
     "Tripwire. These update on launch day, so a sighting before launch would be near conclusive."),
    ("docs", "Anthropic docs or sitemap name an unreleased model", "platform.claude.com, anthropic.com", "prior",
     "Tripwire. Official pages are only staged within days of a launch."),
    ("changelog", "Claude Code changelog names an unreleased model", "Claude Code changelog", "prior",
     "Tripwire. In all 15 public launches from Sonnet 4.5 to Fable 5.2 the changelog named the model on launch day, never earlier."),
    ("notes", "API release-notes burst (14 days)", "Anthropic release notes", "fit", ""),
]


def path(*parts):
    return os.path.join(HERE, *parts)


def load(name, default):
    try:
        with open(path("data", name)) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save(name, obj, indent=None):
    tmp = path("data", name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path("data", name))


def merge_mentions(old, new):
    seen = {(m[0], m[3]) for m in old}
    fresh = [m for m in new if (m[0], m[3]) not in seen and not seen.add((m[0], m[3]))]
    return old + fresh, len(fresh)


def current_versions(releases):
    cur = {}
    for r in releases:
        cur[r["line"]] = max(cur.get(r["line"], (0, 0)), M.vt(r["version"]))
    return cur


def detect_releases(data, releases, today):
    """Add any model the official feeds now list. OpenRouter, the release notes and the lifecycle
    table count as proof of launch; models.dev only supplies a fallback date."""
    cur = current_versions(releases)
    fallback = {(l, v): d for l, v, d in (data.get("modelsdev") or {}).get("anthropic", []) if d}
    seen = {}
    for line, ver, day in data.get("openrouter") or []:
        seen[(line, ver)] = min(day, seen.get((line, ver), day))
    for line, ver, day in (data.get("relnotes") or {}).get("launches", []):
        seen.setdefault((line, ver), day)
    for line, ver in data.get("deprecations") or []:
        seen.setdefault((line, ver), fallback.get((line, ver), today.isoformat()))
    last = {}
    for r in releases:
        last[r["line"]] = max(last.get(r["line"], r["date"]), r["date"])
    added = []
    for (line, ver), day in sorted(seen.items(), key=lambda kv: kv[1]):
        if M.unreleased(M.vt(ver), cur.get(line)) and M.D(day) <= today:
            if day <= last[line]:  # a feed dated it before the line's previous launch: trust today instead
                day = today.isoformat()
            added.append({"date": day, "line": line, "version": ver, "name": "Claude %s %s" % (line.title(), ver), "auto": True})
            cur[line], last[line] = M.vt(ver), day
    return added


def sightings(data, cur, state, today):
    """Unreleased model versions named in feeds that normally only change at launch."""
    first = state.setdefault("sightings", {})
    live = {}
    for group, names in GROUPS.items():
        for name in names:
            payload = data.get(name) or []
            if isinstance(payload, dict):
                payload = payload.get("ids") or payload.get("anthropic") or []
            for item in payload:
                line, ver = item[0], item[1]
                if M.unreleased(M.vt(ver), cur.get(line)):
                    key = "%s|%s|%s" % (group, line, ver)
                    age = (today - M.D(first.setdefault(key, today.isoformat()))).days
                    if age <= 30:  # a name that lingers a month without a launch is not a launch signal
                        live.setdefault(line, {})[group] = (ver, age)
    for key in list(first):
        _, line, ver = key.split("|")
        if not M.unreleased(M.vt(ver), cur.get(line)):
            del first[key]
    return live


def inputs(data, mentions):
    dates = lambda xs: [M.D(x) for x in xs or []]
    return {
        "mentions": [(m[0], M.D(m[1]), m[2]) for m in mentions],
        "rivals": dates((data.get("modelsdev") or {}).get("rivals")),
        "npm": dates(data.get("npm")),
        "notes": dates((data.get("relnotes") or {}).get("dates")),
    }


def make_ctx(releases, inp, asof, strict=False):
    """Model context as of a date. strict=True hides everything dated on or after it (for backtests)."""
    keep = (lambda d: d < asof) if strict else (lambda d: d <= asof)
    return M.build([r for r in releases if keep(M.D(r["date"]))],
                   [m for m in inp["mentions"] if keep(m[1])],
                   [d for d in inp["rivals"] if keep(d)], [d for d in inp["npm"] if keep(d)],
                   [d for d in inp["notes"] if keep(d)], asof)


def pct(x):
    return "%d%%" % round(100 * x)


def ago(days):
    return "today" if days == 0 else "1 day ago" if days == 1 else "%d days ago" % days


def forecast_line(ctx, fitted, line, releases, live, markets, today):
    b = dict(zip(M.FEATS, fitted["beta"]))
    dates = ctx["dates"][line]
    last = dates[-1]
    cur = max(v for _, v in ctx["lines"][line])
    trips = live.get(line, {})
    trip = min(sum(M.TRIP[g] for g in trips), M.TRIP_CAP)
    trip_age = min((age for _, age in trips.values()), default=0)
    hn_dates, gn_dates = ctx["leak_dates"]["hn"][line], ctx["leak_dates"]["gn"][line]

    def gn_on(day):  # news counts only while Hacker News is quiet: both measure the same general chatter
        return M.recent(gn_dates, day, M.WINDOW["gn"], last) and not M.recent(hn_dates, day, M.WINDOW["hn"], last)

    def extra(k):
        day = today + dt.timedelta(k)
        z = trip if trips and k + trip_age <= M.WINDOW["trip"] else 0.0
        return z + (b["hn"] if gn_on(day) else 0.0)

    h_model = M.daily(ctx, fitted["beta"], line, 365, extra)
    if last == today:  # it shipped today, so the next one cannot also be today
        h_model[0] = 0.0
    cum_model = M.cumulative(h_model)
    mk = []
    for l, thr, day, price, vol in markets or []:
        k = (M.D(day) - today).days
        if l == line and M.vt(thr) > cur and 0 <= k < 365:
            mk.append({"date": day, "p": price, "weight": round(M.market_weight(vol), 2), "volume": round(vol), "model": round(cum_model[k], 4), "k": k})
    h = M.blend(h_model, [(m["k"], m["p"], m["weight"]) for m in mk])
    cum = M.cumulative(h)

    x = dict(zip(M.FEATS, M.features(ctx, line, today)))
    why = []
    others = [r for r in releases if r["line"] != line and M.D(r["date"]) < today]
    detail = {
        "train": lambda: "%s shipped %s" % (others[-1]["name"], ago((today - M.D(others[-1]["date"])).days)),
        "rival": lambda: "a rival shipped a new version %s" % ago((today - ctx["rivals"][-1]).days),
        "tc": lambda: "leak blog, %s: %s" % (ctx["leaks"]["tc"][line][-1][0].strftime("%-d %b"), headline(ctx["leaks"]["tc"][line][-1][1])),
        "hn": lambda: "Hacker News, %s: %s" % (ctx["leaks"]["hn"][line][-1][0].strftime("%-d %b"), ctx["leaks"]["hn"][line][-1][1][:90]),
        "cc": lambda: "Claude Code is shipping versions faster than usual",
        "notes": lambda: "the API release notes are busier than usual",
        "offday": lambda: "today is a Friday to Sunday",
        "holiday": lambda: "today is in the holiday freeze",
    }
    for f, text in detail.items():
        if x[f]:
            why.append({"id": f, "mult": round(math.exp(b[f]), 2), "text": text()})
    for g, (ver, age) in trips.items():
        why.append({"id": g, "mult": round(math.exp(M.TRIP[g]), 1), "text": "%s %s named in %s feeds (first seen %s)" % (LABEL.get(line, line), ver, g, ago(age))})
    if gn_on(today):
        why.append({"id": "gn", "mult": round(math.exp(b["hn"]), 2), "text": "news: " + ctx["leaks"]["gn"][line][-1][1][:90]})
    for m in mk:
        why.append({"id": "market", "mult": None, "text": "Polymarket says %s by %s; the model alone says %s" % (pct(m["p"]), M.D(m["date"]).strftime("%-d %b"), pct(m["model"]))})

    def when(q):
        i = M.day_at(cum, q)
        return None if i is None else (today + dt.timedelta(i)).isoformat()

    latest = [r for r in releases if r["line"] == line][-1]
    return {
        "key": line, "label": LABEL.get(line, line.title()), "latest": latest["name"], "last_date": last.isoformat(),
        "days_since": (today - last).days, "usual_gap": round(M.typical_gap(ctx, line, today + M.ONE), 1),
        "p": {str(n): round(cum[n - 1], 4) for n in HORIZONS},
        "p_model": {str(n): round(cum_model[n - 1], 4) for n in HORIZONS},
        "median": when(0.5), "lo": when(0.1), "hi": when(0.9),
        "curve": [round(v, 4) for v in cum[:CURVE_DAYS]], "curve_model": [round(v, 4) for v in cum_model[:CURVE_DAYS]],
        "markets": [{k: v for k, v in m.items() if k != "k"} for m in mk], "why": why,
        "on": {**{f: bool(x[f]) for f in detail}, **{g: g in trips for g in GROUPS}, "gn": gn_on(today), "market": bool(mk), "clock": True},
        "_cum": cum,
    }


def headline(text):
    """Mention text is 'title slug'; show the title part."""
    words = text.split(" ")
    return " ".join(words[:-1])[:90] if len(words) > 1 else text[:90]


def signal_table(fitted, lines):
    b, se = dict(zip(M.FEATS, fitted["beta"])), dict(zip(M.FEATS, fitted["se"]))
    raw = dict(zip(M.FEATS, fitted["raw"]))
    share, hits = dict(zip(M.FEATS, fitted["on_share"])), dict(zip(M.FEATS, fitted["hits"]))
    out = []
    for sid, name, source, kind, note in SIGNALS:
        row = {"id": sid, "name": name, "source": source, "kind": kind, "note": note, "beta": None, "se": None,
               "on": [l["label"] for l in lines if l["on"].get(sid)] if kind != "dropped" else []}
        if kind == "fit":
            row["beta"], row["se"] = round(b[sid], 2), round(se[sid], 2)
            if sid == "clock":
                row["note"] = ("Weight on log(days since the last release / the line's usual gap). Usual gap is the last three gaps, newest counted most: "
                               + ", ".join("%s %dd" % (l["label"].split(" ")[0], round(l["usual_gap"])) for l in lines)
                               + ". The daily rate is also divided by the usual gap, so a line that ships half as often gets half the rate.")
            else:
                row["note"] = "On for %d of %d past launches, and on %d%% of all days." % (hits[sid], fitted["events"], round(100 * share[sid]))
                if raw[sid] < 0 and sid in M.NONNEG:
                    row["note"] += " The fit came out at %+.2f; floored at 0 because a leak cannot make a launch less likely." % raw[sid]
        elif sid in M.TRIP:
            row["beta"] = M.TRIP[sid]
        elif sid == "gn":
            row["beta"] = round(b["hn"], 2)
        out.append(row)
    return out


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def run():
    today = utc_now().date()
    os.makedirs(path("data"), exist_ok=True)
    releases, cache, state = load("releases.json", []), load("cache.json", {}), load("state.json", {})
    data, status = S.fetch_all(cache, today.isoformat())
    save("cache.json", cache)

    added = detect_releases(data, releases, today)
    if added:
        releases = sorted(releases + added, key=lambda r: r["date"])
        save("releases.json", releases, indent=1)

    mentions, fresh = merge_mentions(load("mentions.json", []), (data.get("tc") or []) + (data.get("hn") or []) + (data.get("gnews") or []))
    save("mentions.json", mentions)
    inp = inputs(data, mentions)
    ctx = make_ctx(releases, inp, today)
    fitted = M.fit(ctx)
    live = sightings(data, current_versions(releases), state, today)

    lines = [forecast_line(ctx, fitted, line, releases, live, data.get("polymarket"), today) for line in ctx["lines"]]
    lines.sort(key=lambda l: -l["p"]["30"])
    any_cum = [1 - math.prod(1 - l["_cum"][k] for l in lines) for k in range(365)]
    for l in lines:
        del l["_cum"]

    bt = state.get("backtest")
    settings = repr((M.FEATS, M.HALF_LIFE, M.RIDGE, M.PRIOR, sorted(M.WINDOW.items())))
    # ponytail: weekly is plenty, it barely moves. Recomputed at once if a model setting changes.
    if not bt or bt.get("settings") != settings or (today - M.D(bt["date"])).days >= 7:
        bt = M.backtest(lambda asof: make_ctx(releases, inp, asof, strict=True), dt.date(2025, 7, 1), today) or {}
        bt.update(date=today.isoformat(), settings=settings)
        state["backtest"] = bt
    save("state.json", state, indent=1)

    row = {"date": today.isoformat(), "p30": dict({l["key"]: l["p"]["30"] for l in lines}, any=round(any_cum[29], 4))}
    history = [h for h in load("history.json", []) if h["date"] != row["date"]] + [row]
    save("history.json", history)

    i = M.day_at(any_cum, 0.5)
    payload = {
        "generated": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"), "today": today.isoformat(),
        "lines": lines, "labels": LABEL,
        "any": {"p": {str(n): round(any_cum[n - 1], 4) for n in HORIZONS}, "curve": [round(v, 4) for v in any_cum[:CURVE_DAYS]],
                "median": None if i is None else (today + dt.timedelta(i)).isoformat()},
        "signals": signal_table(fitted, lines),
        "fit": {"rows": fitted["rows"], "events": fitted["events"], "half_life": M.HALF_LIFE, "backtest": bt,
                "base": round(fitted["beta"][0], 2)},
        "releases": [{"date": r["date"], "line": r["line"], "name": r["name"]} for r in releases],
        "history": history, "sources": status, "new_releases": [r["name"] for r in added],
    }
    os.makedirs(path("site"), exist_ok=True)
    with open(path("template.html")) as f:
        # headlines come from outside feeds: escape every "<" so none can close or reopen the script tag
        html = f.read().replace("/*DATA*/null", json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c"))
    with open(path("site", "index.html.tmp"), "w") as f:
        f.write(html)
    os.replace(path("site", "index.html.tmp"), path("site", "index.html"))

    ok = sum(1 for s in status.values() if s == "ok")
    print("%s | %s | any 30d %s | sources %d/%d ok | %d new mentions | new releases: %s" % (
        today, "  ".join("%s 30d %s (model %s)" % (l["key"], pct(l["p"]["30"]), pct(l["p_model"]["30"])) for l in lines),
        pct(any_cum[29]), ok, len(status), fresh, ", ".join(r["name"] for r in added) or "none"))
    for name, s in status.items():
        if s != "ok":
            print("  source %s %s" % (name, s))
    return payload, fitted


def backfill():
    os.makedirs(path("data"), exist_ok=True)
    old = load("mentions.json", [])
    mentions, fresh = merge_mentions(old, S.tc_history() + S.hn(since_days=(dt.date.today() - dt.date(2024, 3, 1)).days, per_page=1000))
    save("mentions.json", mentions)
    by = {}
    for m in mentions:
        by[m[0]] = by.get(m[0], 0) + 1
    print("added %d, total %s, oldest %s" % (fresh, by, min(m[1] for m in mentions)))


def backtest():
    releases = load("releases.json", [])
    inp = inputs(load("cache.json", {}), load("mentions.json", []))
    today = utc_now().date()
    print(M.backtest(lambda asof: make_ctx(releases, inp, asof, strict=True), dt.date(2025, 7, 1), today))


def test():
    D = M.D
    assert M.versions_in("claude-opus-4-20250514") == {("opus", (4, 0))}
    assert M.versions_in("claude-sonnet-4-5-20250929") == {("sonnet", (4, 5))}
    assert M.versions_in("anthropic/claude-3.7-sonnet:thinking") == {("sonnet", (3, 7))}
    assert M.versions_in("claude-3-opus-20240229") == {("opus", (3, 0))}
    assert M.versions_in("Anthropic tests Fable 5.2 and Opus 5.5 ahead of the release") == {("fable", (5, 2)), ("opus", (5, 5))}
    assert M.versions_in("claude-mythos-5-2") == {("fable", (5, 2))}
    assert M.versions_in("Opus 5 2x faster. Sonnet 4.6.") == {("opus", (5, 0)), ("sonnet", (4, 6))}
    assert not M.unreleased((18, 0), (5, 5)) and not M.unreleased((5, 5), (5, 5)) and M.unreleased((6, 0), (5, 5))
    lines = {"opus": [(D("2026-07-24"), (5, 0))], "haiku": [(D("2026-10-07"), (5, 5))], "fable": [(D("2026-04-07"), (0, 0))]}
    assert M.leak_lines("Anthropic tests Opus 5.5 ahead of the release", D("2026-09-21"), lines, "tc") == {"opus"}
    assert M.leak_lines("Opus 5 is great", D("2026-09-21"), lines, "hn") == set()
    assert M.leak_lines("Show HN: a haiku 5-7-5 checker for Claude", D("2026-10-20"), lines, "hn") == set()
    assert M.leak_lines("anthropic started red teaming new mythos models first results", D("2026-06-05"), lines, "tc") == {"fable"}
    assert M.leak_lines("Opus 5.5 leaked", D("2026-07-01"), lines, "hn") == set()  # no Opus release on the book yet at that date

    random.seed(1)
    X, y = [], []
    for _ in range(4000):
        x = float(random.random() < 0.3)
        X.append([1.0, x])
        y.append(float(random.random() < M.sigmoid(-3 + 1.5 * x)))
    beta, se = M.fit_logit(X, y, [1.0] * len(y), ridge=0.01)
    assert abs(beta[0] + 3) < 0.4 and abs(beta[1] - 1.5) < 0.5 and 0 < se[1] < 0.3, (beta, se)

    h = [0.01] * 100
    cum = M.cumulative(h)
    assert all(b >= a for a, b in zip(cum, cum[1:])) and 0 < cum[-1] < 1
    full = M.blend(h, [(29, 0.9, 1.0)])
    assert abs(M.cumulative(full)[29] - 0.9) < 1e-6 and full[50] == h[50]
    assert abs(M.cumulative(M.blend(h, [(29, 0.9, 0.0)]))[29] - cum[29]) < 1e-9
    two = M.cumulative(M.blend(h, [(9, 0.5, 1.0), (29, 0.2, 1.0)]))  # contradictory prices must stay monotone
    assert all(b >= a for a, b in zip(two, two[1:]))
    assert M.day_at(cum, 0.5) == 68 and M.day_at(cum, 0.99) is None

    ctx = M.build([{"date": "2026-01-01", "line": "opus", "version": "5"}, {"date": "2026-03-01", "line": "opus", "version": "5.5"},
                   {"date": "2026-02-01", "line": "sonnet", "version": "5"}],
                  [("tc", D("2026-02-20"), "anthropic prepares opus 5.5"), ("tc", D("2026-03-05"), "anthropic prepares opus 6")], [], [], [], D("2026-03-10"))
    f = dict(zip(M.FEATS, M.features(ctx, "opus", D("2026-02-25"))))
    assert f["tc"] == 1 and f["train"] == 1 and abs(f["clock"] - math.log(56 / 120.0)) < 1e-9  # no gaps yet: default 120
    assert abs(M.typical_gap(ctx, "opus", D("2026-03-10")) - (3 * 59 + 2 * 120 + 120) / 6.0) < 1e-9  # one real gap of 59 days
    assert dict(zip(M.FEATS, M.features(ctx, "opus", D("2026-03-03"))))["tc"] == 0  # the 5.5 leak was used up by the 5.5 launch
    assert dict(zip(M.FEATS, M.features(ctx, "opus", D("2026-03-08"))))["tc"] == 1

    # the paths that only run on the interesting days: a new launch, a tripwire, a market
    book = load("releases.json", [])
    day = D(book[-1]["date"]) + dt.timedelta(20)
    feeds = {"openrouter": [["opus", "5.5", "2026-09-22"], ["opus", "5.6", day.isoformat()], ["opus", "41", "2026-01-01"]],
             "relnotes": {"launches": [["sonnet", "5.6", day.isoformat()]], "ids": [["haiku", "5.6"]], "dates": []},
             "deprecations": [["fable", "5.2"]], "docs_llms": [["haiku", "5.6"], ["haiku", "5.5"]]}
    added = detect_releases(feeds, book, day)
    assert [(r["line"], r["version"]) for r in added] == [("opus", "5.6"), ("sonnet", "5.6")], added
    assert detect_releases(feeds, book, day - M.ONE) == []  # dated tomorrow: not shipped yet
    stale = detect_releases({"relnotes": {"launches": [["opus", "5.6", "2026-04-14"]]}}, book, day)
    assert stale[0]["date"] == day.isoformat()  # a date before the line's last launch is replaced by today
    state = {}
    cur = current_versions(book + added)
    live = sightings(feeds, cur, state, day)
    assert live == {"haiku": {"docs": ("5.6", 0)}}, live
    assert sightings(feeds, dict(cur, haiku=(5, 6)), state, day) == {} and state["sightings"] == {}  # shipped: forgotten
    inp = {"mentions": [], "rivals": [], "npm": [], "notes": []}
    ctx = make_ctx(book + added, inp, day)
    fitted = M.fit(ctx)
    quiet = forecast_line(ctx, fitted, "haiku", book + added, {}, [], day)
    loud = forecast_line(ctx, fitted, "haiku", book + added, live, [["haiku", "5.6", (day + dt.timedelta(30)).isoformat(), 0.9, 50000.0]], day)
    assert loud["p"]["14"] > 5 * quiet["p"]["14"] and loud["p"]["30"] > quiet["p"]["30"] and loud["markets"] and loud["on"]["docs"]
    fresh = forecast_line(ctx, fitted, "opus", book + added, {}, [], day)
    assert fresh["days_since"] == 0 and fresh["curve"][0] == 0.0  # shipped today: the next one is not today
    assert len(signal_table(fitted, [quiet, loud, fresh])) == 20

    # a failing feed uses its last good copy for 3 days, then nothing
    real, S.SOURCES = S.SOURCES, {"good": lambda: [1], "bad": lambda: 1 / 0}
    try:
        cache = {"bad": ["old"], "_fetched": {"bad": "2026-11-08"}}
        got, status = S.fetch_all(cache, "2026-11-10")
        assert got == {"good": [1], "bad": ["old"]} and status["bad"].startswith("stale 2d") and cache["_fetched"]["good"] == "2026-11-10"
        got, status = S.fetch_all(cache, "2026-11-20")
        assert got["bad"] is None and status["bad"].startswith("failed")
    finally:
        S.SOURCES = real
    print("ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "backfill": backfill, "backtest": backtest, "test": test}[cmd]()
