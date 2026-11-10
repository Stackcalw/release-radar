"""Hazard model for "when does the next Claude model ship". Stdlib only (runs on /usr/bin/python3 3.9).

One row per (model line, day). Outcome: did that line ship that day. A ridge logistic regression
gives a daily launch probability; chaining the days gives the curve. Signals without history
(tripwires, news) are added as fixed log-odds; Polymarket prices are blended in at the end.
"""
import bisect
import datetime as dt
import math
import re

ONE = dt.timedelta(1)
ALIAS = {"mythos": "fable"}  # Mythos ships on the same day as Fable, so they are one line
NEW = re.compile(r"(opus|sonnet|haiku|fable|mythos)-(\d{1,2})(?:[.-](\d{1,2}))?(?![0-9a-z])")
OLD = re.compile(r"claude-(\d{1,2})(?:[.-](\d{1,2}))?-(opus|sonnet|haiku)")
LEAK_VERB = re.compile(r"prepar|red-teaming|about-to|spotted|leak|upcoming|ahead-of|coming|rollout|tests?-|testing")

FEATS = ["base", "clock", "scale", "train", "rival", "offday", "holiday", "tc", "hn", "cc", "notes"]
NONNEG = ("tc", "hn")   # a leak cannot make a launch less likely, so these weights are floored at 0
HALF_LIFE = 180         # days; older history counts less because the cadence keeps speeding up
# Ridge penalty per weight and the prior mean it shrinks toward. The intercept and clock are free.
# "scale" is pinned at -1: a line that ships half as often has half the daily rate (renewal process).
# Every other signal gets a prior sd of 1 log-odds around 0, so 22 launches can't overfit it.
# Half-life and the pin were picked on the walk-forward backtest (skill +11% vs +3% unpinned at 365).
RIDGE = [0.0, 0.01, 1000.0] + [1.0] * 8
PRIOR = [0.0, 0.0, -1.0] + [0.0] * 8
WINDOW = {"train": 45, "rival": 14, "tc": 21, "hn": 14, "gn": 14, "trip": 14}
TRIP = {"docs": 3.0, "tooling": 2.5, "cloud": 2.5, "changelog": 2.5}  # priors: no pre-launch sighting on record to fit
TRIP_CAP = 3.5          # ponytail: tripwires are summed then capped, not modelled jointly
MARKET_MAX = 0.5        # most weight a Polymarket price can get in the blend
MARKET_FULL_VOL = 20000.0  # USD of event volume at which it gets that full weight


def D(s):
    return dt.date.fromisoformat(s[:10])


def vt(s):
    a, _, b = str(s).partition(".")
    return int(a), int(b or 0)


def vs(v):
    return "%d.%d" % v if v[1] else str(v[0])


def norm(text):
    return re.sub(r"[^a-z0-9.]+", "-", text.lower())


def versions_in(text, normalised=False):
    """Every (line, (major, minor)) named in free text, ids or slugs."""
    s = text if normalised else norm(text)
    out = {(ALIAS.get(l, l), (int(a), int(b or 0))) for l, a, b in NEW.findall(s)}
    out |= {(l, (int(a), int(b or 0))) for a, b, l in OLD.findall(s)}
    return out


def unreleased(v, cur):
    """Is version v a plausible next version after cur? (rejects 'Sonnet 18' and date fragments)"""
    return cur is not None and cur < v <= (cur[0] + 1, 99)


def leak_lines(text, on, lines, src):
    """Lines for which this headline is a pre-launch mention on date `on`."""
    s = norm(text)
    if "haiku-5-7-5" in s:  # the poem, not the model
        return set()
    named = "claude" in s or "anthropic" in s
    out, versioned = set(), set()
    for line, v in versions_in(s, True):
        versioned.add(line)
        cur = max((ver for d, ver in lines.get(line, ()) if d <= on), default=None)
        if unreleased(v, cur) and (named or line in ("opus", "fable")):
            out.add(line)
    if src == "tc" and "model" in s and LEAK_VERB.search(s):  # "red-teaming new mythos models"
        for word in re.findall(r"opus|sonnet|haiku|fable|mythos", s):
            line = ALIAS.get(word, word)
            if line not in versioned and any(d <= on for d, _ in lines.get(line, ())):
                out.add(line)
    return out


def build(releases, mentions, rivals, npm, notes, today):
    """Everything the model needs, using nothing dated after `today`.

    releases: [{date, line, version}], mentions: [(src, date, text)], the rest: lists of dates.
    """
    lines = {}
    for r in sorted(releases, key=lambda r: r["date"]):
        if D(r["date"]) <= today:
            lines.setdefault(r["line"], []).append((D(r["date"]), vt(r["version"])))
    leaks = {src: {line: [] for line in lines} for src in ("tc", "hn", "gn")}
    for src, date, text in mentions:
        if date <= today and src in leaks:
            for line in leak_lines(text, date, lines, src):
                leaks[src][line].append((date, text))
    for src in leaks:
        for line in leaks[src]:
            leaks[src][line].sort()
    return {
        "today": today,
        "lines": lines,
        "dates": {l: [d for d, _ in h] for l, h in lines.items()},
        "others": {l: sorted(d for m, h in lines.items() if m != l for d, _ in h) for l in lines},
        "rivals": sorted(d for d in rivals if d <= today),
        "npm": sorted(d for d in npm if d <= today),
        "notes": sorted(d for d in notes if d <= today),
        "leaks": leaks,
        "leak_dates": {src: {l: [d for d, _ in v] for l, v in by.items()} for src, by in leaks.items()},
    }


def count_in(dates, lo, hi):
    return bisect.bisect_right(dates, hi) - bisect.bisect_left(dates, lo)


def last_release(ctx, line, day):
    dates = ctx["dates"][line]
    return dates[bisect.bisect_left(dates, day) - 1]


def recent(dates, day, window, after=None):
    """Any date in the `window` days before `day` (and after `after`, if given)?"""
    lo = day - dt.timedelta(window)
    if after is not None and after + ONE > lo:
        lo = after + ONE
    return count_in(dates, lo, day - ONE) > 0


def hot(dates, day, short, long_, floor):
    """Is the last `short` days' count at least `floor` and 1.5x the trailing `long_`-day norm?"""
    n = count_in(dates, day - dt.timedelta(short), day - ONE)
    norm_ = count_in(dates, day - dt.timedelta(short + long_), day - dt.timedelta(short + 1)) * short / float(long_)
    return n >= floor and n >= 1.5 * norm_


def holiday(day):
    return (day.month == 12 and day.day >= 20) or (day.month == 1 and day.day <= 2)


def typical_gap(ctx, line, day):
    """The line's usual days between releases as of `day`: its last three gaps, newest counted most.

    A young line with fewer than three gaps borrows the other lines' latest gap for the missing ones.
    """
    dates = ctx["dates"][line]
    i = bisect.bisect_left(dates, day)
    gaps = [(dates[j] - dates[j - 1]).days for j in range(max(1, i - 3), i)][::-1]
    if len(gaps) < 3:
        borrowed = []
        for other, ds in ctx["dates"].items():
            j = bisect.bisect_left(ds, day)
            if other != line and j >= 2:
                borrowed.append((ds[j - 1] - ds[j - 2]).days)
        gaps += [sum(borrowed) / float(len(borrowed)) if borrowed else 120.0] * (3 - len(gaps))
    return (3 * gaps[0] + 2 * gaps[1] + gaps[2]) / 6.0


def features(ctx, line, day):
    """Covariates for (line, day), built only from information dated before `day`."""
    last = last_release(ctx, line, day)
    gap = typical_gap(ctx, line, day)
    ld = ctx["leak_dates"]
    return [float(x) for x in (
        1,
        math.log(((day - last).days + 1) / gap),  # how far through its usual gap the line is
        math.log(gap / 60.0),                     # slower lines have a lower daily rate
        recent(ctx["others"][line], day, WINDOW["train"]),
        recent(ctx["rivals"], day, WINDOW["rival"]),
        day.weekday() >= 4,
        holiday(day),
        recent(ld["tc"][line], day, WINDOW["tc"], last),
        recent(ld["hn"][line], day, WINDOW["hn"], last),
        hot(ctx["npm"], day, 7, 90, 5),
        hot(ctx["notes"], day, 14, 90, 3),
    )]


def training_rows(ctx):
    """One row per line per day from the day after its first release up to yesterday."""
    X, y, w = [], [], []
    today = ctx["today"]
    for line, dates in ctx["dates"].items():
        shipped = set(dates)
        day = dates[0] + ONE
        while day < today:
            X.append(features(ctx, line, day))
            y.append(1.0 if day in shipped else 0.0)
            w.append(0.5 ** ((today - day).days / float(HALF_LIFE)))
            day += ONE
    return X, y, w


def solve(A, b):
    """Gaussian elimination with partial pivoting. A is small (13x13)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_logit(X, y, w, ridge=None, prior=None, iters=50):
    """Weighted ridge logistic regression by Newton's method. Returns (beta, standard errors)."""
    p = len(X[0])
    ridge = RIDGE if ridge is None else ridge
    if not isinstance(ridge, list):
        ridge = [0.0] + [ridge] * (p - 1)
    prior = prior or [0.0] * p
    beta = [0.0] * p
    beta[0] = logit(sum(a * b for a, b in zip(y, w)) / sum(w))
    H = None
    for _ in range(iters):
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for xi, yi, wi in zip(X, y, w):
            pi = sigmoid(sum(b * x for b, x in zip(beta, xi)))
            r, s = wi * (yi - pi), wi * pi * (1 - pi)
            for j in range(p):
                xj = xi[j]
                if xj:
                    g[j] += r * xj
                    sx, Hj = s * xj, H[j]
                    for k in range(j, p):
                        Hj[k] += sx * xi[k]
        for j in range(p):
            for k in range(j):
                H[j][k] = H[k][j]
        for j in range(p):
            g[j] -= ridge[j] * (beta[j] - prior[j])
            H[j][j] += ridge[j]
        step = solve(H, g)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-9:
            break
    se = [math.sqrt(max(solve(H, [1.0 if i == j else 0.0 for i in range(p)])[j], 0.0)) for j in range(p)]
    return beta, se


def fit(ctx):
    X, y, w = training_rows(ctx)
    raw, se = fit_logit(X, y, w, RIDGE, PRIOR)
    used =[max(b, 0.0) if name in NONNEG else b for name, b in zip(FEATS, raw)]
    events = int(sum(y))
    return {"beta": used, "raw": raw, "se": se, "rows": len(y), "events": events,
            "eff_events": sum(a * b for a, b in zip(y, w)),
            # how often each on/off signal was on, and how many launches it was on for
            "on_share": [sum(1 for x in X if x[j] > 0) / float(len(y)) for j in range(len(FEATS))],
            "hits": [int(sum(yi for x, yi in zip(X, y) if x[j] > 0)) for j in range(len(FEATS))]}


def daily(ctx, beta, line, horizon=365, extra=None):
    """Launch probability for each of the next `horizon` days, today first.

    extra(k) -> added log-odds on day today+k, for signals that have no history to fit.
    """
    out = []
    for k in range(horizon):
        z = sum(b * x for b, x in zip(beta, features(ctx, line, ctx["today"] + dt.timedelta(k))))
        out.append(sigmoid(z + (extra(k) if extra else 0.0)))
    return out


def cumulative(h):
    out, s = [], 1.0
    for x in h:
        s *= 1 - x
        out.append(1 - s)
    return out


def blend(h, targets):
    """Pull the curve toward market prices. targets: [(day index, market P by that day, weight)].

    Blends in log-odds at each market date, then rescales the daily hazards between market
    dates so the curve passes through the blended points. Days after the last market are untouched.
    """
    cum_h, s = [], 0.0
    for x in h:
        s += -math.log(max(1 - x, 1e-12))
        cum_h.append(s)
    out = list(h)
    prev_k, prev_new, prev_old = -1, 0.0, 0.0
    for k, q, wt in sorted(targets):
        if not 0 <= k < len(h) or k <= prev_k:
            continue
        p_model = 1 - math.exp(-cum_h[k])
        p_star = sigmoid((1 - wt) * logit(p_model) + wt * logit(q))
        new = max(-math.log(1 - p_star), prev_new + 1e-9)
        m = (new - prev_new) / max(cum_h[k] - prev_old, 1e-12)
        for j in range(prev_k + 1, k + 1):
            out[j] = 1 - (1 - h[j]) ** m
        prev_k, prev_new, prev_old = k, new, cum_h[k]
    return out


def market_weight(volume):
    return MARKET_MAX * min(1.0, math.sqrt(max(volume, 0.0) / MARKET_FULL_VOL))


def day_at(cum, q):
    """First day index where the cumulative probability reaches q (None if beyond the horizon)."""
    i = bisect.bisect_left(cum, q)
    return i if i < len(cum) else None


def backtest(make_ctx, start, today, horizon=30):
    """Walk forward: on the 1st of each month refit on the past only and forecast 30 days ahead.

    make_ctx(asof) -> ctx built from data dated <= asof. Compares against a constant-rate forecast.
    """
    rows = []
    c = dt.date(start.year, start.month, 1)
    while c + dt.timedelta(horizon) <= today:
        ctx = make_ctx(c)
        X, y, w = training_rows(ctx)
        if sum(y) >= 5:
            beta = fit(ctx)["beta"]
            rate = sum(a * b for a, b in zip(y, w)) / sum(w)
            naive = 1 - (1 - rate) ** horizon
            full = make_ctx(today)
            for line in ctx["lines"]:
                p = cumulative(daily(ctx, beta, line, horizon))[-1]
                hit = count_in(full["dates"][line], c, c + dt.timedelta(horizon - 1)) > 0
                rows.append((p, naive, 1.0 if hit else 0.0))
        c = dt.date(c.year + (c.month == 12), c.month % 12 + 1, 1)
    n = len(rows)
    if not n:
        return None
    return {"n": n, "brier": sum((p - o) ** 2 for p, _, o in rows) / n,
            "brier_naive": sum((q - o) ** 2 for _, q, o in rows) / n,
            "hit_rate": sum(o for _, _, o in rows) / n,
            "mean_p": sum(p for p, _, _ in rows) / n}
