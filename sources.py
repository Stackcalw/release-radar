"""Fetch and parse every signal source. Each fetcher returns a small JSON-able payload.

Only public feeds and APIs meant for scripts; no keys, no logins, a plain honest User-Agent.
"""
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from model import ALIAS, versions_in, vs

UA = "release-radar/1.0 (personal forecasting script)"
DOCS = "https://platform.claude.com/docs/en/"
RAW = "https://raw.githubusercontent.com/"
TC = "https://www.testingcatalog.com/"
LINE_WORDS = ("opus", "sonnet", "haiku", "fable", "mythos")
RIVAL_FAMILIES = {"openai": r"^(gpt-|o)(\d+(?:\.\d+)?)", "google": r"^(gemini-)(\d+(?:\.\d+)?)", "xai": r"^(grok-)(\d+(?:\.\d+)?)"}


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def id_list(text):
    return sorted([line, vs(v)] for line, v in versions_in(text))


def ids(url):
    """Fetcher: every Claude model version a page names."""
    return lambda: id_list(get(url))


def litellm():
    keys = json.loads(get(RAW + "BerriAI/litellm/main/model_prices_and_context_window.json"))
    return id_list(" ".join(k for k in keys if "claude" in k))


def openrouter():
    out = []
    for m in json.loads(get("https://openrouter.ai/api/v1/models"))["data"]:
        if m["id"].startswith("anthropic/"):
            day = dt.datetime.fromtimestamp(m["created"], dt.timezone.utc).date().isoformat()
            out += [[line, vs(v), day] for line, v in versions_in(m["id"])]
    return sorted(out)


def modelsdev():
    """Anthropic ids with their listed release date, plus rival 'new version number' dates."""
    j = json.loads(get("https://models.dev/api.json"))
    ant = sorted([line, vs(v), m.get("release_date") or ""] for mid, m in j["anthropic"]["models"].items() for line, v in versions_in(mid))
    first = {}
    for prov, pat in RIVAL_FAMILIES.items():
        for mid, m in j.get(prov, {}).get("models", {}).items():
            hit, day = re.match(pat, mid), m.get("release_date")
            if hit and day:
                key = (prov,) + hit.groups()
                first[key] = min(first.get(key, day), day)
    return {"anthropic": ant, "rivals": sorted(set(first.values()))}


def relnotes():
    """Official API release notes: every dated entry, and the 'We've launched <model>' lines."""
    notes = get(DOCS + "release-notes/overview.md")
    dates, launches, day = [], [], None
    for line in notes.splitlines():
        if line.startswith("### "):
            try:
                day = dt.datetime.strptime(line[4:].strip(), "%B %d, %Y").date().isoformat()
                dates.append(day)
            except ValueError:
                day = None
        elif day and re.search(r"We.ve \[?launched|We announced", line):
            launches += [[ln, vs(v), day] for ln, v in versions_in(line[:200])]
    return {"dates": sorted(set(dates)), "launches": sorted(launches, key=lambda x: x[2]), "ids": id_list(notes)}


def deprecations():
    """Models the official lifecycle table marks Active."""
    out = []
    for row in get(DOCS + "about-claude/model-deprecations.md").splitlines():
        cells = [c.strip() for c in row.split("|")]
        if len(cells) > 3 and cells[2] == "Active":
            out += id_list(cells[1])
    return sorted(out)


def npm():
    times = json.loads(get("https://registry.npmjs.org/@anthropic-ai/claude-code"))["time"]
    return sorted(v[:10] for k, v in times.items() if k not in ("created", "modified"))


def rss(url):
    for it in ET.fromstring(get(url)).iter("item"):
        yield parsedate_to_datetime(it.findtext("pubDate")).date().isoformat(), (it.findtext("title") or "").strip(), (it.findtext("link") or "").strip()


def tc():
    """TestingCatalog: a blog that reports unreleased AI features, often days before launch."""
    out = []
    for day, title, link in rss(TC + "rss/"):
        if re.search("anthropic|claude", (title + link).lower()):
            out.append(["tc", day, title + " " + link.rstrip("/").rsplit("/", 1)[-1], link])
    return out


def tc_history():
    """One-off backfill: every post whose slug names a Claude line, with its true publish date.

    The sitemap's lastmod moves when a post is edited, so each post page is read for its publish date.
    """
    out = []
    for slug in re.findall(r"<loc>%s([a-z0-9-]+)/</loc>" % re.escape(TC), get(TC + "sitemap-posts.xml")):
        if re.search("|".join(LINE_WORDS), slug):
            try:
                found = re.search(r'article:published_time" content="(\d{4}-\d\d-\d\d)', get(TC + slug + "/"))
            except urllib.error.URLError:
                found = None
            if found:
                out.append(["tc", found.group(1), slug.replace("-", " ") + " " + slug, TC + slug + "/"])
            time.sleep(0.3)
    return out


def hn(since_days=30, per_page=200):
    """Hacker News stories whose title names a Claude line (exact word, title only)."""
    since = int(time.time()) - since_days * 86400
    out = {}
    for word in LINE_WORDS:
        url = ("https://hn.algolia.com/api/v1/search_by_date?query=%s&tags=story&restrictSearchableAttributes=title"
               "&typoTolerance=false&queryType=prefixNone&hitsPerPage=%d&numericFilters=created_at_i>%d" % (word, per_page, since))
        for h in json.loads(get(url))["hits"]:
            out[h["objectID"]] = ["hn", h["created_at"][:10], h.get("title") or "", h["objectID"]]
    return list(out.values())


def gnews():
    out = []
    for word in LINE_WORDS:
        url = "https://news.google.com/rss/search?q=%s+when:14d&hl=en-US&gl=US&ceid=US:en" % urllib.parse.quote("claude " + word)
        out += [["gn", day, title, title] for day, title, _ in rss(url)]
    return out


def polymarket():
    """Open 'Next Claude <line> (X+) released by <date>' markets: [line, threshold, date, price, event volume]."""
    out = []
    j = json.loads(get("https://gamma-api.polymarket.com/public-search?q=claude&events_status=active&limit_per_type=50"))
    for ev in j.get("events") or []:
        m = re.match(r"Next Claude (\w+) \(([\d.]+)\+\) released by", ev.get("title") or "")
        if not m:
            continue
        line = ALIAS.get(m.group(1).lower(), m.group(1).lower())
        for mk in ev.get("markets") or []:
            if mk.get("closed") or not mk.get("endDate"):
                continue
            # endDate is end-of-day US time expressed in UTC, so step back to land on the named day
            end = dt.datetime.strptime(mk["endDate"][:19], "%Y-%m-%dT%H:%M:%S") - dt.timedelta(hours=12)
            out.append([line, m.group(2), end.date().isoformat(), float(json.loads(mk["outcomePrices"])[0]), float(ev.get("volume") or 0)])
    return sorted(out)


SOURCES = {
    "openrouter": openrouter,
    "modelsdev": modelsdev,
    "relnotes": relnotes,
    "deprecations": deprecations,
    "docs_llms": ids("https://platform.claude.com/llms.txt"),
    "docs_overview": ids(DOCS + "about-claude/models/overview.md"),
    "sitemap": ids("https://www.anthropic.com/sitemap.xml"),
    "sdk_py": ids(RAW + "anthropics/anthropic-sdk-python/main/src/anthropic/types/model_param.py"),
    "sdk_ts": ids(RAW + "anthropics/anthropic-sdk-typescript/main/src/resources/messages/messages.ts"),
    "litellm": litellm,
    "cc_changelog": ids(RAW + "anthropics/claude-code/main/CHANGELOG.md"),
    "aws_news": ids("https://aws.amazon.com/about-aws/whats-new/recent/feed/"),
    "bedrock_cards": ids("https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html"),
    "npm": npm,
    "tc": tc,
    "hn": hn,
    "gnews": gnews,
    "polymarket": polymarket,
}


def fetch_all(cache, today, max_age=3):
    """Run every fetcher. `today` is an ISO date.

    A failure falls back to the last good payload while it is at most `max_age` days old, then to
    nothing: a feed that stays blocked must not freeze its signal (an old market price, say) forever.
    """
    with cf.ThreadPoolExecutor(8) as pool:
        futures = {name: pool.submit(fn) for name, fn in SOURCES.items()}
    stamps = cache.setdefault("_fetched", {})
    data, status = {}, {}
    for name, fut in futures.items():
        try:
            data[name] = cache[name] = fut.result()
            stamps[name], status[name] = today, "ok"
        except Exception as e:  # noqa: BLE001 - any one feed failing must not stop the run
            age = (dt.date.fromisoformat(today) - dt.date.fromisoformat(stamps[name])).days if name in stamps else None
            usable = age is not None and age <= max_age
            data[name] = cache.get(name) if usable else None
            status[name] = ("stale %dd: " % age if usable else "failed: ") + str(e)[:90]
    return data, status
