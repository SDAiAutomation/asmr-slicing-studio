"""Veille buzz : récupère les sujets GTA 6 du jour (texte uniquement, jamais de vidéos).

Sources : Google News (FR + EN) et Reddit r/GTA6. Résultat : out/topics_YYYY-MM-DD.json
"""
import datetime as dt
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import config

UA = {"User-Agent": "Mozilla/5.0 (gta-buzz-engine; veille texte)"}


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def google_news(query="GTA 6", lang="fr"):
    region = {"fr": ("fr", "FR", "FR:fr"), "en": ("en-US", "US", "US:en")}[lang]
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(query + " when:1d") +
           f"&hl={region[0]}&gl={region[1]}&ceid={region[2]}")
    root = ET.fromstring(_get(url))
    out = []
    for it in root.iter("item"):
        out.append({
            "source": "google_news_" + lang,
            "title": it.findtext("title", ""),
            "link": it.findtext("link", ""),
            "date": it.findtext("pubDate", ""),
            "score": 0,
        })
    return out


def reddit(sub="GTA6", period="day", limit=25):
    url = f"https://www.reddit.com/r/{sub}/top.json?t={period}&limit={limit}"
    data = json.loads(_get(url))
    out = []
    for c in data["data"]["children"]:
        p = c["data"]
        if p.get("over_18"):
            continue
        out.append({
            "source": "reddit_" + sub,
            "title": p["title"],
            "link": "https://www.reddit.com" + p["permalink"],
            "date": dt.datetime.fromtimestamp(p["created_utc"], dt.timezone.utc).isoformat(),
            "score": p.get("score", 0),
            "flair": p.get("link_flair_text") or "",
        })
    return out


SPOILER_HINTS = ("spoiler", "ending", "leak", "fuite", "final mission", "dernière mission")


def run_veille():
    topics, errors = [], []
    for fn, args in ((google_news, ("GTA 6", "fr")), (google_news, ("GTA 6", "en")), (reddit, ())):
        try:
            topics += fn(*args)
        except Exception as e:  # une source en panne ne bloque pas les autres
            errors.append(f"{fn.__name__}{args}: {e}")
    # filtre spoilers / fuites, dédoublonne
    seen, clean = set(), []
    for t in topics:
        key = t["title"].lower()[:70]
        flair = t.get("flair", "").lower()
        if key in seen or any(h in key or h in flair for h in SPOILER_HINTS):
            continue
        seen.add(key)
        clean.append(t)
    clean.sort(key=lambda t: t["score"], reverse=True)
    config.OUT_DIR.mkdir(exist_ok=True)
    path = config.OUT_DIR / f"topics_{dt.date.today().isoformat()}.json"
    path.write_text(json.dumps({"topics": clean, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, clean, errors
