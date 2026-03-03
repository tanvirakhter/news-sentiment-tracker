from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEEDS = {
    "BBC": "https://feeds.bbci.co.uk/news/rss.xml",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Hacker News": "https://news.ycombinator.com/rss",
    "NYTimes Home": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "WSJ World": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) NewsSentimentTracker/1.0"
}


def parse_feed(name: str, url: str, limit: int = 50) -> tuple[list[dict], dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
        status = r.status_code
        content_type = r.headers.get("content-type", "")
        text = r.text

        d = feedparser.parse(text)

        bozo = int(getattr(d, "bozo", 0) or 0)
        err = str(getattr(d, "bozo_exception", "")) if bozo else ""

        rows: list[dict] = []
        for e in (getattr(d, "entries", []) or [])[:limit]:
            title = getattr(e, "title", None) or getattr(e, "summary", None) or ""
            rows.append(
                {
                    "source": name,
                    "title": str(title).strip(),
                    "link": getattr(e, "link", "") or "",
                    "published": getattr(e, "published", None),
                }
            )

        stats = {
            "feed": name,
            "url": url,
            "http_status": status,
            "content_type": content_type,
            "entries_count": len(rows),
            "bozo": bozo,
            "error": err,
        }
        return rows, stats

    except Exception as e:
        return [], {
            "feed": name,
            "url": url,
            "http_status": None,
            "content_type": None,
            "entries_count": 0,
            "bozo": 1,
            "error": repr(e),
        }


def plot_sentiment_by_source(by_source: pd.DataFrame, outpath: Path) -> None:
    plt.figure()
    plt.bar(by_source["source"], by_source["sent_compound"])
    plt.xticks(rotation=30, ha="right")
    plt.title("Average Headline Sentiment by Source")
    plt.ylabel("VADER Compound Score")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_sentiment_over_time(hourly: pd.DataFrame, outpath: Path) -> None:
    plt.figure()
    plt.plot(hourly["hour"].dt.to_pydatetime(), hourly["sent_compound"])
    plt.title("Average Headline Sentiment Over Time")
    plt.ylabel("VADER Compound Score")
    plt.xlabel("Time (UTC)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="News Sentiment Tracker (RSS + VADER)")
    parser.add_argument("--limit", type=int, default=50, help="Headlines per feed")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running main()...")

    # Collect + diagnostics
    data: list[dict] = []
    stats_rows: list[dict] = []

    for name, url in FEEDS.items():
        rows, stats = parse_feed(name, url, limit=args.limit)
        data.extend(rows)
        stats_rows.append(stats)

    stats_df = pd.DataFrame(stats_rows)
    stats_path = outdir / "feed_diagnostics.csv"
    stats_df.to_csv(stats_path, index=False)

    print("\nFeed diagnostics:")
    cols = ["feed", "http_status", "content_type", "entries_count", "bozo"]
    print(stats_df[cols].to_string(index=False))
    print(f"\nSaved diagnostics: {stats_path}")

    if not data:
        print("\nNo data collected from any feed.")
        print("Open outputs/feed_diagnostics.csv and check http_status/content_type/error.")
        return

    # DataFrame cleanup
    df = pd.DataFrame(data)
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].ne("")].drop_duplicates(subset=["title", "source"]).reset_index(drop=True)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)

    # Sentiment
    analyzer = SentimentIntensityAnalyzer()
    df["sent_compound"] = df["title"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    df["sent_label"] = pd.cut(
        df["sent_compound"],
        bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["negative", "neutral", "positive"],
    )

    mood = float(df["sent_compound"].mean())
    print(f"\nMood Index (avg compound): {mood:.3f}")

    top_pos = df.sort_values("sent_compound", ascending=False).head(10)[
        ["source", "sent_compound", "title", "link"]
    ]
    top_neg = df.sort_values("sent_compound", ascending=True).head(10)[
        ["source", "sent_compound", "title", "link"]
    ]
    by_source = (
        df.groupby("source", as_index=False)["sent_compound"]
        .mean()
        .sort_values("sent_compound", ascending=False)
    )

    print("\nTop Positive Headlines:\n")
    print(top_pos.to_string(index=False))
    print("\nTop Negative Headlines:\n")
    print(top_neg.to_string(index=False))

    # Save plots
    by_source_plot = outdir / "sentiment_by_source.png"
    plot_sentiment_by_source(by_source, by_source_plot)
    print(f"\nSaved chart: {by_source_plot}")

    df_time = df.dropna(subset=["published"]).copy()
    if not df_time.empty:
        df_time["hour"] = df_time["published"].dt.floor("h")
        hourly = df_time.groupby("hour", as_index=False)["sent_compound"].mean().sort_values("hour")
        time_plot = outdir / "sentiment_over_time.png"
        plot_sentiment_over_time(hourly, time_plot)
        print(f"Saved chart: {time_plot}")
    else:
        print("\n(No published timestamps found; skipping time-series chart.)")

    # Export
    csv_path = outdir / "news_sentiment.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} (rows: {len(df)})")


if __name__ == "__main__":
    main()