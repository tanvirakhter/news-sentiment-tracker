# 📰 News Sentiment Tracker

A Python project that ingests live RSS feeds from major news outlets and performs sentiment analysis on headlines using VADER.

## 🚀 Features

- Multi-source RSS ingestion
- Sentiment scoring per headline
- Mood Index calculation
- Sentiment aggregation by source
- Time-series sentiment tracking
- Exported CSV + PNG visualizations
- Feed diagnostics logging

## 📊 Example Outputs

- `news_sentiment.csv`
- `sentiment_by_source.png`
- `sentiment_over_time.png`

## 🧠 Mood Index

The Mood Index is the average VADER compound score across all collected headlines.

- > 0.05 → Positive news cycle  
- -0.05 to 0.05 → Neutral  
- < -0.05 → Negative  

## 🛠 Tech Stack

- Python
- pandas
- requests
- feedparser
- vaderSentiment
- matplotlib

## ▶️ How to Run

```bash
git clone https://github.com/YOUR_USERNAME/news-sentiment-tracker.git
cd news-sentiment-tracker

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python main.py

```
## ▶️ How to Run
news-sentiment-tracker/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/

