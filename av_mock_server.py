"""Optional local mock Alpha Vantage server used by `mp3_backend`.

The backend no longer requires this server for normal app usage because it now
falls back to in-process mock handlers. This file is kept so `start_mock_server()`
still has a valid target when you explicitly want the HTTP mock.
"""

from flask import Flask, jsonify, request
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf


app = Flask(__name__)
_info_cache = {}


def _get_info(ticker):
    if ticker not in _info_cache:
        try:
            _info_cache[ticker] = yf.Ticker(ticker).info
        except Exception:
            _info_cache[ticker] = None
    return _info_cache[ticker]


def _is_market_open(tz_name, open_h, open_m, close_h, close_m):
    now = datetime.now(ZoneInfo(tz_name))
    if now.weekday() >= 5:
        return False
    current = now.hour * 60 + now.minute
    return (open_h * 60 + open_m) <= current < (close_h * 60 + close_m)


def _handle_overview(params):
    ticker = params.get("symbol", "")
    if not ticker:
        return {}

    info = _get_info(ticker)
    if info and info.get("shortName"):
        def safe(value):
            return "None" if value is None else str(value)

        pe = info.get("trailingPE") or info.get("forwardPE")
        eps = info.get("trailingEps") or info.get("forwardEps")
        return {
            "Symbol": ticker,
            "Name": info.get("shortName", ticker),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "MarketCapitalization": safe(info.get("marketCap")),
            "PERatio": safe(pe),
            "EPS": safe(eps),
            "52WeekHigh": safe(info.get("fiftyTwoWeekHigh")),
            "52WeekLow": safe(info.get("fiftyTwoWeekLow")),
            "DividendYield": safe(info.get("dividendYield")),
            "Beta": safe(info.get("beta")),
        }
    return {}


def _handle_market_status():
    return {
        "endpoint": "Global Market Open & Close Status",
        "markets": [
            {
                "market_type": "Equity",
                "region": "United States",
                "primary_exchanges": "NYSE, NASDAQ, AMEX, BATS",
                "local_open": "09:30",
                "local_close": "16:15",
                "current_status": "open" if _is_market_open("America/New_York", 9, 30, 16, 15) else "closed",
                "notes": "",
            },
            {
                "market_type": "Equity",
                "region": "United Kingdom",
                "primary_exchanges": "London Stock Exchange",
                "local_open": "08:00",
                "local_close": "16:30",
                "current_status": "open" if _is_market_open("Europe/London", 8, 0, 16, 30) else "closed",
                "notes": "",
            },
            {
                "market_type": "Equity",
                "region": "Japan",
                "primary_exchanges": "Tokyo Stock Exchange",
                "local_open": "09:00",
                "local_close": "15:00",
                "current_status": "open" if _is_market_open("Asia/Tokyo", 9, 0, 15, 0) else "closed",
                "notes": "",
            },
        ],
    }


def _handle_top_gainers_losers():
    tickers = [
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "INTC", "NFLX",
        "CRM", "ORCL", "PYPL", "SQ", "SHOP", "PLTR", "RIVN", "LCID", "NIO", "SOFI",
    ]

    def random_rows():
        rows = []
        for ticker in random.sample(tickers, 5):
            price = round(random.uniform(10, 400), 2)
            change = round(random.uniform(2, 15), 2)
            rows.append(
                {
                    "ticker": ticker,
                    "price": str(price),
                    "change_amount": str(round(price * change / 100, 2)),
                    "change_percentage": f"{change}%",
                    "volume": str(random.randint(1_000_000, 50_000_000)),
                }
            )
        return rows

    return {
        "metadata": "Top Gainers, Losers, and Most Active (random fallback)",
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_gainers": random_rows(),
        "top_losers": random_rows(),
        "most_actively_traded": random_rows(),
    }


def _handle_news_sentiment(params):
    ticker = params.get("tickers", "")
    limit = int(params.get("limit", 5))
    feed = []
    headlines = [
        f"{ticker} Shows Strong Momentum Amid Market Volatility",
        f"Analysts Upgrade {ticker} Price Target Following Earnings Beat",
        f"{ticker} Faces Headwinds From Regulatory Concerns",
        f"Institutional Investors Increase Holdings in {ticker}",
        f"{ticker} Announces Strategic Partnership in AI Sector",
    ]
    for headline in headlines[:limit]:
        sentiment = random.choice(
            ["Bullish", "Somewhat-Bullish", "Neutral", "Somewhat-Bearish", "Bearish"]
        )
        feed.append(
            {
                "title": headline,
                "source": random.choice(["Reuters", "Bloomberg", "CNBC", "MarketWatch"]),
                "overall_sentiment_label": sentiment,
                "overall_sentiment_score": str(round(random.uniform(-0.5, 0.5), 6)),
                "time_published": time.strftime("%Y%m%dT%H%M%S"),
            }
        )
    return {"items": str(len(feed)), "sentiment_score_definition": {}, "feed": feed}


@app.route("/query", methods=["GET"])
def handle_query():
    function = request.args.get("function", "")
    if function == "OVERVIEW":
        return jsonify(_handle_overview(request.args))
    if function == "MARKET_STATUS":
        return jsonify(_handle_market_status())
    if function == "TOP_GAINERS_LOSERS":
        return jsonify(_handle_top_gainers_losers())
    if function == "NEWS_SENTIMENT":
        return jsonify(_handle_news_sentiment(request.args))
    return jsonify({"error": f"Unknown function: {function}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2345, debug=False)
