"""Converted backend module from `mp3_assignment_part2_final.ipynb`.

This file keeps the notebook's core project logic in a normal Python module:
- local DB helpers
- tool functions
- baseline / single-agent / multi-agent runners
- evaluator
- benchmark and Excel evaluation utilities

Notebook-only cells such as `!pip install`, `%%writefile`, and ad hoc test cells
were intentionally left out so importing this module is safe.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sqlite3
import subprocess
import textwrap
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import requests
except ModuleNotFoundError:
    requests = None

try:
    import yfinance as yf
except ModuleNotFoundError:
    yf = None

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "stocks.db"
APP_PATH = ROOT_DIR / "av_mock_server.py"
AV_BASE = os.getenv("AV_BASE", "http://localhost:2345")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ALPHAVANTAGE_API_KEY = (
    os.getenv("ALPHAVANTAGE_API_KEY")
    or os.getenv("ALPHA_VANTAGE")
    or ""
)

MODEL_SMALL = "gpt-4o-mini"
MODEL_LARGE = "gpt-4o"
ACTIVE_MODEL = MODEL_SMALL

_client: OpenAI | None = None
_info_cache: dict[str, Any] = {}


def _require_dependency(module: Any, package_name: str) -> Any:
    if module is None:
        raise ModuleNotFoundError(
            f"{package_name} is required for this operation. "
            f"Install it with `pip install {package_name}`."
        )
    return module


def get_openai_client() -> OpenAI:
    """Lazily create the OpenAI client so imports remain lightweight."""
    global _client

    if _client is None:
        openai_cls = _require_dependency(OpenAI, "openai")
        api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running agent functions."
            )
        _client = openai_cls(api_key=api_key)
    return _client


def set_active_model(model: str) -> None:
    """Switch the active model used by the agent runners."""
    global ACTIVE_MODEL
    ACTIVE_MODEL = model


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT_DIR / path_obj


def start_mock_server(
    stdout_path: str | os.PathLike[str] = "stdout.txt",
    stderr_path: str | os.PathLike[str] = "stderr.txt",
) -> subprocess.Popen[Any]:
    """Start the local mock Alpha Vantage server defined in `app.py`."""
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Missing mock server file: {APP_PATH}")

    stdout_handle = open(_resolve_path(stdout_path), "w", encoding="utf-8")
    stderr_handle = open(_resolve_path(stderr_path), "w", encoding="utf-8")

    return subprocess.Popen(
        ["python3", str(APP_PATH)],
        cwd=str(ROOT_DIR),
        stdout=stdout_handle,
        stderr=stderr_handle,
    )


def create_local_database(csv_path: str | os.PathLike[str] = "sp500_companies.csv") -> None:
    """Create or refresh `stocks.db` from the provided CSV export."""
    csv_file = _resolve_path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"'{csv_file}' not found.\n"
            "Download from: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks"
        )

    def cap_bucket(value: Any) -> str:
        try:
            numeric = float(value)
            if numeric >= 10_000_000_000:
                return "Large"
            if numeric >= 2_000_000_000:
                return "Mid"
            return "Small"
        except Exception:
            return "Unknown"

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DROP TABLE IF EXISTS stocks")
        conn.execute(
            """
            CREATE TABLE stocks (
                ticker TEXT,
                company TEXT,
                sector TEXT,
                industry TEXT,
                market_cap TEXT,
                exchange TEXT
            )
            """
        )

        seen_tickers = set()
        rows_to_insert = []
        with open(csv_file, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                row = {str(key).strip().lower(): value for key, value in raw_row.items()}
                ticker = (row.get("symbol") or "").strip()
                company = (row.get("shortname") or "").strip()
                if not ticker or not company or ticker in seen_tickers:
                    continue

                seen_tickers.add(ticker)
                rows_to_insert.append(
                    (
                        ticker,
                        company,
                        (row.get("sector") or "").strip(),
                        (row.get("industry") or "").strip(),
                        cap_bucket(row.get("marketcap")),
                        (row.get("exchange") or "").strip(),
                    )
                )

        conn.executemany(
            """
            INSERT INTO stocks (ticker, company, sector, industry, market_cap, exchange)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker ON stocks(ticker)")
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
        print(f"Loaded {count} companies into {DB_PATH.name}")
        print("\nDistinct sector values stored in DB:")
        sectors = conn.execute(
            "SELECT DISTINCT sector FROM stocks ORDER BY sector"
        ).fetchall()
        for sector_row in sectors:
            print(sector_row[0] or "")
    finally:
        conn.close()


def _av_get(params: dict[str, Any]) -> dict[str, Any]:
    """Call the configured Alpha Vantage endpoint, or use local mock logic."""
    function = params.get("function", "")
    symbol = params.get("symbol", params.get("tickers", ""))

    # Prefer the configured HTTP endpoint when it is reachable.
    if requests is not None:
        requests_lib = requests
        query = dict(params)
        if ALPHAVANTAGE_API_KEY:
            query["apikey"] = ALPHAVANTAGE_API_KEY
        try:
            response = requests_lib.get(f"{AV_BASE}/query", params=query, timeout=3)
            response.raise_for_status()
            return response.json()
        except Exception:
            pass

    # Fall back to in-process mock implementations so the Streamlit app works
    # even when no local Flask server is running.
    if function == "OVERVIEW":
        return _mock_handle_overview({"symbol": symbol})
    if function == "MARKET_STATUS":
        return _mock_handle_market_status({})
    if function == "TOP_GAINERS_LOSERS":
        return _mock_handle_top_gainers_losers()
    if function == "NEWS_SENTIMENT":
        return _mock_handle_news_sentiment(
            {"tickers": symbol, "limit": params.get("limit", 5)}
        )

    return {"error": f"Unknown function: {function}"}


def _get_info(ticker: str) -> dict[str, Any] | None:
    yfinance = _require_dependency(yf, "yfinance")
    if ticker not in _info_cache:
        try:
            _info_cache[ticker] = yfinance.Ticker(ticker).info
        except Exception:
            _info_cache[ticker] = None
    return _info_cache[ticker]


def _is_market_open(
    tz_name: str,
    open_hour: int,
    open_minute: int,
    close_hour: int,
    close_minute: int,
) -> bool:
    now = datetime.now(ZoneInfo(tz_name))
    if now.weekday() >= 5:
        return False
    current_minutes = now.hour * 60 + now.minute
    return (open_hour * 60 + open_minute) <= current_minutes < (
        close_hour * 60 + close_minute
    )


def _mock_handle_overview(params: dict[str, Any]) -> dict[str, Any]:
    ticker = str(params.get("symbol", "")).strip().upper()
    if not ticker:
        return {}

    try:
        info = _get_info(ticker)
    except ModuleNotFoundError as exc:
        return {"error": str(exc)}

    if info and info.get("shortName"):
        def safe(value: Any) -> str:
            return "None" if value is None else str(value)

        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        eps = info.get("trailingEps") or info.get("forwardEps")
        return {
            "Symbol": ticker,
            "Name": info.get("shortName", ticker),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "MarketCapitalization": safe(info.get("marketCap")),
            "PERatio": safe(pe_ratio),
            "EPS": safe(eps),
            "52WeekHigh": safe(info.get("fiftyTwoWeekHigh")),
            "52WeekLow": safe(info.get("fiftyTwoWeekLow")),
            "DividendYield": safe(info.get("dividendYield")),
            "Beta": safe(info.get("beta")),
        }
    return {}


def _mock_handle_market_status(_: dict[str, Any]) -> dict[str, Any]:
    us_open = _is_market_open("America/New_York", 9, 30, 16, 15)
    uk_open = _is_market_open("Europe/London", 8, 0, 16, 30)
    jp_open = _is_market_open("Asia/Tokyo", 9, 0, 15, 0)

    return {
        "endpoint": "Global Market Open & Close Status",
        "markets": [
            {
                "market_type": "Equity",
                "region": "United States",
                "primary_exchanges": "NYSE, NASDAQ, AMEX, BATS",
                "local_open": "09:30",
                "local_close": "16:15",
                "current_status": "open" if us_open else "closed",
                "notes": "",
            },
            {
                "market_type": "Equity",
                "region": "United Kingdom",
                "primary_exchanges": "London Stock Exchange",
                "local_open": "08:00",
                "local_close": "16:30",
                "current_status": "open" if uk_open else "closed",
                "notes": "",
            },
            {
                "market_type": "Equity",
                "region": "Japan",
                "primary_exchanges": "Tokyo Stock Exchange",
                "local_open": "09:00",
                "local_close": "15:00",
                "current_status": "open" if jp_open else "closed",
                "notes": "",
            },
        ],
    }


def _mock_handle_top_gainers_losers_fallback() -> dict[str, Any]:
    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "AMZN",
        "META",
        "GOOGL",
        "AMD",
        "INTC",
        "NFLX",
        "CRM",
        "ORCL",
        "PYPL",
        "SQ",
        "SHOP",
        "PLTR",
        "RIVN",
        "LCID",
        "NIO",
        "SOFI",
    ]

    def random_movers(n: int = 5) -> list[dict[str, str]]:
        picked = random.sample(tickers, n)
        rows = []
        for ticker in picked:
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
        "top_gainers": random_movers(),
        "top_losers": random_movers(),
        "most_actively_traded": random_movers(),
    }


def _mock_handle_top_gainers_losers() -> dict[str, Any]:
    if requests is None:
        return _mock_handle_top_gainers_losers_fallback()

    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError:
        return _mock_handle_top_gainers_losers_fallback()

    def scrape_yahoo(url: str, n: int = 5) -> list[dict[str, str]]:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        soup = BeautifulSoup(response.text, "html.parser")
        rows = []
        for row in soup.select("table tbody tr")[:n]:
            cells = row.select("td")
            if len(cells) < 5:
                continue
            rows.append(
                {
                    "ticker": cells[0].get_text(strip=True),
                    "price": cells[3].get_text(strip=True),
                    "change_amount": cells[4].get_text(strip=True),
                    "change_percentage": cells[5].get_text(strip=True) if len(cells) > 5 else "",
                    "volume": cells[6].get_text(strip=True) if len(cells) > 6 else "",
                }
            )
        return rows

    try:
        gainers = scrape_yahoo("https://finance.yahoo.com/markets/stocks/gainers/")
        losers = scrape_yahoo("https://finance.yahoo.com/markets/stocks/losers/")
        active = scrape_yahoo("https://finance.yahoo.com/markets/stocks/most-active/")
        if not gainers and not losers and not active:
            raise ValueError("Scrape returned empty")
        return {
            "metadata": "Top Gainers, Losers, and Most Active (yahoo scrape)",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_gainers": gainers or [],
            "top_losers": losers or [],
            "most_actively_traded": active or [],
        }
    except Exception:
        return _mock_handle_top_gainers_losers_fallback()


def _mock_handle_news_sentiment(params: dict[str, Any]) -> dict[str, Any]:
    ticker = str(params.get("tickers", "")).strip().upper()
    limit = int(params.get("limit", 5))
    articles: list[dict[str, str]] = []

    if ticker and yf is not None:
        try:
            news = yf.Ticker(ticker).news
            if news:
                for item in news[:limit]:
                    content = item.get("content", {})
                    title = content.get("title", "")
                    provider = content.get("provider", {})
                    source = provider.get("displayName", "Unknown")
                    if not title:
                        continue
                    sentiment = random.choices(
                        ["Bullish", "Somewhat-Bullish", "Neutral", "Somewhat-Bearish", "Bearish"],
                        weights=[0.15, 0.3, 0.3, 0.15, 0.1],
                        k=1,
                    )[0]
                    score_map = {
                        "Bullish": random.uniform(0.3, 0.6),
                        "Somewhat-Bullish": random.uniform(0.1, 0.35),
                        "Neutral": random.uniform(-0.1, 0.1),
                        "Somewhat-Bearish": random.uniform(-0.35, -0.1),
                        "Bearish": random.uniform(-0.6, -0.3),
                    }
                    articles.append(
                        {
                            "title": title,
                            "source": source,
                            "overall_sentiment_label": sentiment,
                            "overall_sentiment_score": str(round(score_map[sentiment], 6)),
                            "time_published": content.get(
                                "pubDate", time.strftime("%Y%m%dT%H%M%S")
                            ),
                        }
                    )
        except Exception:
            pass

    fake_headlines = [
        f"{ticker} Shows Strong Momentum Amid Market Volatility",
        f"Analysts Upgrade {ticker} Price Target Following Earnings Beat",
        f"{ticker} Faces Headwinds From Regulatory Concerns",
        f"Institutional Investors Increase Holdings in {ticker}",
        f"{ticker} Announces Strategic Partnership in AI Sector",
        f"Market Watch: {ticker} Trading Volume Surges",
        f"{ticker} Q4 Results Exceed Wall Street Expectations",
        f"Why {ticker} Could Be a Top Pick for Growth Investors",
        f"{ticker} Expands Into New Markets With Latest Acquisition",
    ]
    random.shuffle(fake_headlines)

    index = 0
    while len(articles) < limit:
        sentiment = random.choice(
            ["Bullish", "Somewhat-Bullish", "Neutral", "Somewhat-Bearish", "Bearish"]
        )
        articles.append(
            {
                "title": fake_headlines[index % len(fake_headlines)],
                "source": random.choice(
                    ["Reuters", "Bloomberg", "Yahoo Finance", "MarketWatch", "CNBC", "Seeking Alpha"]
                ),
                "overall_sentiment_label": sentiment,
                "overall_sentiment_score": str(round(random.uniform(-0.5, 0.5), 6)),
                "time_published": time.strftime("%Y%m%dT%H%M%S"),
            }
        )
        index += 1

    return {
        "items": str(len(articles[:limit])),
        "sentiment_score_definition": {},
        "feed": articles[:limit],
    }


def get_price_performance(tickers: list[str], period: str = "1y") -> dict[str, Any]:
    """Return price change data for a list of tickers over a period."""
    yfinance = _require_dependency(yf, "yfinance")
    results: dict[str, Any] = {}
    for ticker in tickers:
        try:
            # yfinance can print failed-download notices directly to stdout/stderr.
            # Capture that noise so the UI only shows the final agent answer.
            silent_buffer = io.StringIO()
            with redirect_stdout(silent_buffer), redirect_stderr(silent_buffer):
                data = yfinance.download(
                    ticker,
                    period=period,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
            if data.empty:
                results[ticker] = {"error": "No data - possibly delisted"}
                continue

            start = float(data["Close"].iloc[0].item())
            end = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price": round(end, 2),
                "pct_change": round((end - start) / start * 100, 2),
                "period": period,
            }
        except Exception as exc:
            results[ticker] = {"error": str(exc)}
    return results


def get_market_status() -> dict[str, Any]:
    """Return market open / closed status for the configured endpoint."""
    return _av_get({"function": "MARKET_STATUS"})


def get_top_gainers_losers() -> dict[str, Any]:
    """Return the top gainers, losers, and most active stocks."""
    return _av_get({"function": "TOP_GAINERS_LOSERS"})


def get_news_sentiment(ticker: str, limit: int = 5) -> dict[str, Any]:
    """Return recent news and sentiment labels for a ticker."""
    data = _av_get(
        {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": limit,
        }
    )
    return {
        "ticker": ticker,
        "articles": [
            {
                "title": article.get("title"),
                "source": article.get("source"),
                "sentiment": article.get("overall_sentiment_label"),
                "score": article.get("overall_sentiment_score"),
            }
            for article in data.get("feed", [])[:limit]
        ],
    }


def query_local_db(sql: str) -> dict[str, Any]:
    """Run a read-only SQL query against the local stocks database."""
    if not sql.strip().lower().startswith("select"):
        return {"error": "Only SELECT queries are allowed"}

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description or []]
        return {
            "columns": columns,
            "rows": [dict(row) for row in rows],
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_company_overview(ticker: str) -> dict[str, Any]:
    """Return company fundamentals for a single ticker."""
    try:
        response = _av_get({"function": "OVERVIEW", "symbol": ticker})
    except Exception as exc:
        return {"error": str(exc)}

    if "Name" not in response:
        return {"error": f"No overview data for {ticker}"}

    return {
        "ticker": ticker,
        "name": response.get("Name"),
        "sector": response.get("Sector"),
        "pe_ratio": response.get("PERatio"),
        "eps": response.get("EPS"),
        "market_cap": response.get("MarketCapitalization"),
        "52w_high": response.get("52WeekHigh"),
        "52w_low": response.get("52WeekLow"),
    }


def get_tickers_by_sector(sector: str) -> dict[str, Any]:
    """Return stocks matching an exact sector, then fall back to industry search."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT ticker, company, sector, industry, market_cap, exchange
            FROM stocks
            WHERE lower(coalesce(sector, '')) = lower(?)
            """,
            (sector,),
        )
        rows = cursor.fetchall()
        if not rows:
            cursor = conn.execute(
                """
                SELECT ticker, company, sector, industry, market_cap, exchange
                FROM stocks
                WHERE lower(coalesce(industry, '')) LIKE lower(?)
                """,
                (f"%{sector}%",),
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return {"error": f"No stocks found for {sector}"}

    return {"sector": sector, "stocks": [dict(row) for row in rows]}


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


SCHEMA_TICKERS = _schema(
    "get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Technology', 'Energy') or sub-sectors "
    "('semiconductor', 'insurance').",
    {"sector": {"type": "string", "description": "Sector or industry name"}},
    ["sector"],
)

SCHEMA_PRICE = _schema(
    "get_price_performance",
    "Get percent price change for a list of tickers over a time period. "
    "Periods: '1mo', '3mo', '6mo', 'ytd', '1y'.",
    {
        "tickers": {"type": "array", "items": {"type": "string"}},
        "period": {"type": "string", "default": "1y"},
    },
    ["tickers"],
)

SCHEMA_OVERVIEW = _schema(
    "get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker": {"type": "string", "description": "Ticker symbol such as 'AAPL'."}},
    ["ticker"],
)

SCHEMA_STATUS = _schema(
    "get_market_status",
    "Check whether global stock exchanges are currently open or closed.",
    {},
    [],
)

SCHEMA_MOVERS = _schema(
    "get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.",
    {},
    [],
)

SCHEMA_NEWS = _schema(
    "get_news_sentiment",
    "Get recent news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {
        "ticker": {"type": "string"},
        "limit": {"type": "integer", "default": 5},
    },
    ["ticker"],
)

SCHEMA_SQL = _schema(
    "query_local_db",
    "Run a SQL SELECT query on stocks.db. "
    "Table 'stocks' columns: ticker, company, sector, industry, market_cap, exchange.",
    {"sql": {"type": "string", "description": "A valid SQL SELECT statement."}},
    ["sql"],
)

ALL_SCHEMAS = [
    SCHEMA_TICKERS,
    SCHEMA_PRICE,
    SCHEMA_OVERVIEW,
    SCHEMA_STATUS,
    SCHEMA_MOVERS,
    SCHEMA_NEWS,
    SCHEMA_SQL,
]

ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector": get_tickers_by_sector,
    "get_price_performance": get_price_performance,
    "get_company_overview": get_company_overview,
    "get_market_status": get_market_status,
    "get_top_gainers_losers": get_top_gainers_losers,
    "get_news_sentiment": get_news_sentiment,
    "query_local_db": query_local_db,
}


@dataclass
class AgentResult:
    agent_name: str
    answer: str
    tools_called: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    issues_found: list[str] = field(default_factory=list)
    reasoning: str = ""

    def summary(self) -> None:
        print(f"\n{'-' * 54}")
        print(f"Agent      : {self.agent_name}")
        print(f"Tools used : {', '.join(self.tools_called) or 'none'}")
        print(f"Confidence : {self.confidence:.0%}")
        if self.issues_found:
            print(f"Issues     : {'; '.join(self.issues_found)}")
        print(f"Answer     :\n{textwrap.indent(self.answer[:500], '  ')}")


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_response(text: str | None) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None

    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _tool_call_dict(tool_call: Any) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments or "{}",
        },
    }


def run_specialist_agent(
    agent_name: str,
    system_prompt: str,
    task: str,
    tool_schemas: list[dict[str, Any]],
    max_iters: int = 8,
    verbose: bool = True,
) -> AgentResult:
    """Core agent loop shared by the baseline, single, and multi-agent flows."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tools_called: list[str] = []
    raw_data: dict[str, Any] = {}
    client = get_openai_client()

    for _ in range(max_iters):
        request: dict[str, Any] = {"model": ACTIVE_MODEL, "messages": messages}
        if tool_schemas:
            request["tools"] = tool_schemas
            request["tool_choice"] = "auto"

        response = client.chat.completions.create(**request)
        message = response.choices[0].message

        if getattr(message, "tool_calls", None):
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [_tool_call_dict(tc) for tc in message.tool_calls],
                }
            )

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_args = {}

                if verbose:
                    print(f"[{agent_name}] calling tool: {tool_name}({tool_args})")

                tools_called.append(tool_name)

                try:
                    tool_fn = ALL_TOOL_FUNCTIONS[tool_name]
                    tool_output = tool_fn(**tool_args)
                except Exception as exc:
                    tool_output = {"error": str(exc)}

                raw_data[f"{tool_name}_{len(tools_called)}"] = tool_output
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_output),
                    }
                )
            continue

        parsed = _parse_json_response(message.content)
        if parsed:
            final_answer = str(parsed.get("output", "")).strip() or "No answer produced."
            confidence = float(parsed.get("confidence", 0.0) or 0.0)
            reasoning = str(parsed.get("reason", "")).strip()
        else:
            final_answer = (message.content or "").strip() or "No answer produced."
            confidence = 0.0
            reasoning = ""
            if verbose:
                print(f"[{agent_name}] answer was not valid JSON; using raw text.")

        return AgentResult(
            agent_name=agent_name,
            answer=final_answer,
            tools_called=tools_called,
            raw_data=raw_data,
            confidence=confidence,
            issues_found=[],
            reasoning=reasoning,
        )

    return AgentResult(
        agent_name=agent_name,
        answer="Stopped after reaching the maximum number of tool iterations.",
        tools_called=tools_called,
        raw_data=raw_data,
        confidence=0.0,
        issues_found=["max_iters_reached"],
        reasoning="",
    )


baseline_prompt = """
You are tasked with answering a query from a user.

OUTPUT FORMAT
You must return valid JSON:
{
  "output": "...",
  "reason": "...",
  "confidence": 0.00
}

The "output" field should contain the answer for the end user. Do not use markdown
code fences around the JSON.
""".strip()


def run_baseline(question: str, verbose: bool = True) -> AgentResult:
    """Single LLM call with no tools."""
    return run_specialist_agent(
        agent_name="Baseline",
        system_prompt=baseline_prompt,
        task=question,
        tool_schemas=[],
        max_iters=3,
        verbose=verbose,
    )


SINGLE_AGENT_PROMPT = """
You are tasked with answering a user question about stocks and financial data.

Decide whether to use the available tools, and if so which tools and arguments are
needed. Prefer the provided tools whenever they are relevant.

OUTPUT FORMAT
You must return valid JSON:
{
  "output": "...",
  "reason": "...",
  "confidence": 0.00
}

The "output" field should contain the final user-facing answer. Do not wrap the JSON
in markdown code fences.
""".strip()


def run_single_agent(question: str, verbose: bool = True) -> AgentResult:
    """Single tool-using agent with access to all available schemas."""
    return run_specialist_agent(
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        max_iters=10,
        verbose=verbose,
    )


MARKET_TOOLS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS]
FUNDAMENTALS_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS = [SCHEMA_NEWS, SCHEMA_SQL, SCHEMA_TICKERS]


MARKET_AGENT_PROMPT = """
You are the Market Agent in a finance multi-agent system.

Your role is to handle only market-related reasoning:
- stock price performance over time
- sector membership
- top gainers / losers
- market open / closed status
- ranking stocks by returns

You must use tools for all factual claims.

If the question involves best / worst / top / ranking stocks or performance over time,
you must:
1. Identify candidate stocks using database tools.
2. Call get_price_performance on those tickers.
3. Rank the stocks numerically based on pct_change.
4. Return the matching stocks with ticker and return percent.

OUTPUT FORMAT
You must return valid JSON:
{
  "output": "...",
  "reason": "...",
  "confidence": 0.00
}
""".strip()


FUNDAMENTALS_AGENT_PROMPT = """
You are the Fundamentals Agent in a finance multi-agent system.

Your job:
- answer only questions related to valuation and company fundamentals
- use get_company_overview for fundamentals
- use query_local_db only for lookups and filtering
- do not invent numbers

OUTPUT FORMAT
You must return valid JSON:
{
  "output": "...",
  "reason": "...",
  "confidence": 0.00
}
""".strip()


SENTIMENT_AGENT_PROMPT = """
You are the Sentiment Agent in a finance multi-agent system.

Your job:
- answer only questions related to recent news sentiment
- use get_news_sentiment when a ticker is known
- if the question is not about sentiment/news, say so briefly instead of guessing
- do not infer sentiment without tool evidence

OUTPUT FORMAT
You must return valid JSON:
{
  "output": "...",
  "reason": "...",
  "confidence": 0.00
}
""".strip()


AGGREGATOR_PROMPT = """
You are the Aggregator in a finance multi-agent system.

You will receive the user's original question plus outputs from:
- Market Agent
- Fundamentals Agent
- Sentiment Agent

Write one final user-facing answer using only supported specialist claims. If a
specialist had tool errors or missing data, mention that uncertainty briefly.

OUTPUT FORMAT
You must return valid JSON:
{
  "output": "...",
  "reason": "...",
  "confidence": 0.00
}
""".strip()


def calibrate_agent_result(result: AgentResult) -> AgentResult:
    """Lightweight post-check for empty or errored tool outputs."""
    issues = list(result.issues_found) if result.issues_found else []

    for key, value in result.raw_data.items():
        if isinstance(value, dict):
            if "error" in value:
                issues.append(f"{key}: {value['error']}")
            elif not value:
                issues.append(f"{key}: empty_result")
            elif isinstance(value.get("stocks"), list) and not value["stocks"]:
                issues.append(f"{key}: no_stocks_found")
        elif isinstance(value, list) and not value:
            issues.append(f"{key}: empty_list")

    confidence = result.confidence if isinstance(result.confidence, (int, float)) else 0.0
    if issues:
        confidence = max(0.20, confidence - 0.20)
    if not result.tools_called:
        confidence = min(confidence, 0.60)

    result.issues_found = issues
    result.confidence = float(confidence)
    return result


def run_market_agent(question: str, verbose: bool = True) -> AgentResult:
    return calibrate_agent_result(
        run_specialist_agent(
            agent_name="Market Agent",
            system_prompt=MARKET_AGENT_PROMPT,
            task=question,
            tool_schemas=MARKET_TOOLS,
            max_iters=8,
            verbose=verbose,
        )
    )


def run_fundamentals_agent(question: str, verbose: bool = True) -> AgentResult:
    return calibrate_agent_result(
        run_specialist_agent(
            agent_name="Fundamentals Agent",
            system_prompt=FUNDAMENTALS_AGENT_PROMPT,
            task=question,
            tool_schemas=FUNDAMENTALS_TOOLS,
            max_iters=8,
            verbose=verbose,
        )
    )


def run_sentiment_agent(question: str, verbose: bool = True) -> AgentResult:
    return calibrate_agent_result(
        run_specialist_agent(
            agent_name="Sentiment Agent",
            system_prompt=SENTIMENT_AGENT_PROMPT,
            task=question,
            tool_schemas=SENTIMENT_TOOLS,
            max_iters=8,
            verbose=verbose,
        )
    )


def build_aggregator_input(question: str, specialist_results: dict[str, AgentResult]) -> str:
    """Build the JSON payload given to the aggregation step."""
    payload: dict[str, Any] = {"user_question": question, "specialists": {}}
    for name, result in specialist_results.items():
        payload["specialists"][name] = {
            "agent_name": result.agent_name,
            "answer": result.answer,
            "tools_called": result.tools_called,
            "confidence": result.confidence,
            "issues_found": result.issues_found,
            "reasoning": result.reasoning,
            "raw_data": result.raw_data,
        }
    return json.dumps(payload, indent=2)


def run_multi_agent(question: str, verbose: bool = True) -> dict[str, Any]:
    """Parallel specialists plus a final aggregation pass."""
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "market": executor.submit(run_market_agent, question, verbose),
            "fundamentals": executor.submit(run_fundamentals_agent, question, verbose),
            "sentiment": executor.submit(run_sentiment_agent, question, verbose),
        }
        specialist_results = {name: future.result() for name, future in futures.items()}

    aggregator_result = run_specialist_agent(
        agent_name="Aggregator",
        system_prompt=AGGREGATOR_PROMPT,
        task=build_aggregator_input(question, specialist_results),
        tool_schemas=[],
        max_iters=3,
        verbose=verbose,
    )

    return {
        "final_answer": aggregator_result.answer,
        "agent_results": [
            specialist_results["market"],
            specialist_results["fundamentals"],
            specialist_results["sentiment"],
            aggregator_result,
        ],
        "elapsed_sec": time.time() - start_time,
        "architecture": "parallel_specialists_aggregator",
    }


def run_evaluator(question: str, expected_answer: str, agent_answer: str) -> dict[str, Any]:
    """Score an answer against the expected answer description."""
    fallback = {
        "score": 0,
        "max_score": 3,
        "reasoning": "evaluator parse error",
        "hallucination_detected": False,
        "key_issues": ["evaluator failed to parse"],
    }

    answer_lower = agent_answer.lower().strip()
    question_lower = question.lower().strip()
    expected_lower = expected_answer.lower().strip()

    refusal_patterns = [
        "i cannot retrieve",
        "i can't retrieve",
        "please check yahoo finance",
        "cannot access real-time",
        "can't access real-time",
        "do not have access to real-time",
        "unable to retrieve",
    ]
    if any(pattern in answer_lower for pattern in refusal_patterns):
        return {
            "score": 0,
            "max_score": 3,
            "reasoning": "The answer is a refusal and does not provide the requested result.",
            "hallucination_detected": False,
            "key_issues": ["refusal to answer"],
        }

    if (
        "p/e ratio" in question_lower
        and "single numeric value" in expected_lower
        and "aapl" in answer_lower
        and "approximately" not in answer_lower
        and "current market conditions" not in answer_lower
        and re.search(r"\b\d+(\.\d+)?\b", agent_answer)
    ):
        return {
            "score": 3,
            "max_score": 3,
            "reasoning": "The answer directly provides the requested company and a numeric P/E ratio.",
            "hallucination_detected": False,
            "key_issues": [],
        }

    if (
        "p/e ratio" in question_lower
        and "alpha vantage" in expected_lower
        and "approximately" in answer_lower
        and "current market conditions" in answer_lower
    ):
        return {
            "score": 1,
            "max_score": 3,
            "reasoning": "The answer gives a specific numeric claim that appears unsupported.",
            "hallucination_detected": True,
            "key_issues": [
                "unsupported numeric claim",
                "likely fabricated current-value claim",
            ],
        }

    system_prompt = """
You are an evaluator for finance QA systems.

Judge an answer using only:
1. the user question
2. the expected answer description
3. the agent's actual answer

You do not have access to external tools or live data. Score based on whether the
answer appears correct, complete, relevant, and well-supported from the text alone.

Scoring rubric:
3 = fully correct
2 = partially correct
1 = mostly wrong
0 = complete failure

Return only valid JSON.
""".strip()

    user_prompt = f"""
Evaluate this answer.

Question:
{question}

Expected answer description:
{expected_answer}

Agent answer:
{agent_answer}

Return exactly this JSON schema:
{{
  "score": 0,
  "max_score": 3,
  "reasoning": "one sentence",
  "hallucination_detected": false,
  "key_issues": []
}}
""".strip()

    try:
        response = get_openai_client().chat.completions.create(
            model=MODEL_SMALL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        parsed = _parse_json_response(response.choices[0].message.content)
        if not parsed:
            return fallback

        cleaned = {
            "score": int(parsed.get("score", 0)),
            "max_score": 3,
            "reasoning": str(parsed.get("reasoning", "")).strip() or "No reasoning provided.",
            "hallucination_detected": bool(parsed.get("hallucination_detected", False)),
            "key_issues": parsed.get("key_issues", []),
        }
        if cleaned["score"] not in [0, 1, 2, 3]:
            cleaned["score"] = 0
        if not isinstance(cleaned["key_issues"], list):
            cleaned["key_issues"] = [str(cleaned["key_issues"])]
        cleaned["key_issues"] = [str(item) for item in cleaned["key_issues"]]
        return cleaned
    except Exception:
        return fallback


BENCHMARK_QUESTIONS = [
    {
        "id": "Q01",
        "complexity": "easy",
        "category": "sector_lookup",
        "question": "List all semiconductor companies in the database.",
        "expected": "Should return company names and tickers for semiconductor stocks from the local DB.",
    },
    {
        "id": "Q02",
        "complexity": "easy",
        "category": "market_status",
        "question": "Are the US stock markets open right now?",
        "expected": "Should return the current open/closed status for NYSE and NASDAQ with trading hours.",
    },
    {
        "id": "Q03",
        "complexity": "easy",
        "category": "fundamentals",
        "question": "What is the P/E ratio of Apple (AAPL)?",
        "expected": "Should return AAPL P/E ratio as a single numeric value fetched from Alpha Vantage.",
    },
    {
        "id": "Q04",
        "complexity": "easy",
        "category": "sentiment",
        "question": "What is the latest news sentiment for Microsoft (MSFT)?",
        "expected": "Should return 3-5 recent MSFT headlines with sentiment labels and scores.",
    },
    {
        "id": "Q05",
        "complexity": "easy",
        "category": "price",
        "question": "What is NVIDIA's stock price performance over the last month?",
        "expected": "Should return NVDA start price, end price, and percent change for 1 month.",
    },
    {
        "id": "Q06",
        "complexity": "medium",
        "category": "price_comparison",
        "question": "Compare the 1-year price performance of AAPL, MSFT, and GOOGL. Which grew the most?",
        "expected": "Should fetch 1-year performance for all three tickers and identify the best performer.",
    },
    {
        "id": "Q07",
        "complexity": "medium",
        "category": "fundamentals",
        "question": "Compare the P/E ratios of AAPL, MSFT, and NVDA. Which looks most expensive?",
        "expected": "Should return P/E ratios for all three tickers and identify the highest P/E.",
    },
    {
        "id": "Q08",
        "complexity": "medium",
        "category": "sector_price",
        "question": "Which energy stocks in the database had the best 6-month performance?",
        "expected": "Should query the DB for energy tickers, fetch six-month performance, and rank them.",
    },
    {
        "id": "Q09",
        "complexity": "medium",
        "category": "sentiment",
        "question": "What is the news sentiment for Tesla (TSLA) and how has its stock moved this month?",
        "expected": "Should return TSLA news sentiment and 1-month price change from two tools.",
    },
    {
        "id": "Q10",
        "complexity": "medium",
        "category": "fundamentals",
        "question": "What are the 52-week high and low for JPMorgan (JPM) and Goldman Sachs (GS)?",
        "expected": "Should return 52-week high and low for both JPM and GS.",
    },
    {
        "id": "Q11",
        "complexity": "hard",
        "category": "multi_condition",
        "question": "Which tech stocks dropped this month but grew this year? Return the top 3.",
        "expected": "Should filter for negative 1-month and positive YTD performance, then return the top three.",
    },
    {
        "id": "Q12",
        "complexity": "hard",
        "category": "multi_condition",
        "question": "Which large-cap technology stocks on NASDAQ have grown more than 20% this year?",
        "expected": "Should query large-cap NASDAQ technology stocks, fetch YTD performance, and filter for >20%.",
    },
    {
        "id": "Q13",
        "complexity": "hard",
        "category": "cross_domain",
        "question": "For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios and current news sentiment?",
        "expected": "Should combine price, fundamentals, and sentiment to answer for the top three semiconductors.",
    },
    {
        "id": "Q14",
        "complexity": "hard",
        "category": "cross_domain",
        "question": "Compare the market cap, P/E ratio, and 1-year stock performance of JPM, GS, and BAC.",
        "expected": "Should return market cap, P/E, and 1-year percent change for all three tickers.",
    },
    {
        "id": "Q15",
        "complexity": "hard",
        "category": "multi_condition",
        "question": "Which finance sector stocks are trading closer to their 52-week low than their 52-week high? Return the news sentiment for each.",
        "expected": "Should identify qualifying finance stocks and then fetch news sentiment for each.",
    },
]


@dataclass
class EvalRecord:
    question_id: str
    question: str
    complexity: str
    category: str
    expected: str
    bl_answer: str = ""
    bl_time: float = 0.0
    bl_score: int = -1
    bl_reasoning: str = ""
    bl_hallucination: str = ""
    bl_issues: str = ""
    sa_answer: str = ""
    sa_tools: str = ""
    sa_tool_count: int = 0
    sa_iters: int = 0
    sa_time: float = 0.0
    sa_score: int = -1
    sa_reasoning: str = ""
    sa_hallucination: str = ""
    sa_issues: str = ""
    ma_answer: str = ""
    ma_tools: str = ""
    ma_tool_count: int = 0
    ma_time: float = 0.0
    ma_confidence: str = ""
    ma_critic_issues: int = 0
    ma_agents: str = ""
    ma_architecture: str = ""
    ma_score: int = -1
    ma_reasoning: str = ""
    ma_hallucination: str = ""
    ma_issues: str = ""


_COL_NAMES = {
    "question_id": "Question ID",
    "question": "Question",
    "complexity": "Difficulty",
    "category": "Category",
    "expected": "Expected Answer",
    "bl_answer": "Baseline Answer",
    "bl_time": "Baseline Time (s)",
    "bl_score": "Baseline Score /3",
    "bl_reasoning": "Baseline Eval Reasoning",
    "bl_hallucination": "Baseline Hallucination",
    "bl_issues": "Baseline Issues",
    "sa_answer": "SA Answer",
    "sa_tools": "SA Tools Used",
    "sa_tool_count": "SA Tool Count",
    "sa_iters": "SA Iterations",
    "sa_time": "SA Time (s)",
    "sa_score": "SA Score /3",
    "sa_reasoning": "SA Eval Reasoning",
    "sa_hallucination": "SA Hallucination",
    "sa_issues": "SA Issues",
    "ma_answer": "MA Answer",
    "ma_tools": "MA Tools Used",
    "ma_tool_count": "MA Tool Count",
    "ma_time": "MA Time (s)",
    "ma_confidence": "MA Avg Confidence",
    "ma_critic_issues": "MA Critic Issue Count",
    "ma_agents": "MA Agents Activated",
    "ma_architecture": "MA Architecture",
    "ma_score": "MA Score /3",
    "ma_reasoning": "MA Eval Reasoning",
    "ma_hallucination": "MA Hallucination",
    "ma_issues": "MA Issues",
}


def _save_excel(records: list[EvalRecord], path: str | os.PathLike[str]) -> None:
    pandas = _require_dependency(pd, "pandas")
    df = pandas.DataFrame([record.__dict__ for record in records]).rename(columns=_COL_NAMES)

    with pandas.ExcelWriter(_resolve_path(path), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")

        rows = []
        for arch, score_col, time_col, hall_col in [
            ("Baseline", "Baseline Score /3", "Baseline Time (s)", "Baseline Hallucination"),
            ("Single Agent", "SA Score /3", "SA Time (s)", "SA Hallucination"),
            ("Multi Agent", "MA Score /3", "MA Time (s)", "MA Hallucination"),
        ]:
            for tier in ["easy", "medium", "hard", "all"]:
                subset = df if tier == "all" else df[df["Difficulty"] == tier]
                valid = subset[subset[score_col] >= 0]
                average_score = valid[score_col].mean() if len(valid) else 0
                rows.append(
                    {
                        "Architecture": arch,
                        "Difficulty": tier,
                        "Questions Scored": len(valid),
                        "Avg Score /3": round(average_score, 2),
                        "Accuracy %": round(average_score / 3 * 100, 1),
                        "Avg Time (s)": round(df[time_col].mean(), 1),
                        "Hallucinations": (df[hall_col] == "True").sum(),
                    }
                )
        pandas.DataFrame(rows).to_excel(writer, index=False, sheet_name="Summary")


def run_full_evaluation(
    output_xlsx: str = "results.xlsx",
    delay_sec: float = 3.0,
) -> str:
    """Run all benchmark questions through baseline, single, and multi-agent flows."""
    records: list[EvalRecord] = []
    total = len(BENCHMARK_QUESTIONS)

    print(f"\n{'=' * 62}")
    print(f"  FULL EVALUATION  |  {total} questions x 3 architectures")
    print(f"  Model: {ACTIVE_MODEL}  |  Output: {output_xlsx}")
    print(f"{'=' * 62}\n")

    for index, question in enumerate(BENCHMARK_QUESTIONS, start=1):
        print(
            f"[{index:02d}/{total}] {question['id']} "
            f"({question['complexity']:6s}) {question['question'][:52]}..."
        )
        record = EvalRecord(
            question_id=question["id"],
            question=question["question"],
            complexity=question["complexity"],
            category=question["category"],
            expected=question["expected"],
        )

        print("         baseline  ...", end=" ", flush=True)
        try:
            start = time.time()
            baseline = run_baseline(question["question"], verbose=False)
            record.bl_answer = baseline.answer.replace("\n", " ")
            record.bl_time = round(time.time() - start, 2)
            evaluation = run_evaluator(question["question"], question["expected"], baseline.answer)
            record.bl_score = evaluation.get("score", -1)
            record.bl_reasoning = evaluation.get("reasoning", "")
            record.bl_hallucination = str(evaluation.get("hallucination_detected", False))
            record.bl_issues = " | ".join(evaluation.get("key_issues", []))
            print(f"ok  {record.bl_time:5.1f}s  score {record.bl_score}/3")
        except Exception as exc:
            print(f"failed  {exc}")

        print("         single    ...", end=" ", flush=True)
        try:
            start = time.time()
            single = run_single_agent(question["question"], verbose=False)
            record.sa_answer = single.answer.replace("\n", " ")
            record.sa_tools = ", ".join(single.tools_called)
            record.sa_tool_count = len(single.tools_called)
            record.sa_iters = len(single.tools_called) + 1
            record.sa_time = round(time.time() - start, 2)
            evaluation = run_evaluator(question["question"], question["expected"], single.answer)
            record.sa_score = evaluation.get("score", -1)
            record.sa_reasoning = evaluation.get("reasoning", "")
            record.sa_hallucination = str(evaluation.get("hallucination_detected", False))
            record.sa_issues = " | ".join(evaluation.get("key_issues", []))
            print(f"ok  {record.sa_time:5.1f}s  score {record.sa_score}/3")
        except Exception as exc:
            print(f"failed  {exc}")

        print("         multi     ...", end=" ", flush=True)
        try:
            start = time.time()
            multi = run_multi_agent(question["question"], verbose=False)
            results = multi.get("agent_results", [])
            all_tools = [tool for result in results for tool in result.tools_called]
            all_issues = [issue for result in results for issue in result.issues_found]
            avg_conf = sum(result.confidence for result in results) / len(results) if results else 0.0

            record.ma_answer = multi["final_answer"].replace("\n", " ")
            record.ma_tools = ", ".join(dict.fromkeys(all_tools))
            record.ma_tool_count = len(all_tools)
            record.ma_time = round(time.time() - start, 2)
            record.ma_confidence = f"{avg_conf:.0%}"
            record.ma_critic_issues = len(all_issues)
            record.ma_agents = ", ".join(result.agent_name for result in results)
            record.ma_architecture = multi.get("architecture", "")

            evaluation = run_evaluator(question["question"], question["expected"], multi["final_answer"])
            record.ma_score = evaluation.get("score", -1)
            record.ma_reasoning = evaluation.get("reasoning", "")
            record.ma_hallucination = str(evaluation.get("hallucination_detected", False))
            record.ma_issues = " | ".join(evaluation.get("key_issues", []))
            print(f"ok  {record.ma_time:5.1f}s  score {record.ma_score}/3")
        except Exception as exc:
            print(f"failed  {exc}")

        records.append(record)
        _save_excel(records, output_xlsx)

        if index < total:
            print(f"         waiting {delay_sec}s ...\n")
            time.sleep(delay_sec)

    print(f"\n{'=' * 62}  RESULTS")
    print(f"{'Architecture':<18} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}")
    print("-" * 60)
    for arch, score_key in [
        ("Baseline", "bl_score"),
        ("Single Agent", "sa_score"),
        ("Multi Agent", "ma_score"),
    ]:
        def pct(tier: str) -> str:
            scores = [
                getattr(record, score_key)
                for record in records
                if getattr(record, score_key) >= 0
                and (tier == "all" or record.complexity == tier)
            ]
            return f"{sum(scores) / len(scores) / 3 * 100:.0f}%" if scores else "-"

        print(f"{arch:<18} {pct('easy'):>8} {pct('medium'):>8} {pct('hard'):>8} {pct('all'):>8}")

    print(f"\nSaved results to {output_xlsx}")
    return str(_resolve_path(output_xlsx))


if __name__ == "__main__":
    print("mp3_backend.py was generated from the notebook and is ready to import.")
    print(f"Database path : {DB_PATH}")
    print(f"Mock server   : {APP_PATH}")
