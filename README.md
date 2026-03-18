# Agentic AI in FinTech

An agentic AI project for financial question answering that compares baseline, single-agent, and multi-agent LLM architectures on live market, fundamentals, and sentiment tasks.

This project includes:
- a Streamlit chat app with conversational memory
- tool-integrated finance agents
- benchmark evaluation with an LLM-as-judge pipeline
- comparisons across `gpt-4o-mini` and `gpt-4o`

## Features

- `Baseline`, `Single Agent`, and `Multi-Agent` finance QA workflows
- Parallel multi-agent design with:
  - Market Agent
  - Fundamentals Agent
  - Sentiment Agent
  - Aggregator
- Tool calling over:
  - local SQLite stock database
  - Yahoo Finance price data
  - Alpha Vantage-style mock endpoints
- Streamlit demo with:
  - agent selector
  - model selector
  - full chat history
  - clear conversation button
  - 3-turn conversational memory

## Architecture

### 1. Baseline

The baseline sends the user question directly to the LLM with no tool access. It serves as a reference point to show the limitations of non-agentic responses in a live finance setting.

### 2. Single Agent

The single-agent setup gives one LLM access to all available tools. The agent decides which tools to call, in what order, and when it has enough information to answer.

### 3. Multi-Agent

The multi-agent setup splits the task across three parallel specialists:
- `Market Agent` for sector lookup, performance, and market status
- `Fundamentals Agent` for P/E, EPS, market cap, and company overview
- `Sentiment Agent` for recent finance news sentiment

An `Aggregator` combines specialist outputs into one final user-facing answer.

## Tools

- `get_tickers_by_sector`
- `get_price_performance`
- `get_company_overview`
- `get_market_status`
- `get_top_gainers_losers`
- `get_news_sentiment`
- `query_local_db`

## Project Files

- `app.py` - Streamlit chat interface
- `mp3_backend.py` - backend logic, tools, agent orchestration, evaluator, benchmark runner
- `mp3_assignment_part2_final.ipynb` - notebook version of the project
- `stocks.db` - local stock database
- `sp500_companies.csv` - source dataset for rebuilding the database
- `av_mock_server.py` - optional local mock server

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create a local `.env` or export environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ALPHAVANTAGE_API_KEY="your_alpha_vantage_key"
```

### 3. Run the app

```bash
streamlit run app.py
```

## Example Questions

- `Are the US stock markets open right now?`
- `Compare the 1-year price performance of AAPL, MSFT, and GOOGL. Which grew the most?`
- `Compare the P/E ratios of AAPL, MSFT, and NVDA. Which looks most expensive?`
- `Which tech stocks dropped this month but grew this year? Return the top 3.`

## Conversational Memory Demo

Use this 3-turn flow in the Streamlit app:

1. `What is NVIDIA's P/E ratio?`
2. `How does that compare to AMD?`
3. `Which of the two has better news sentiment right now?`

This demonstrates follow-up resolution across recent turns without repeating the entities.

## Evaluation

The project includes benchmark evaluation across easy, medium, and hard finance questions. Answers are scored with an LLM-as-judge evaluator on a `0-3` rubric for correctness, completeness, and likely hallucination.

## Resume Summary

This project demonstrates:
- agentic AI system design
- multi-agent orchestration
- LLM tool calling
- LLM evaluation
- Streamlit application development
- finance data reasoning

