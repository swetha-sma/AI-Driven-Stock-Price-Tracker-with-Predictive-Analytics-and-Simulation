# AI-Driven-Stock-Price-Tracker-with-Predictive-Analytics-and-Simulation

**Problem statement:**

Tracking stock prices and learning how markets behave is often time-consuming and confusing for students, beginners, and casual traders.
Currently:
  Users switch between multiple apps (Yahoo Finance, broker apps, TradingView) just to check live prices.
  Beginners have no safe way to practice trades without risking real money.
  Most trading platforms are too complex with heavy charts, indicators, and jargon — intimidating for someone who only wants simple insights.
  There is no single tool that combines live price lookup, short-term prediction, and practice trading in one clean interface.

**Objective**

The objective of this project is to build a simple, AI-powered web application that:
  Shows real-time stock prices for multiple tickers on one page.
  Uses machine learning to predict whether a stock is likely to go up or stay flat in the next 5 minutes.
  Allows users to simulate trades safely and view their profit/loss progress.
  Provides an AI chatbot (RAG-based) to answer questions about prices, trades, and predictions based on real app data.


# Key Features

**Multi-Ticker Price Lookup**

View real-time prices for multiple stocks (e.g., AAPL MSFT TSLA) in a single, clean table with price, % change, and last update time.

**5-Minute Predictive Analytics**

AI model predicts whether the next 5-minute movement is likely to go UP or stay FLAT, with a confidence score (e.g., “58% chance UP → LONG”).

**Paper Trading Simulation**

Safely “pretend” buy/sell stocks without using real money and track open positions, average cost, and P&L.

**Profit Tracking Chart**

Visualize your profit/loss progress over time with a simple line chart.

**RAG-Based AI Chatbot**

Ask questions like:

“What was Tesla’s price 10 minutes ago?”
“Show my last 3 trades.”
Chatbot gives data-grounded answers with references to real app data.

# Tech Stack

**Programming Language**
 - Python

**Frontend / UI**
- Streamlit – single-page app with live price table, prediction panel, paper-trade simulation, and chatbot interface.
  
**Backend Framework**
- FastAPI – REST APIs for price lookup (/prices), prediction (/predict), trade simulation (/trade), profit tracking (/pnl), and chatbot queries (/chat/ask).

**Machine Learning Model**

- XGBoost – for next 5-minute stock price movement prediction (Up vs Flat), chosen for its speed and ability to capture non-linear patterns.
- Features: Recent 5-minute candles, returns, rolling mean/std, momentum indicators, wick/body ratios, volume change.
- Output: Probability of price going Up → displayed as “Confidence % + LONG/FLAT suggestion.”
  
**Vector Database**

- ChromaDB – used to store embeddings of recent prices, prediction outputs, trade logs, and help docs for the chatbot’s Retrieval-Augmented Generation (RAG) answers.

**APIs Used**

- Finnhub API – primary source for live quotes and 5-minute candles.
- Yahoo Finance (yfinance) – used for intraday 5-minute history and as a fallback when Finnhub limits are hit.
- LLM API (GPT-4/4o) – generates chatbot responses grounded in retrieved data.
- Embeddings: OpenAI or Sentence-Transformers for creating vector representations of data for RAG.
  
**Containerization & Deployment**

- Docker – containerizes the Streamlit + FastAPI app for portability
- AWS for cloud deployment


# Future Roadmap

**Sprint 1 — Core Functionality (Weeks 1–2)**

- Build single-page Streamlit UI: price lookup, prediction panel, and paper trading simulation.
- Connect Finnhub API (primary) + Yahoo Finance (fallback) for live and intraday 5-min data.
- Implement CSV-based trade ledger + simple profit/loss chart.
- Add real-time source labels (“Finnhub / Yahoo”) + last updated timestamp in the UI.
- Start baseline Logistic Regression model for next 5-minute price movement prediction.

**Sprint 2 — Machine Learning & Chatbot (Weeks 3–4)**

- Train XGBoost model on recent intraday candles and tune thresholds for “Up vs Flat.”
- Integrate prediction results into Streamlit with probability + LONG/FLAT suggestion.
- Implement ChromaDB vector store to index:
    - Recent price candles
    - Prediction results
    - Paper-trade logs
- Short help documentation
- Connect GPT-4/4o (LLM API) for chatbot responses with RAG (retrieval-augmented generation).
- Add chatbot interface block in the UI to allow natural language queries (e.g., “Show last 3 trades”).

**Sprint 3 — Cloud Deployment & Advanced Features (Weeks 5–6)**

- Containerize app with Docker (Streamlit + FastAPI).
- Push image to Amazon ECR and deploy on AWS App Runner with automatic scaling and HTTPS.
- Configure AWS Secrets Manager for API keys (Finnhub, GPT).
- Enable logging and monitoring with Amazon CloudWatch Logs.




