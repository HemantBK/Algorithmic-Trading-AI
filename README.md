<p align="center">
  <img src="https://raw.githubusercontent.com/INFO-698-InfoSci-Capstone/algorithmic-trading-ai/main/Poster/stocex_logo.png" alt="Stocex Logo" width="200"/>
</p>

<h1 align="center">Stocex — AI-Powered Algorithmic Trading System</h1>

<p align="center">
  <strong>Real-time news sentiment analysis + time-series forecasting = actionable trade signals</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FinBERT-NLP-green?style=for-the-badge&logo=huggingface&logoColor=white" alt="FinBERT"/>
  <img src="https://img.shields.io/badge/TimeGPT-Forecasting-orange?style=for-the-badge" alt="TimeGPT"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-GPLv3-yellow?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <a href="#system-architecture">Architecture</a> &bull;
  <a href="#pipeline-deep-dive">Pipeline</a> &bull;
  <a href="#ml-models">ML Models</a> &bull;
  <a href="#dashboard">Dashboard</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#api-integrations">APIs</a>
</p>

---

## The Problem

Retail traders are overwhelmed. Thousands of news articles drop daily, markets move in milliseconds, and manual analysis can't keep up. By the time you've read the headline, the price has already moved.

## The Solution

**Stocex** is an end-to-end algorithmic trading pipeline that automates the entire workflow — from ingesting financial news to generating BUY/SELL/HOLD signals — by combining **NLP-based sentiment analysis** with **transformer-powered price forecasting**.

One script. Zero manual intervention. Actionable signals in minutes.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION LAYER                              │
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│   │   NewsAPI     │    │   Yahoo       │    │  S&P 500 Constituents      │  │
│   │  (Headlines)  │    │  Finance      │    │  (Ticker Mapping)          │  │
│   │  100+ daily   │    │  (OHLCV)      │    │  500+ companies            │  │
│   └──────┬───────┘    └──────┬───────┘    └──────────────┬───────────────┘  │
│          │                   │                            │                  │
└──────────┼───────────────────┼────────────────────────────┼──────────────────┘
           │                   │                            │
           ▼                   ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NLP PROCESSING LAYER                               │
│                                                                             │
│   ┌──────────────────┐    ┌──────────────────┐    ┌───────────────────┐    │
│   │    spaCy NER      │    │     FinBERT       │    │  Fuzzy Matcher   │    │
│   │  en_core_web_sm   │    │  Sentiment Model  │    │  Company→Ticker  │    │
│   │                   │    │                   │    │                  │    │
│   │  Extract ORG      │───▶│  Score: 0.0-1.0   │    │  "Apple"→AAPL   │    │
│   │  entities from    │    │  Label: pos/neg/  │    │  "Microsoft"→   │    │
│   │  headlines        │    │  neutral           │    │  MSFT            │    │
│   └───────────────────┘    └──────────┬────────┘    └───────────────────┘    │
│                                       │                                     │
└───────────────────────────────────────┼─────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FORECASTING ENGINE                                    │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                    TimeGPT-1 (Nixtla)                           │      │
│   │                                                                  │      │
│   │   Input: 30-day historical OHLCV data                           │      │
│   │   Output: 7-12 bar price forecast + confidence intervals        │      │
│   │   Confidence Bands: 80% | 90% | 95%                            │      │
│   │   Frequency: Daily / 5-min intraday                             │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SIGNAL GENERATION ENGINE                               │
│                                                                             │
│   ┌────────────────────────────────────────────────────────┐                │
│   │                                                        │                │
│   │   Sentiment Score ≥ 0.98  AND  Forecast ↑  →  BUY     │                │
│   │   Sentiment Score ≤ 0.02  AND  Forecast ↓  →  SELL    │                │
│   │   Conflicting / Neutral signals            →  HOLD    │                │
│   │                                                        │                │
│   └────────────────────────────────────────────────────────┘                │
│                                                                             │
│   Risk Controls:                                                            │
│   • High-confidence threshold (0.98) filters weak signals                   │
│   • Volatility-based position sizing                                        │
│   • Forecast confidence intervals as stop-loss boundaries                   │
│   • Minimum news mention count requirement                                  │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                                    │
│                                                                             │
│   ┌─────────────────┐  ┌────────────────┐  ┌──────────────────────────┐    │
│   │   Streamlit      │  │  Power BI       │  │  CSV Data Exports       │    │
│   │   Dashboard      │  │  Analytics      │  │                          │    │
│   │                  │  │                 │  │  • sentiment_summary     │    │
│   │  • Sentiment     │  │  • Advanced     │  │  • news_headlines        │    │
│   │  • News Feed     │  │    drill-down   │  │  • price_history         │    │
│   │  • Forecasts     │  │  • Cross-       │  │  • volatility_data       │    │
│   │  • AI Q&A        │  │    filtering    │  │  • forecast_results      │    │
│   │  • Dark Mode     │  │                 │  │                          │    │
│   └─────────────────┘  └────────────────┘  └──────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Deep Dive

The system executes a **7-stage sequential pipeline** — each stage feeds into the next:

```
  NewsAPI          spaCy NER        FinBERT         yFinance        TimeGPT
    │                 │                │                │               │
    ▼                 ▼                ▼                ▼               ▼
┌────────┐     ┌───────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Stage 1 │────▶│  Stage 2  │───▶│ Stage 3  │───▶│ Stage 4  │───▶│ Stage 5  │
│ Fetch   │     │ Extract   │    │ Score    │    │ Get      │    │ Forecast │
│ News    │     │ Companies │    │ Sentiment│    │ Prices   │    │ Prices   │
└────────┘     └───────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                                     │
                                      ┌──────────────────────────────┘
                                      ▼
                               ┌──────────┐    ┌──────────┐
                               │ Stage 6  │───▶│ Stage 7  │
                               │ Epoch    │    │ Generate │
                               │ Analysis │    │ Signals  │
                               └──────────┘    └──────────┘
```

### Stage 1 — News Ingestion
Pulls **100+ headlines daily** from 15 premium financial sources via NewsAPI:
- Bloomberg, WSJ, CNBC, Reuters, MarketWatch, Seeking Alpha, Barron's, Forbes, Fortune, TechCrunch, Business Insider, Yahoo Finance, Investopedia, The Motley Fool, CNN Business

Queries cover: earnings, mergers, acquisitions, IPOs, Fed decisions, guidance, revenue, buybacks, analyst ratings, and more.

### Stage 2 — Named Entity Recognition
Uses **spaCy's `en_core_web_sm`** model to extract organization entities (`ORG` labels) from each headline. Extracted names are fuzzy-matched against the full S&P 500 constituent list to resolve company names to ticker symbols (e.g., "Apple" -> `AAPL`).

### Stage 3 — Sentiment Scoring with FinBERT
Each headline is tokenized and passed through **FinBERT** (`yiyanghkust/finbert-tone`), a BERT model fine-tuned on 10,000+ financial texts. Outputs a 3-class probability distribution:
```
Input:  "Apple reports record Q4 earnings, beats estimates"
Output: { negative: 0.02, neutral: 0.08, positive: 0.90 }
        → Label: POSITIVE | Confidence: 0.90
```
Sentiment is aggregated per ticker across all mentioning articles. Only tickers exceeding the confidence threshold are forwarded.

### Stage 4 — Historical Price Data
Downloads **5-year daily** and **30-day intraday (5-min)** OHLCV data from Yahoo Finance for each qualifying ticker. Includes resampling logic to convert intraday bars to daily frequency when needed.

### Stage 5 — Price Forecasting with TimeGPT
Sends preprocessed time series to **Nixtla's TimeGPT-1** — a transformer-based foundation model trained on 100B+ data points. Returns forecasted values with multi-level confidence intervals:
```
Forecast Horizon: 7-12 bars
Confidence Bands: 80% (±1.28σ) | 90% (±1.64σ) | 95% (±1.96σ)
```

### Stage 6 — Superposed Epoch Analysis
A statistical technique borrowed from geophysics. Detects **price spike events** (Z-score > 2σ) and overlays price windows around each event to reveal average market behavior patterns before and after sentiment-driven events.

### Stage 7 — Signal Generation
Combines sentiment direction + forecast trajectory into actionable signals:

| Sentiment | Forecast | Signal |
|-----------|----------|--------|
| Positive (score >= 0.98) | Price forecast UP | **BUY** |
| Negative (score <= 0.02) | Price forecast DOWN | **SELL** |
| Neutral / Conflicting | Any | **HOLD** |

---

## ML Models

### FinBERT — Financial Sentiment Analysis

| Property | Detail |
|----------|--------|
| **Model** | `yiyanghkust/finbert-tone` |
| **Base Architecture** | BERT (Bidirectional Encoder Representations from Transformers) |
| **Training Data** | Financial news, analyst reports, earnings calls |
| **Input** | Raw text, max 512 tokens |
| **Output** | 3-class softmax: [negative, neutral, positive] |
| **Framework** | HuggingFace Transformers + PyTorch |
| **Why FinBERT over VADER/TextBlob?** | Domain-specific — understands that "short squeeze" is bearish context, not just the word "short" |

### TimeGPT-1 — Time Series Forecasting

| Property | Detail |
|----------|--------|
| **Provider** | Nixtla |
| **Architecture** | Transformer-based foundation model |
| **Training Data** | 100B+ time series data points across industries |
| **Input** | Historical OHLCV series (JSON) |
| **Output** | Point forecasts + confidence intervals |
| **Advantage** | Zero-shot forecasting — no fine-tuning required |

### spaCy NER — Entity Extraction

| Property | Detail |
|----------|--------|
| **Model** | `en_core_web_sm` |
| **Task** | Named Entity Recognition |
| **Target Labels** | `ORG` (Organizations) |
| **Post-Processing** | Fuzzy match against S&P 500 constituents list |

---

## Dashboard

Interactive **Streamlit** web dashboard with 5 tabs:

| Tab | Description |
|-----|-------------|
| **Sentiment Overview** | Histogram of sentiment scores, top mentioned tickers bar chart, volatility vs. return scatter plot, keyword frequency analysis |
| **News Headlines** | Filterable news feed with date/ticker selectors |
| **Historical Prices** | Multi-ticker price charts with 5-year lookback |
| **Forecast** | 1-hour simulated price forecast with confidence bands |
| **AI Q&A** | Natural language query interface — ask "highest sentiment?", "most volatile stock?", "return of AAPL?" |

**Features:** Dark/light mode toggle, sidebar filters (sentiment type, min mentions, score range), fully responsive layout.

```bash
streamlit run src/app.py
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **NLP** | HuggingFace Transformers, PyTorch | FinBERT sentiment analysis |
| **NER** | spaCy (`en_core_web_sm`) | Company name extraction |
| **Forecasting** | Nixtla TimeGPT-1 API | Price prediction |
| **Market Data** | yfinance | Historical & intraday OHLCV |
| **News** | NewsAPI | Financial headline ingestion |
| **Dashboard** | Streamlit, Altair | Interactive visualization |
| **Analytics** | Power BI | Advanced drill-down reporting |
| **Data Processing** | Pandas, NumPy | ETL and analysis |
| **Visualization** | Matplotlib, Altair | Charts and plots |
| **Execution** | Google Colab | Cloud-based pipeline runs |

---

## Quickstart

### Prerequisites
- Python 3.9+
- [NewsAPI key](https://newsapi.org/) (free tier available)
- [Nixtla API key](https://nixtla.io/) (for TimeGPT forecasting)

### Installation

```bash
# Clone the repository
git clone https://github.com/INFO-698-InfoSci-Capstone/algorithmic-trading-ai.git
cd algorithmic-trading-ai

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Configuration

Set your API keys in `src/Stocex.py`:
```python
NEWSAPI_KEY = "your_newsapi_key_here"
# TimeGPT key is set in the forecast_with_timegpt() function headers
```

### Run the Pipeline

```bash
# Execute the full trading pipeline
python src/Stocex.py
```

### Launch the Dashboard

```bash
# Start the interactive dashboard
streamlit run src/app.py
```

---

## Project Structure

```
algorithmic-trading-ai/
│
├── src/
│   ├── Stocex.py                  # Core trading pipeline (7-stage engine)
│   └── app.py                     # Streamlit dashboard application
│
├── Notebooks/
│   ├── Capstone_Project_Trading_AI.ipynb   # Full implementation notebook
│   └── Capstone_Project_Trading_AI.pdf     # Notebook export
│
├── analysis/
│   ├── data/
│   │   ├── sentiment_summary.csv           # Per-ticker sentiment scores
│   │   ├── news_headlines.csv              # Daily ingested headlines
│   │   ├── stock_volatility_data.csv       # Volatility metrics
│   │   ├── scatter_volatility_return.csv   # Risk-return profiles
│   │   ├── combined_price_data.csv         # 5-year price history
│   │   ├── newsapi_last_30_days.csv        # 30-day news archive
│   │   ├── news_keyword_frequency.csv      # Keyword extraction results
│   │   └── historical_price_data/          # Per-ticker CSV files
│   └── logs/
│       └── log.md                          # Development logs
│
├── Final Report/
│   ├── Stocex Dashboard.pbix              # Power BI dashboard
│   └── Stocex Dashboard.pdf               # Dashboard export
│
├── Visualization/
│   └── Stocex Visualization.pdf           # Visual analysis report
│
├── Poster/
│   └── stocex_logo.png                    # Project logo
│
├── requirements.txt                        # Python dependencies
├── CONDUCT.md                              # Code of conduct
├── LICENSE                                 # GNU GPLv3
└── README.md                               # You are here
```

---

## API Integrations

| API | Endpoint | Auth | Rate Limit | Data |
|-----|----------|------|-----------|------|
| **NewsAPI** | `newsapi.org/v2/everything` | API Key | 100 req/day (free) | Headlines, descriptions, sources |
| **TimeGPT** | `api.nixtla.io/forecast` | Bearer Token | Varies by plan | Point forecasts, confidence intervals |
| **Yahoo Finance** | via `yfinance` | None | Unofficial | OHLCV, intraday, historical |
| **S&P 500 List** | GitHub CSV | None | Unlimited | Company name to ticker mapping |

---

## Data Flow

```
                    ┌─────────────┐
                    │   NewsAPI    │
                    │  100+ daily  │
                    │  headlines   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  spaCy NER  │──── Extract company names
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   FinBERT   │──── Score sentiment (0.0 → 1.0)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐    │     ┌──────▼──────┐
       │  yFinance   │    │     │  Fuzzy Match │
       │  5yr daily  │    │     │  → S&P 500   │
       │  30d intra  │    │     │    tickers   │
       └──────┬──────┘    │     └─────────────┘
              │            │
       ┌──────▼──────┐    │
       │  TimeGPT-1  │    │
       │  7-12 bar   │    │
       │  forecast   │    │
       └──────┬──────┘    │
              │            │
              └────────────┤
                           │
                    ┌──────▼──────┐
                    │   Signal    │
                    │  Generator  │
                    │             │
                    │ BUY / SELL  │
                    │   / HOLD   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼───┐ ┌──────▼──────┐
       │  Streamlit  │ │ CSV  │ │  Power BI   │
       │  Dashboard  │ │Export │ │  Analytics  │
       └─────────────┘ └──────┘ └─────────────┘
```

---

## Risk Management

| Control | Implementation |
|---------|---------------|
| **Signal Confidence Filter** | Only acts on sentiment scores >= 0.98 — filters out noise |
| **Volume Validation** | Requires minimum article mention count before generating signals |
| **Volatility Analysis** | Calculates annualized volatility per ticker for position sizing |
| **Forecast Confidence Bands** | 80/90/95% intervals provide stop-loss and take-profit zones |
| **Epoch Analysis** | Validates signals against historical event-driven price patterns |

---

## Sample Output

```
📅 News fetched for 2025-04-10 — 87 articles from 15 sources
🧠 spaCy extracted 23 unique company mentions
📊 FinBERT scored sentiment across all headlines

🔍 Top Tickers by Sentiment:
   AAPL  →  Score: 0.987  |  Mentions: 12  |  Sentiment: POSITIVE
   TSLA  →  Score: 0.034  |  Mentions: 8   |  Sentiment: NEGATIVE
   NCLH  →  Score: 0.512  |  Mentions: 3   |  Sentiment: NEUTRAL

📈 TimeGPT Forecasts:
   AAPL  →  Current: $178.32  |  Forecast (7d): $182.15  |  Direction: ↑
   TSLA  →  Current: $245.10  |  Forecast (7d): $238.47  |  Direction: ↓

✅ AAPL: BUY  (positive sentiment + forecast UP)
❌ TSLA: SELL (negative sentiment + forecast DOWN)
🤝 NCLH: HOLD (neutral — conflicting signals)
```

---

## What I Learned Building This

- Designed and implemented a **multi-stage data pipeline** that chains NLP, time-series, and signal generation
- Integrated **3 ML models** (FinBERT, TimeGPT, spaCy NER) into a single cohesive system
- Built a **production-style dashboard** with Streamlit for real-time data exploration
- Applied **Superposed Epoch Analysis** — a statistical technique from geophysics — to financial data
- Worked with **5 external APIs** and handled rate limiting, auth, and data normalization
- Practiced **end-to-end ML engineering**: data ingestion, preprocessing, inference, post-processing, and visualization

---

## Future Roadmap

- [ ] **Live paper trading** — integrate with Alpaca API for simulated execution
- [ ] **WebSocket price feeds** — real-time streaming instead of batch polling
- [ ] **Database backend** — migrate from CSV to PostgreSQL/TimescaleDB
- [ ] **Backtesting framework** — walk-forward optimization with Sharpe/Sortino/Calmar metrics
- [ ] **CI/CD pipeline** — GitHub Actions for automated daily runs and testing
- [ ] **Alerting system** — Slack/email notifications on high-confidence signals
- [ ] **Portfolio-level risk** — correlation matrix, VaR, maximum drawdown limits
- [ ] **Multi-model ensemble** — combine TimeGPT with LSTM and Prophet for robust forecasts

---

## License

Licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built by <a href="mailto:stocex.team@gmail.com">Stocex Team</a></strong> | University of Arizona — School of Information
</p>

<p align="center">
  <em>"Let the AI read the news, so you can read the profits."</em>
</p>
