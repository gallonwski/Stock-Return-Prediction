# Stock Return Prediction using Machine Learning

This project develops a **machine learning framework for predicting short-term stock returns** using historical market data and technical indicators.

The objective is to evaluate whether machine learning models can identify predictive signals in financial data and whether these signals can be used to construct a portfolio that outperforms simple benchmark strategies.

---

# Project Objective

Financial markets contain a large amount of historical information. This project investigates whether patterns in past price and volume data can be used to **predict next-day stock returns**.

Two machine learning models are used:

- Linear Regression
- Random Forest Regressor

The predictions are then used to construct a **systematic trading strategy** whose performance is compared to benchmark portfolios.

---

# Data Source

Market data is obtained from **Yahoo Finance** using the `yfinance` Python library.

The dataset includes:

- Daily adjusted close prices
- Daily trading volume

The project analyzes a universe of **large-cap U.S. equities**.

Example tickers used in the analysis include:

AAPL, MSFT, AMZN, GOOGL, META, NVDA
JPM, BAC, GS, MS
XOM, CVX
JNJ, PFE, MRK
PG, KO, PEP, WMT



---

# Feature Engineering

Several financial indicators are created from raw market data to capture **momentum, volatility, and liquidity effects**.

### Return-based features
- 1-day lagged return
- 5-day return
- 20-day return

### Volatility
- 20-day rolling volatility of returns

### Trend indicators
- Distance from 5-day moving average
- Distance from 20-day moving average

### Volume indicators
- 1-day change in trading volume
- 5-day change in trading volume

The **target variable** is the **next-day return** of each stock.

---

# Model Training

The dataset is divided into:

- Training period
- Testing period

Two predictive models are trained on the training dataset.

## Linear Regression
A simple baseline model used to evaluate whether a linear relationship exists between features and future returns.

## Random Forest
A non-linear ensemble model capable of capturing complex interactions between variables.

The models predict the **expected next-day return** for each stock.

---

# Portfolio Construction

Model predictions are used to construct a simple **long-only investment strategy**.

Each day:

1. Stocks are ranked by predicted returns.
2. The top percentile of stocks is selected.
3. Selected stocks are equally weighted in the portfolio.

The resulting portfolio generates a **time series of daily returns**.

---

# Performance Evaluation

The strategy is evaluated using standard financial metrics:

- Cumulative Return
- CAGR (Compound Annual Growth Rate)
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio

The strategy is compared against two benchmarks:

- Equal-weighted portfolio of all stocks
- Market benchmark (SPY ETF)

---

# Additional Analysis

The project also evaluates the predictive quality of the models using:

## Feature Importance
Random Forest feature importance scores identify the most influential predictors.

## Information Coefficient (IC)
Measures the cross-sectional correlation between predicted returns and realized returns.

## CAPM Analysis
Regression of strategy returns against market returns provides:

- Alpha
- Beta
- Market-adjusted performance

---

# Project Structure

Stock-Return-Prediction
│
├── Stock_Return_Prediction.ipynb # Main analysis notebook
├── stock_return_prediction.py # Python version of the analysis
└── README.md # Project documentation


---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- yfinance

---

# Disclaimer

This project is intended for **educational and research purposes only**.

It does **not constitute financial advice or an investment recommendation**.

