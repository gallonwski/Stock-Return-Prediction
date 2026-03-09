import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

st.set_page_config(page_title="Stock Return Prediction", layout="wide")
st.title("Stock Return Prediction")
st.markdown("Analisi predittiva dei rendimenti azionari con Linear Regression e Random Forest.")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "JPM", "BAC", "GS", "MS",
    "XOM", "CVX",
    "JNJ", "PFE", "MRK", "ABBV",
    "PG", "KO", "PEP", "WMT", "COST",
    "HD", "LOW", "MCD", "SBUX",
    "CRM", "ORCL", "ADBE", "CSCO", "AMD"
]

with st.sidebar:
    st.header("Parametri")
    tickers_input = st.text_area(
        "Ticker separati da virgola",
        value=", ".join(DEFAULT_TICKERS),
        height=120,
    )
    start_date = st.date_input("Data inizio", value=pd.to_datetime("2016-01-01").date())
    end_date = st.date_input("Data fine", value=pd.to_datetime("2024-12-31").date())
    split_date = st.date_input("Data split train/test", value=pd.to_datetime("2023-01-01").date())
    run_button = st.button("Esegui analisi")


def performance_metrics(r: pd.Series) -> pd.Series:
    ann_factor = 252
    cumulative_return = (1 + r).prod() - 1
    cagr = (1 + cumulative_return) ** (ann_factor / len(r)) - 1 if len(r) > 0 else np.nan
    volatility = r.std() * np.sqrt(ann_factor)
    sharpe = (r.mean() / r.std()) * np.sqrt(ann_factor) if r.std() != 0 else np.nan

    cumulative = (1 + r).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if pd.notna(max_dd) and max_dd != 0 else np.nan

    return pd.Series({
        "CAGR": cagr,
        "CumReturn": cumulative_return,
        "AnnVol": volatility,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "Calmar": calmar,
    })


@st.cache_data(show_spinner=False)
def download_data(tickers, start_date, end_date):
    raw = yf.download(
        tickers,
        start=str(start_date),
        end=str(end_date),
        auto_adjust=True,
        progress=False,
    )
    return raw


if run_button:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if not tickers:
        st.error("Inserisci almeno un ticker.")
        st.stop()

    if start_date >= end_date:
        st.error("La data di inizio deve essere precedente alla data di fine.")
        st.stop()

    if split_date <= start_date or split_date >= end_date:
        st.error("La data di split deve essere compresa tra data inizio e data fine.")
        st.stop()

    with st.spinner("Scarico i dati e costruisco il modello..."):
        raw = download_data(tickers, start_date, end_date)

        if raw.empty:
            st.error("Nessun dato scaricato. Controlla ticker e intervallo date.")
            st.stop()

        if not {"Close", "Volume"}.issubset(set(raw.columns.get_level_values(0))):
            st.error("Il dataset scaricato non contiene le colonne attese Close e Volume.")
            st.stop()

        panel = raw[["Close", "Volume"]].copy()
        panel = panel.stack(level=1).reset_index()
        panel.columns = ["Date", "Ticker", "Close", "Volume"]
        panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        panel["Return_1d"] = panel.groupby("Ticker")["Close"].pct_change()
        panel["Target"] = panel.groupby("Ticker")["Return_1d"].shift(-1)
        panel["Ret_lag_1"] = panel.groupby("Ticker")["Return_1d"].shift(1)
        panel["Ret_lag_5"] = panel.groupby("Ticker")["Close"].pct_change(5)
        panel["Ret_lag_20"] = panel.groupby("Ticker")["Close"].pct_change(20)
        panel["Volatility_20"] = (
            panel.groupby("Ticker")["Return_1d"].rolling(20).std().reset_index(level=0, drop=True)
        )
        panel["MA_5"] = panel.groupby("Ticker")["Close"].rolling(5).mean().reset_index(level=0, drop=True)
        panel["MA_20"] = panel.groupby("Ticker")["Close"].rolling(20).mean().reset_index(level=0, drop=True)
        panel["Dist_MA_5"] = panel["Close"] / panel["MA_5"] - 1
        panel["Dist_MA_20"] = panel["Close"] / panel["MA_20"] - 1
        panel["Volume_Change_1"] = panel.groupby("Ticker")["Volume"].pct_change(1)
        panel["Volume_Change_5"] = panel.groupby("Ticker")["Volume"].pct_change(5)

        features = [
            "Ret_lag_1",
            "Ret_lag_5",
            "Ret_lag_20",
            "Volatility_20",
            "Dist_MA_5",
            "Dist_MA_20",
            "Volume_Change_1",
            "Volume_Change_5",
        ]

        model_data = panel[["Date", "Ticker", "Target"] + features].copy()
        model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        train_data = model_data[model_data["Date"] < pd.Timestamp(split_date)]
        test_data = model_data[model_data["Date"] >= pd.Timestamp(split_date)]

        if train_data.empty or test_data.empty:
            st.error("Train o test vuoti. Cambia l'intervallo date o la data di split.")
            st.stop()

        X_train = train_data[features]
        y_train = train_data["Target"]
        X_test = test_data[features]
        y_test = test_data["Target"]

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        mse_lr = mean_squared_error(y_test, pred_lr)

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, pred_rf)

        results = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "MSE": [mse_lr, mse_rf],
        })

        importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

        test_results = test_data.copy()
        test_results["Pred_LR"] = pred_lr
        test_results["Pred_RF"] = pred_rf
        test_results["Rank_Pct"] = test_results.groupby("Date")["Pred_RF"].rank(pct=True)
        test_results["Position"] = 0
        test_results.loc[test_results["Rank_Pct"] >= 0.7, "Position"] = 1
        test_results["Weighted_Return"] = test_results["Position"] * test_results["Target"]

        daily_strategy = test_results.groupby("Date")["Weighted_Return"].sum()
        n_positions = test_results.groupby("Date")["Position"].sum()
        daily_strategy = (daily_strategy / n_positions).replace([np.inf, -np.inf], 0).fillna(0)
        cumulative_returns = (1 + daily_strategy).cumprod()

        benchmark_returns = test_results.groupby("Date")["Target"].mean()
        benchmark_cum = (1 + benchmark_returns).cumprod()

        spy = yf.download("SPY", start=str(split_date), end=str(end_date), auto_adjust=True, progress=False)
        spy["Return"] = spy["Close"].pct_change()
        spy_returns = spy["Return"].dropna()
        spy_cum = (1 + spy_returns).cumprod()

        performance_df = pd.DataFrame({
            "Strategy": daily_strategy,
            "EqualWeighted": benchmark_returns.reindex(daily_strategy.index),
            "SPY": spy_returns.reindex(daily_strategy.index),
        }).dropna()
        performance_table = performance_df.apply(performance_metrics).T.round(3)

        ic_series = test_results.groupby("Date").apply(lambda x: x["Pred_RF"].corr(x["Target"])).dropna()
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std != 0 else np.nan

        market = spy_returns.reindex(daily_strategy.index).fillna(0)
        strategy = daily_strategy.reindex(market.index).fillna(0)
        capm_df = pd.DataFrame({"strategy": strategy, "market": market}).dropna()
        X_capm = sm.add_constant(capm_df["market"])
        capm_model = sm.OLS(capm_df["strategy"], X_capm).fit()
        alpha = capm_model.params["const"]
        beta = capm_model.params["market"]
        alpha_annual = alpha * 252

        excess_returns = capm_df["strategy"] - beta * capm_df["market"]
        alpha_cumulative = (1 + excess_returns).cumprod()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", len(tickers))
    col2.metric("Train obs", len(train_data))
    col3.metric("Test obs", len(test_data))
    col4.metric("RF MSE", f"{mse_rf:.8f}")

    st.subheader("Confronto modelli")
    st.dataframe(results, use_container_width=True)

    st.subheader("Feature importance")
    fig_imp = plt.figure(figsize=(8, 4))
    importances.plot(kind="bar")
    plt.title("Feature Importance. Random Forest")
    plt.ylabel("Importance")
    plt.grid(True)
    st.pyplot(fig_imp)
    plt.close(fig_imp)

    st.subheader("Predicted vs Actual")
    fig_scatter = plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_rf, alpha=0.3)
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.title("Random Forest. Predicted vs Actual")
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    st.pyplot(fig_scatter)
    plt.close(fig_scatter)

    st.subheader("Performance cumulata")
    fig_perf = plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label="ML Long-Only Strategy")
    plt.plot(benchmark_cum, label="Equal Weighted Benchmark", linestyle="--")
    plt.plot(spy_cum, label="Market Benchmark (SPY)", linestyle=":")
    plt.title("Strategy Performance vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_perf)
    plt.close(fig_perf)

    st.subheader("Metriche di performance")
    performance_table["CAPM Alpha"] = np.nan
    performance_table["CAPM Alpha Annualized"] = np.nan
    if "Strategy" in performance_table.index:
        performance_table.loc["Strategy", "CAPM Alpha"] = alpha
        performance_table.loc["Strategy", "CAPM Alpha Annualized"] = alpha_annual
    st.dataframe(performance_table, use_container_width=True)

    st.subheader("Information Coefficient")
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Mean IC", f"{ic_mean:.4f}")
    ic2.metric("IC Std", f"{ic_std:.4f}")
    ic3.metric("ICIR", f"{icir:.4f}")

    fig_ic = plt.figure(figsize=(8, 4))
    plt.hist(ic_series, bins=30)
    plt.title("Distribution of Information Coefficient")
    plt.xlabel("IC")
    plt.ylabel("Frequency")
    plt.grid(True)
    st.pyplot(fig_ic)
    plt.close(fig_ic)

    st.subheader("CAPM")
    capm1, capm2, capm3 = st.columns(3)
    capm1.metric("Daily Alpha", f"{alpha:.6f}")
    capm2.metric("Annualized Alpha", f"{alpha_annual:.2%}")
    capm3.metric("Market Beta", f"{beta:.4f}")

    fig_capm = plt.figure(figsize=(7, 6))
    plt.scatter(capm_df["market"], capm_df["strategy"], alpha=0.4)
    x_vals = capm_df["market"]
    y_vals = alpha + beta * x_vals
    plt.plot(x_vals, y_vals, color="red")
    plt.xlabel("Market Returns (SPY)")
    plt.ylabel("Strategy Returns")
    plt.title("CAPM Regression. Strategy vs Market")
    plt.grid(True)
    st.pyplot(fig_capm)
    plt.close(fig_capm)

    st.subheader("Performance aggiustata per il mercato")
    fig_alpha = plt.figure(figsize=(10, 6))
    strategy_cum = (1 + strategy).cumprod()
    market_cum = (1 + market).cumprod()
    plt.plot(strategy_cum, label="Strategy", linewidth=2)
    plt.plot(market_cum, label="Market (SPY)", linestyle="--")
    plt.plot(alpha_cumulative, label="Market-Adjusted Performance (Alpha)", linestyle=":")
    plt.title("Strategy vs Market and Alpha Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_alpha)
    plt.close(fig_alpha)

    st.subheader("Dati di output")
    st.dataframe(test_results.head(20), use_container_width=True)
else:
    st.info("Imposta i parametri nella sidebar e clicca 'Esegui analisi'.")
