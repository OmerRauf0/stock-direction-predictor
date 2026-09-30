"""
Stock Price Direction Predictor
--------------------------------
Predicts whether a stock's closing price will go UP or DOWN the next trading day,
using engineered technical indicators and a Random Forest classifier.

Usage:
    python predictor.py AAPL
    python predictor.py AAPL --start 2020-01-01 --end 2024-01-01

Requires internet access (uses yfinance to pull real historical data).
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data for a given ticker."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol and date range.")
    df = df.reset_index()
    # yfinance sometimes returns MultiIndex columns for a single ticker — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical indicators used as model inputs."""
    df = df.copy()

    # Moving averages
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # Daily return and volatility
    df["Return"] = df["Close"].pct_change()
    df["Volatility10"] = df["Return"].rolling(10).std()

    # Relative Strength Index (14-day)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    # Volume trend
    df["VolChange"] = df["Volume"].pct_change()

    # Lagged returns (yesterday, 2 days ago, 3 days ago)
    for lag in (1, 2, 3):
        df[f"Return_lag{lag}"] = df["Return"].shift(lag)

    # Target: 1 if next day's close is higher than today's, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "MA5", "MA10", "MA20", "Return", "Volatility10",
    "RSI14", "VolChange", "Return_lag1", "Return_lag2", "Return_lag3",
]


def train_and_evaluate(df: pd.DataFrame):
    """Time-ordered train/test split (no shuffling — this is a time series)."""
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[FEATURE_COLS], train["Target"]
    X_test, y_test = test[FEATURE_COLS], test["Target"]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nTest accuracy: {acc:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["Down", "Up"]))

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Top features:")
    print(importances.head(5).to_string())

    return model, test, preds, acc, importances


def plot_results(ticker, test, preds, importances):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(test["Date"], test["Close"], label="Actual Close", color="steelblue")
    correct = preds == test["Target"].values
    axes[0].scatter(test["Date"][correct], test["Close"][correct], color="green", s=12, label="Correct direction call")
    axes[0].scatter(test["Date"][~correct], test["Close"][~correct], color="red", s=12, label="Wrong direction call")
    axes[0].set_title(f"{ticker}: Predicted direction vs. actual price (test period)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    importances.head(8).sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Feature importance")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved chart to results.png")


def main():
    parser = argparse.ArgumentParser(description="Predict next-day stock price direction.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")
    args = parser.parse_args()

    print(f"Fetching {args.ticker} data from {args.start} to {args.end}...")
    raw = fetch_data(args.ticker, args.start, args.end)

    print("Engineering features...")
    featured = engineer_features(raw)

    print(f"Training on {len(featured)} rows of data...")
    model, test, preds, acc, importances = train_and_evaluate(featured)

    plot_results(args.ticker, test, preds, importances)


if __name__ == "__main__":
    main()
