# Stock Price Direction Predictor

Predicts whether a stock's closing price will go **up or down the next trading day**,
using engineered technical indicators (moving averages, RSI, volatility, lagged returns)
and a Random Forest classifier.

## How it works
1. Pulls real historical daily price data for a given ticker via `yfinance`
2. Engineers features from raw price/volume data: 5/10/20-day moving averages, 10-day
   volatility, 14-day RSI, volume change, and lagged daily returns
3. Splits data **chronologically** (not randomly — this matters for time series, so the
   model is never trained on future data) into 80% train / 20% test
4. Trains a Random Forest classifier to predict next-day direction (up/down)
5. Reports test accuracy, a classification report, and saves a chart (`results.png`)
   showing which predictions were correct and which features mattered most

## Setup

```bash
pip install -r requirements.txt
python predictor.py AAPL
```

You can swap `AAPL` for any valid ticker, and adjust the date range:

```bash
python predictor.py TSLA --start 2019-01-01 --end 2024-01-01
```

## Output
- Console: test accuracy, precision/recall, top predictive features
- `results.png`: chart of actual price with correct/incorrect direction calls marked,
  plus a feature importance bar chart

## Notes
- This predicts **direction** (up/down), not exact price — that's a much more realistic
  and honest framing for a short-term model like this.
- Stock direction prediction from price history alone is a genuinely hard problem —
  don't be surprised if accuracy lands somewhere in the 50-60% range. That's expected
  and worth explaining honestly (rather than overselling) if asked about this project.
- `sample_data.csv` is included as synthetic placeholder data, only used to confirm the
  code runs correctly end-to-end without needing a live internet connection.
