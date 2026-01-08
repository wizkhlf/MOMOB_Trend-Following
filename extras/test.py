import yfinance as yf

tickers = ["AI.PA", "MC.PA", "OR.PA", "^FCHI"]  # CAC40 index en bonus

for t in tickers:
    try:
        df = yf.download(t, period="1mo", progress=False)
        print(t, "rows:", len(df), "cols:", df.columns.tolist() if len(df) else "EMPTY")
    except Exception as e:
        print(t, "ERROR:", repr(e))
