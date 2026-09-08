# MACD Zero-Line Nifty 500 Scanner

## Strategy
Universe:
- Nifty 500
- Strict market-cap filter above ₹10,000 crore (Yahoo Finance fast_info when available)

Signal:
- MACD crossed above zero today, OR
- MACD is below zero but rising for 3 candles and close to zero

False-signal filters:
- MACD rising for 3 candles
- MACD above signal line
- Histogram rising
- Price above EMA20
- EMA20 >= EMA50
- Volume >= 1.3x 20-day average
- At least 3% room to calculated resistance for maximum score

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Streamlit Cloud
Upload these files to GitHub and deploy app.py.

Note:
A full Nifty 500 scan can take several minutes because Yahoo Finance data is downloaded for many symbols.
