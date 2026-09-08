
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="MACD Zero-Line Scanner", layout="wide")
st.title("🔥 MACD Zero-Line Crossover Scanner")
st.caption("Nifty 500 • Target Market Cap Filter ₹10,000 Cr+ • Daily • False Signal Reduction")

@st.cache_data(ttl=3600)
def nifty500_symbols():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    df = pd.read_csv(url)
    symbol_col = next(c for c in df.columns if "symbol" in c.lower())
    return [f"{x}.NS" for x in df[symbol_col].dropna().astype(str)]

def calc_indicators(df):
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    out = pd.DataFrame(index=df.index)
    out["Close"] = c
    out["Volume"] = v
    out["EMA20"] = ema20
    out["EMA50"] = ema50
    out["MACD"] = macd
    out["Signal"] = signal
    out["Hist"] = hist
    out["Vol20"] = v.rolling(20).mean()
    out["Support"] = df["Low"].rolling(20).min()
    out["Resistance"] = df["High"].rolling(20).max()
    return out.dropna()

def evaluate(ind):
    if len(ind) < 4:
        return None

    a,b,c,d = ind.iloc[-1], ind.iloc[-2], ind.iloc[-3], ind.iloc[-4]
    close = float(a.Close)
    macd = float(a.MACD)
    prev_macd = float(b.MACD)
    signal = float(a.Signal)

    cross = prev_macd < 0 <= macd
    rising3 = macd > float(b.MACD) > float(c.MACD)
    hist_rising = float(a.Hist) > float(b.Hist) > float(c.Hist)
    above_signal = macd > signal
    ema_trend = close > float(a.EMA20) and float(a.EMA20) >= float(a.EMA50)
    volume_ratio = float(a.Volume / a.Vol20) if float(a.Vol20) else 0
    volume_ok = volume_ratio >= 1.3

    # "Ready" means below zero, rising for 3 candles and close to zero.
    recent_abs = max(abs(float(c.MACD)), 1e-6)
    ready = (prev_macd < 0 and macd < 0 and rising3 and
             abs(macd) <= max(recent_abs * 0.70, close * 0.0008))

    support = float(a.Support)
    resistance = float(a.Resistance)
    room = ((resistance / close) - 1) * 100 if resistance > close else 0.0

    score = 0
    score += 30 if cross else (22 if ready else 0)
    score += 15 if rising3 else 0
    score += 10 if above_signal else 0
    score += 10 if hist_rising else 0
    score += 15 if ema_trend else 0
    score += 10 if volume_ok else 0
    score += 10 if room >= 3 else 0
    score = min(score, 100)

    if not (cross or ready):
        return None

    if score >= 85 and room >= 3:
        status = "🔥 HIGH-CONVICTION"
    elif score >= 75:
        status = "🟢 CONFIRMED"
    elif score >= 65:
        status = "🟡 READY"
    else:
        status = "⚪ WATCH"

    return {
        "CMP": close,
        "MACD": macd,
        "Signal": signal,
        "Histogram": float(a.Hist),
        "Support": support,
        "Strong Resistance": resistance,
        "Room %": room,
        "Volume x": volume_ratio,
        "Score": score,
        "Status": status,
        "MACD Cross": cross,
        "Ready": ready,
    }

def market_cap_ok(ticker, min_cr=10000):
    # Market cap is fetched from Yahoo Finance when available.
    try:
        info = yf.Ticker(ticker).fast_info
        cap = getattr(info, "market_cap", None)
        if cap is None:
            return False, None
        cap_cr = float(cap) / 1e7
        return cap_cr >= min_cr, cap_cr
    except Exception:
        return False, None

def scan(symbols, strict_market_cap=True, min_cap=10000, progress=None):
    rows = []
    total = len(symbols)

    for i, ticker in enumerate(symbols, 1):
        try:
            if progress:
                progress.progress(i / total, text=f"Scanning {i}/{total}: {ticker}")

            if strict_market_cap:
                ok, cap = market_cap_ok(ticker, min_cap)
                if not ok:
                    continue
            else:
                cap = np.nan

            df = yf.download(
                ticker, period="9mo", interval="1d",
                auto_adjust=False, progress=False, threads=False
            )
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            ind = calc_indicators(df)
            result = evaluate(ind)
            if result is None:
                continue

            result["Stock"] = ticker.replace(".NS", "")
            result["Market Cap ₹ Cr"] = cap
            rows.append(result)
        except Exception:
            continue

    return pd.DataFrame(rows)

st.sidebar.header("Scanner Filters")
strict_cap = st.sidebar.checkbox("Strict Market Cap > ₹10,000 Cr", value=True)
min_score = st.sidebar.slider("Minimum Quality Score", 50, 95, 65)
only_confirmed = st.sidebar.checkbox("Only Confirmed / High-Conviction", value=False)
run = st.sidebar.button("🚀 RUN FULL SCAN", type="primary")

st.markdown("""
### False-signal filters
MACD zero-line setup → 3-candle MACD slope → signal confirmation → histogram improvement →
EMA20/EMA50 trend → volume confirmation → support/resistance room.
""")

if run:
    try:
        symbols = nifty500_symbols()
    except Exception as e:
        st.error(f"Could not load Nifty 500 constituents: {e}")
        st.stop()

    progress = st.progress(0, text="Starting scan...")
    with st.spinner("Downloading Yahoo Finance data and calculating signals..."):
        result = scan(symbols, strict_cap, 10000, progress)
    progress.empty()

    if result.empty:
        st.warning("No stocks passed the current filters. Lower the minimum score or disable only-confirmed.")
        st.stop()

    result = result[result["Score"] >= min_score]
    if only_confirmed:
        result = result[result["Score"] >= 75]

    result = result.sort_values(["Score", "Room %"], ascending=[False, False]).reset_index(drop=True)
    result.index += 1
    result.insert(0, "Rank", result.index)

    st.success(f"Found {len(result)} eligible stocks")

    cols = ["Rank","Stock","CMP","Market Cap ₹ Cr","MACD","Signal","Histogram",
            "Support","Strong Resistance","Room %","Volume x","Score","Status"]
    st.dataframe(
        result[cols].style.format({
            "CMP":"₹{:.2f}",
            "Market Cap ₹ Cr":"{:,.0f}",
            "MACD":"{:.3f}",
            "Signal":"{:.3f}",
            "Histogram":"{:.3f}",
            "Support":"₹{:.2f}",
            "Strong Resistance":"₹{:.2f}",
            "Room %":"{:.2f}%",
            "Volume x":"{:.2f}x",
            "Score":"{:.0f}",
        }),
        use_container_width=True,
        height=700,
    )

    st.download_button(
        "⬇ Download Results CSV",
        result.to_csv(index=False),
        "macd_zero_line_results.csv",
        "text/csv",
    )
else:
    st.info("Click RUN FULL SCAN. The scan may take several minutes because it checks Yahoo Finance data for the Nifty 500 universe.")
