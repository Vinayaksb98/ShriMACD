import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

st.set_page_config(page_title="Nifty 500 MACD Scanner", layout="wide")

TF = {
    "1 Min": ("1m", "7d"),
    "15 Min": ("15m", "60d"),
    "1 Hour": ("1h", "730d"),
    "1 Day": ("1d", "2y"),
    "1 Week": ("1wk", "10y"),
}
URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

@st.cache_data(ttl=21600)
def universe():
    r = requests.get(URL, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    d = pd.read_csv(StringIO(r.text))
    col = "Symbol" if "Symbol" in d.columns else d.columns[0]
    out = []
    for raw in d[col].dropna().astype(str):
        s = raw.strip().upper()
        if not s or "DUMMY" in s or "TEST" in s:
            continue
        if not s.endswith(".NS"):
            s += ".NS"
        if s not in out:
            out.append(s)
    return sorted(out)

@st.cache_data(ttl=21600)
def market_cap(symbol):
    try:
        t = yf.Ticker(symbol)
        try:
            return float(t.fast_info["market_cap"]) / 1e7
        except Exception:
            return float((t.info or {}).get("marketCap", np.nan)) / 1e7
    except Exception:
        return np.nan

@st.cache_data(ttl=300)
def getdata(symbol, interval, period):
    try:
        d = yf.download(symbol, interval=interval, period=period,
                         auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        cols = ["Open","High","Low","Close","Volume"]
        if not all(c in d.columns for c in cols):
            return pd.DataFrame()
        return d[cols].dropna()
    except Exception:
        return pd.DataFrame()

def analyze(symbol, d, stoch_filter):
    if len(d) < 80:
        return None
    x = d.copy()
    c, h, l, v = [x[k].astype(float) for k in ["Close","High","Low","Volume"]]
    x["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    x["SIG"] = x.MACD.ewm(span=9, adjust=False).mean()
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    raw_k = 100*(c-ll)/(hh-ll).replace(0,np.nan)
    x["K"] = raw_k.rolling(3).mean()
    x["D"] = x.K.rolling(3).mean()
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    x["VR"] = v/v.rolling(20).mean().replace(0,np.nan)
    x = x.dropna()
    if len(x) < 80:
        return None

    cross = (x.MACD.shift(1) <= 0) & (x.MACD > 0)
    pos = np.flatnonzero(cross.to_numpy())
    if not len(pos):
        return None
    z = int(pos[-1])

    try:
        latest = pd.Timestamp(x.index[-1])
        ct = pd.Timestamp(x.index[z])
        if latest.tzinfo and not ct.tzinfo: ct = ct.tz_localize(latest.tz)
        if ct.tzinfo and not latest.tzinfo: ct = ct.tz_localize(None)
        if latest - ct > pd.Timedelta(days=7):
            return None
    except Exception:
        if len(x)-1-z > 7:
            return None

    if not (x.MACD.iloc[max(0,z-7):z] <= 0).any():
        return None

    k, dd = float(x.K.iloc[-1]), float(x.D.iloc[-1])
    stoch_status = "EXTENDED" if (k > 80 and dd > 80) else "NORMAL"
    # No hard rejection for K/D above 70.
    if stoch_filter == "Above 70" and not (k > 70 and dd > 70):
        return None
    if stoch_filter == "Below 70" and not (k < 70 and dd < 70):
        return None

    end = min(z+20, len(x)-1)
    peak = z + int(np.argmax(x.MACD.iloc[z:end+1].to_numpy()))
    if peak <= z:
        return None
    tail = x.iloc[peak:min(peak+15,len(x))]
    ret = peak + int(np.argmin(np.abs(tail.MACD.to_numpy())))

    macd, sig = float(x.MACD.iloc[-1]), float(x.SIG.iloc[-1])
    atr, vr, cur = float(x.ATR.iloc[-1]), float(x.VR.iloc[-1]), float(x.Close.iloc[-1])

    score = 45
    score += 15 if macd >= float(x.MACD.iloc[ret]) else 0
    score += 10 if k > dd else 0
    score += 10 if 45 <= k <= 80 and k >= dd else 0
    score += 10 if k > float(x.K.iloc[-3]) else 0
    score += 10 if vr >= 1 else 0

    trigger = max(float(x.High.iloc[peak:min(peak+6,len(x))].max()),
                  float(x.High.tail(20).max()))
    br = trigger + max(trigger*.0015, atr*.05)
    sl = min(float(x.Low.iloc[max(0,ret-3):min(len(x),ret+4)].min()),
             br-1.25*atr)
    if sl >= br: sl = br-1.25*atr

    levels = []
    H = x.High.to_numpy()
    for i in range(2,len(x)-2):
        if H[i] >= H[i-1] and H[i] >= H[i+1] and H[i] >= H[i-2] and H[i] >= H[i+2] and H[i] > br*1.002:
            levels.append(float(H[i]))
    levels = sorted(set(round(a,2) for a in levels))
    ts = levels[:3]
    while len(ts) < 3:
        ts.append(br + atr*(len(ts)+1))
    rr = (ts[0]-br)/(br-sl) if br > sl else np.nan
    recovered = macd >= float(x.MACD.iloc[ret])

    call = None
    if cur >= br and score >= 75 and macd > sig and recovered:
        call = "BUY CALL"
    elif cur >= br*.995 and score >= 65 and macd > sig and recovered:
        call = "BUY ON BREAKOUT"
    elif score >= 50:
        call = "WATCH"
    if not call:
        return None

    return {"Stock":symbol.replace(".NS",""),"CALL":call,"Score":int(score),
            "Current":round(cur,2),"Stoch K":round(k,2),"Stoch D":round(dd,2),"Stoch Status":stoch_status,
            "BUY/Breakout":round(br,2),"Stop Loss":round(sl,2),
            "T1":round(ts[0],2),"T2":round(ts[1],2),"T3":round(ts[2],2),
            "Next Resistance":round(levels[0],2) if levels else round(ts[0],2),
            "R:R T1":round(rr,2) if np.isfinite(rr) else None,"Volume":round(vr,2)}

st.title("🔥 Nifty 500 Fresh MACD Zero-Cross Scanner")
st.caption("Original strategy retained: MACD 60% + Stochastic 30% + Volume 10%. K/D above 70 is NOT a rejection.")
tf = st.selectbox("Timeframe", list(TF))
minimum = st.number_input("Minimum market cap (₹ Cr)",1000,1000000,10000,1000)
stoch_filter = st.selectbox("Stochastic Filter",["All","Above 70","Below 70"])

if st.button("🚀 SCAN NIFTY 500", type="primary"):
    try:
        syms = universe()
    except Exception as e:
        st.error(f"Could not load Nifty 500 universe: {e}")
        st.stop()

    elig=[]
    for s in syms:
        mc=market_cap(s)
        if np.isfinite(mc) and mc >= minimum:
            elig.append(s)

    rows=[]
    interval,period=TF[tf]
    bar=st.progress(0)
    for n,s in enumerate(elig):
        r=analyze(s,getdata(s,interval,period),stoch_filter)
        if r:
            r["TF"]=tf
            rows.append(r)
        if n%3==0: bar.progress((n+1)/max(1,len(elig)))
    bar.empty()

    q=pd.DataFrame(rows)
    a,b,c,d,e=st.columns(5)
    a.metric("Nifty 500 scanned",len(syms))
    b.metric("Market-cap eligible",len(elig))
    c.metric("BUY CALL",int((q.CALL=="BUY CALL").sum()) if not q.empty else 0)
    d.metric("BUY ON BREAKOUT",int((q.CALL=="BUY ON BREAKOUT").sum()) if not q.empty else 0)
    e.metric("WATCH",int((q.CALL=="WATCH").sum()) if not q.empty else 0)

    cols=["Rank","Stock","TF","CALL","Score","Current","Stoch K","Stoch D","Stoch Status",
          "BUY/Breakout","Stop Loss","T1","T2","T3","Next Resistance","R:R T1","Volume"]
    if q.empty:
        st.warning("No qualifying setup found for the selected filters.")
    else:
        q=q.sort_values(["Score","Stock"],ascending=[False,True]).reset_index(drop=True)
        q.insert(0,"Rank",np.arange(1,len(q)+1))
        tabs=st.tabs(["🔥 BUY CALL","🟡 BUY ON BREAKOUT","👀 WATCH","📊 ALL QUALIFIED"])
        for tab,cat in zip(tabs,["BUY CALL","BUY ON BREAKOUT","WATCH",None]):
            with tab:
                view=q if cat is None else q[q.CALL==cat]
                st.dataframe(view[cols],use_container_width=True,hide_index=True)
else:
    st.info("Choose timeframe, market cap and optional Stochastic filter, then click SCAN NIFTY 500.")
