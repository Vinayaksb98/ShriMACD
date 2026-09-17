import streamlit as st
import pandas as pd, numpy as np, yfinance as yf, requests
from io import StringIO

st.set_page_config(page_title="Nifty 500 Fresh MACD Scanner",layout="wide")
URL="https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
TF={"1 Min":("1m","7d"),"15 Min":("15m","60d"),"1 Hour":("1h","730d"),"1 Day":("1d","2y")}

@st.cache_data(ttl=21600)
def universe():
    r=requests.get(URL,headers={"User-Agent":"Mozilla/5.0"},timeout=20); r.raise_for_status()
    d=pd.read_csv(StringIO(r.text)); c="Symbol" if "Symbol" in d.columns else d.columns[0]
    return sorted(set(str(x).strip().upper()+".NS" for x in d[c].dropna()))

@st.cache_data(ttl=21600)
def cap(s):
    try:
        t=yf.Ticker(s)
        try:return float(t.fast_info["market_cap"])/1e7
        except:return float((t.info or {}).get("marketCap",np.nan))/1e7
    except:return np.nan

@st.cache_data(ttl=300)
def getdata(s,i,period):
    try:
        d=yf.download(s,interval=i,period=period,auto_adjust=False,progress=False,threads=False)
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        cols=["Open","High","Low","Close","Volume"]
        return d[cols].dropna() if all(c in d.columns for c in cols) else pd.DataFrame()
    except:return pd.DataFrame()

def analyze(s,d):
    if len(d)<80:return None
    x=d.copy(); c,h,l,v=x.Close,x.High,x.Low,x.Volume
    x["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    x["SIG"]=x.MACD.ewm(span=9,adjust=False).mean()
    ll,hh=l.rolling(14).min(),h.rolling(14).max()
    x["K"]=(100*(c-ll)/(hh-ll).replace(0,np.nan)).rolling(3).mean()
    x["D"]=x.K.rolling(3).mean()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14).mean(); x["VR"]=v/v.rolling(20).mean().replace(0,np.nan)
    x=x.dropna()
    cross=(x.MACD.shift(1)<=0)&(x.MACD>0); pos=np.flatnonzero(cross.to_numpy())
    if not len(pos):return None
    z=int(pos[-1])
    # Fresh crossover: latest cross within 7 bars and MACD was below zero before it.
    if len(x)-1-z>7 or not (x.MACD.iloc[max(0,z-7):z]<=0).any():return None
    post=x.iloc[z:min(z+3,len(x))]
    if ((post.K>70)&(post.D>70)).any():return None
    end=min(z+20,len(x)-1); peak=z+int(np.argmax(x.MACD.iloc[z:end+1].to_numpy()))
    if peak<=z:return None
    tail=x.iloc[peak:min(peak+15,len(x))]; ret=peak+int(np.argmin(np.abs(tail.MACD.to_numpy())))
    macd,sig=float(x.MACD.iloc[-1]),float(x.SIG.iloc[-1]); k,dd=float(x.K.iloc[-1]),float(x.D.iloc[-1])
    atr,vr,cur=float(x.ATR.iloc[-1]),float(x.VR.iloc[-1]),float(x.Close.iloc[-1])
    score=45+(15 if macd>=float(x.MACD.iloc[ret]) else 0)+(10 if k>dd else 0)+(10 if 45<=k<=80 and k>=dd else 0)+(10 if k>float(x.K.iloc[-3]) else 0)+(10 if vr>=1 else 0)
    trigger=max(float(x.High.iloc[peak:min(peak+6,len(x))].max()),float(x.High.tail(20).max()))
    br=trigger+max(trigger*.0015,atr*.05)
    sl=min(float(x.Low.iloc[max(0,ret-3):min(len(x),ret+4)].min()),br-1.25*atr)
    if sl>=br:sl=br-1.25*atr
    levels=[]; H=x.High.to_numpy()
    for i in range(2,len(x)-2):
        if H[i]>=H[i-1] and H[i]>=H[i+1] and H[i]>=H[i-2] and H[i]>=H[i+2] and H[i]>br*1.002:levels.append(float(H[i]))
    levels=sorted(set(round(a,2) for a in levels)); ts=levels[:3]
    while len(ts)<3:ts.append(br+atr*(len(ts)+1))
    rr=(ts[0]-br)/(br-sl) if br>sl else np.nan
    recovered=macd>=float(x.MACD.iloc[ret])
    call="BUY CALL" if cur>=br and score>=75 and macd>sig and recovered else ("BUY ON BREAKOUT" if cur>=br*.995 and score>=65 and macd>sig and recovered else ("WATCH" if score>=50 else None))
    if not call:return None
    return {"Stock":s.replace(".NS",""),"CALL":call,"Score":int(score),"Current":round(cur,2),"BUY/Breakout":round(br,2),"Stop Loss":round(sl,2),"T1":round(ts[0],2),"T2":round(ts[1],2),"T3":round(ts[2],2),"Next Resistance":round(levels[0],2) if levels else round(ts[0],2),"R:R T1":round(rr,2) if np.isfinite(rr) else None,"Volume":round(vr,2)}

st.title("🔥 Nifty 500 Fresh MACD Zero-Cross Scanner")
st.caption("Fresh = MACD was below zero during the previous 7 bars and has just crossed above zero.")
tf=st.sidebar.selectbox("Timeframe",list(TF)); minimum=st.sidebar.number_input("Minimum market cap (₹ Cr)",1000,1000000,10000,1000)
if st.sidebar.button("🚀 SCAN NIFTY 500",type="primary"):
    syms=universe(); caps={s:cap(s) for s in syms}; elig=[s for s in syms if np.isfinite(caps[s]) and caps[s]>=minimum]
    rows=[]; bar=st.progress(0); interval,period=TF[tf]
    for n,s in enumerate(elig):
        r=analyze(s,getdata(s,interval,period))
        if r:r["TF"]=tf; rows.append(r)
        if n%3==0:bar.progress((n+1)/max(1,len(elig)))
    q=pd.DataFrame(rows)
    a,b,c,d=st.columns(4); a.metric("Nifty 500 scanned",len(syms)); b.metric("Market-cap eligible",len(elig)); c.metric("BUY CALL",int((q.CALL=="BUY CALL").sum()) if not q.empty else 0); d.metric("BUY ON BREAKOUT",int((q.CALL=="BUY ON BREAKOUT").sum()) if not q.empty else 0)
    cols=["Stock","TF","CALL","Score","Current","BUY/Breakout","Stop Loss","T1","T2","T3","Next Resistance","R:R T1","Volume"]
    if q.empty:st.warning("No fresh qualifying setup found.")
    else:
        for tab,cat in zip(st.tabs(["🔥 BUY CALL","🟡 BUY ON BREAKOUT","👀 WATCH","📊 ALL QUALIFIED"]),["BUY CALL","BUY ON BREAKOUT","WATCH",None]):
            with tab:
                st.dataframe(q if cat is None else q[q.CALL==cat][cols],use_container_width=True,hide_index=True)
else: st.info("Choose a timeframe and click SCAN NIFTY 500.")
