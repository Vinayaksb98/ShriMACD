import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="MACD Stochastic Breakout", layout="wide")

TF = {
    "1 Min": ("1m", "7d"),
    "15 Min": ("15m", "60d"),
    "1 Hour": ("1h", "730d"),
    "1 Day": ("1d", "2y"),
}

def indicators(x):
    d=x.copy()
    d["EMA20"]=d["Close"].ewm(span=20,adjust=False).mean()
    d["EMA50"]=d["Close"].ewm(span=50,adjust=False).mean()
    f=d["Close"].ewm(span=12,adjust=False).mean()
    s=d["Close"].ewm(span=26,adjust=False).mean()
    d["MACD"]=f-s
    d["SIGNAL"]=d["MACD"].ewm(span=9,adjust=False).mean()
    lo=d["Low"].rolling(14).min(); hi=d["High"].rolling(14).max()
    rk=100*(d["Close"]-lo)/(hi-lo).replace(0,np.nan)
    d["K"]=rk.rolling(3).mean()
    d["D"]=d["K"].rolling(3).mean()
    d["V20"]=d["Volume"].rolling(20).mean()
    d["VR"]=d["Volume"]/d["V20"]
    pc=d["Close"].shift(1)
    tr=pd.concat([(d["High"]-d["Low"]),(d["High"]-pc).abs(),(d["Low"]-pc).abs()],axis=1).max(axis=1)
    d["ATR"]=tr.rolling(14).mean()
    return d.dropna()

def resistances(d, price):
    x=d.iloc[:-1]
    swings=x[(x["High"]>=x["High"].shift(1))&(x["High"]>=x["High"].shift(-1))]["High"].dropna()
    levels=[float(v) for v in swings if v>price*1.003]
    for n in (20,50,100):
        if len(x)>=n:
            v=float(x["High"].tail(n).max())
            if v>price*1.003: levels.append(v)
    return sorted(set(round(v,2) for v in levels))

def analyze(raw, buf, slbuf):
    d=indicators(raw)
    if len(d)<100: return {"status":"Insufficient data"}
    cross=((d["MACD"].shift(1)<=0)&(d["MACD"]>0))
    pos=np.flatnonzero(cross.values)
    if not len(pos): return {"status":"No bullish MACD zero-line cross"}
    cp=pos[-1]
    if cp+1>=len(d): return {"status":"Waiting for next candle"}
    imm=d.iloc[cp+1]
    if imm["K"]>70 and imm["D"]>70:
        return {"status":"REJECTED","reason":"Both Stochastic %K and %D are above 70 immediately after the MACD zero-line cross.","K":float(imm["K"]),"D":float(imm["D"])}
    wave=d.iloc[cp:max(cp+5,len(d)-2)]
    peak=wave["MACD"].idxmax(); pi=d.index.get_loc(peak)
    h=float(d.loc[peak,"MACD"]); p=float(d.loc[peak,"High"])
    if h<=0: return {"status":"No positive MACD first wave"}
    after=d.iloc[pi:]
    ri=after["MACD"].idxmin()
    rm=float(d.loc[ri,"MACD"])
    rlow=float(d.loc[peak:ri,"Low"].min())
    ret=max(0,min(150,(h-rm)/h*100))
    m0,m1,m2=d["MACD"].iloc[-1],d["MACD"].iloc[-2],d["MACD"].iloc[-3]
    k0,k1,k2=d["K"].iloc[-1],d["K"].iloc[-2],d["K"].iloc[-3]
    score=15
    score+=15 if h/max(abs(float(d["MACD"].iloc[cp:].max())),1e-9)>=.6 else 10 if h/max(abs(float(d["MACD"].iloc[cp:].max())),1e-9)>=.35 else 5
    score+=15 if 70<=ret<=105 and rm>=-.15*h else 8 if 50<=ret<70 else 3 if ret>=30 else 0
    score+=15 if m0>m1>m2 else 10 if m0>m1 else 0
    score+=10 if k0>d["D"].iloc[-1] else 0
    kmin=float(d["K"].iloc[max(0,pi):].min())
    score+=10 if kmin<30 and k0<85 else 8 if kmin<50 and k0<85 else 5 if k0<85 else 0
    score+=10 if k0>k1>k2 else 0
    vr=float(d["VR"].iloc[-1])
    score+=10 if vr>1.5 else 8 if vr>=1.2 else 5 if vr>=1 else 3 if vr>=.8 else 0
    price=float(d["Close"].iloc[-1])
    breakout=max(p,float(d["High"].iloc[-21:-1].max()))
    entry=breakout*(1+buf)
    atr=float(d["ATR"].iloc[-1])
    struct=rlow*(1-slbuf)
    stop=min(struct,entry-1.25*atr,entry*.97)
    risk=entry-stop
    if risk<=0:return {"status":"Invalid risk"}
    rs=resistances(d,entry)
    r1=next((v for v in rs if v>entry*1.003),None)
    r2=next((v for v in rs if r1 and v>r1*1.003),None)
    t1=r1 if r1 else entry+risk
    t2=r2 if r2 else entry+2*risk
    t3=(r2+2*risk) if r2 else entry+3*risk
    trend=10 if price>d["EMA20"].iloc[-1]>d["EMA50"].iloc[-1] else 6 if price>d["EMA20"].iloc[-1] else 0
    room=((t2/entry)-1)*100
    pot=min(100,score+trend+(10 if room>=10 else 7 if room>=6 else 4 if room>=3 else 0))
    reach=round(min(95,max(20,45+(pot-70)*1.1+min(10,max(0,(vr-1)*10))),1)
    ready=ret>=50 and m0>m1 and k0>d["D"].iloc[-1] and k0>k1 and score>=70
    action="BUY CALL" if price>breakout and score>=75 else "BUY ON BREAKOUT" if ready else "WATCH" if ret>=50 else "NO TRADE"
    return dict(status=action,score=int(score),potential=int(pot),reach=reach,price=price,entry=entry,stop=stop,t1=t1,t2=t2,t3=t3,risk=risk,room=room,r1=r1,r2=r2,vr=vr,macd=m0,k=k0,d=d["D"].iloc[-1],immk=imm["K"],immd=imm["D"])

st.title("MACD + Stochastic + Volume Breakout Strategy")
st.caption("MACD 60% • Stochastic 30% • Volume 10% • Resistance/ATR targets")

ticker=st.sidebar.text_input("NSE Ticker","CDSL").upper().strip()
tf=st.sidebar.selectbox("Time Frame",list(TF))
buf=st.sidebar.number_input("Breakout buffer %",0.05,1.0,0.20,0.05)/100
slbuf=st.sidebar.number_input("SL structure buffer %",0.1,2.0,0.50,0.1)/100

if st.sidebar.button("Analyze",type="primary"):
    interval,period=TF[tf]
    sym=ticker if ticker.endswith(".NS") else ticker+".NS"
    raw=yf.download(sym,interval=interval,period=period,auto_adjust=False,progress=False)
    if raw.empty: st.error("No Yahoo Finance data found."); st.stop()
    if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
    raw=raw[["Open","High","Low","Close","Volume"]].dropna()
    r=analyze(raw,buf,slbuf)
    if r["status"]=="REJECTED":
        st.error("❌ REJECTED: "+r["reason"])
        st.write({"Immediate %K":round(r["K"],2),"Immediate %D":round(r["D"],2)})
        st.stop()
    if r["status"] in ["Insufficient data","No bullish MACD zero-line cross","Waiting for next candle","No positive MACD first wave","Invalid risk"]:
        st.warning(r["status"]); st.stop()
    a,b,c,d,e=st.columns(5)
    a.metric("CALL",r["status"]); b.metric("Strategy Score",f'{r["score"]}/100'); c.metric("Potential",f'{r["potential"]}/100'); d.metric("Reachability",f'{r["reach"]}%*'); e.metric("Volume",f'{r["vr"]:.2f}x')
    st.subheader("BUY / STOP LOSS / TARGETS")
    st.dataframe(pd.DataFrame([{"BUY / Breakout":round(r["entry"],2),"STOP LOSS":round(r["stop"],2),"TARGET 1":round(r["t1"],2),"TARGET 2":round(r["t2"],2),"TARGET 3":round(r["t3"],2),"Risk/Share":round(r["risk"],2)}]),use_container_width=True,hide_index=True)
    st.subheader("Next Resistance & Potential")
    st.write(f"Next resistance: {r['r1'] if r['r1'] else 'not found — R1 fallback used'} | Second resistance: {r['r2'] if r['r2'] else 'not found — R2 fallback used'} | Room to T2: {r['room']:.2f}%")
    st.info("*Reachability is a technical ranking score, not a guaranteed probability or return.")
    st.subheader("Indicator Confirmation")
    st.dataframe(pd.DataFrame([{"MACD":round(r["macd"],4),"Stoch %K":round(r["k"],2),"Stoch %D":round(r["d"],2),"Immediate K":round(r["immk"],2),"Immediate D":round(r["immd"],2),"Volume Ratio":round(r["vr"],2)}]),use_container_width=True,hide_index=True)
else:
    st.write("Select an NSE ticker and timeframe, then click Analyze.")
