#!/usr/bin/env python3
"""
Life on the Hedge Fund — build_dashboard.py
Institutional Portfolio Analytics Terminal
Trinity College Dublin · Investment Analysis · Academic Project

Single source of truth:
  1. Load holdings.csv
  2. Download prices via yfinance
  3. Compute all metrics in Python
  4. Generate docs/index.html  (fully static HTML, no client-side data fetching)
  5. Write data/dashboard_snapshot.json
"""
from __future__ import annotations

import json
import math
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings("ignore")


# ================================================================
# NUMPY-SAFE JSON ENCODER
# Fixes: TypeError: Object of type ndarray is not JSON serializable
# ================================================================
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_NpEncoder)


# ================================================================
# PATHS
# ================================================================
ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DATA = ROOT / "data"
DOCS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

OUT_HTML = DOCS / "index.html"
OUT_JSON = DATA / "dashboard_snapshot.json"

# ================================================================
# CONFIG
# ================================================================
CFG = {
    "portfolio_name": "Life on the Hedge Fund",
    "school":         "Trinity College Dublin",
    "school_tag":     "#1 in Ireland",
    "course":         "Investment Analysis · Academic Portfolio",
    "benchmark":      "QQQ",
    "bench2":         "SPY",
    "inception":      "2025-03-06",
    "rf":             0.0450,
    "af":             252,
    "roll_w":         30,
    "mc_seed":        42,
    "mc_paths":       600,
    "news_n":         10,
}

# ================================================================
# COLOUR PALETTE
# ================================================================
C = {
    "bg":     "#06080d", "panel":  "#0b1018", "panel2": "#111826",
    "card":   "#0d1420", "card2":  "#111b2a",
    "border": "#1d2a3f", "border2":"#2a3d5a",
    "text":   "#dde7f3", "muted":  "#91a4bf", "dim":    "#5d708d",
    "grid":   "#172232",
    "green":  "#21d07a", "red":    "#f45b69", "amber":  "#ffbe55",
    "blue":   "#4d8dff", "cyan":   "#45d7ff", "purple": "#b085ff",
}

SC = {
    "AI / Semiconductors":  "#4d8dff",
    "AI / Tech Platform":   "#6ea0ff",
    "AI / Defence Tech":    "#7d7cf7",
    "AI / AdTech":          "#915df8",
    "AI / Voice":           "#bf8dff",
    "Defense / Aerospace":  "#6d78d6",
    "Space Economy":        "#8b59d3",
    "Energy Transition":    "#ffbe55",
    "Crypto Infrastructure":"#ff7d72",
    "Bitcoin Mining":       "#db4b42",
    "Fintech / Retail":     "#21d07a",
    "Mobility / Platform":  "#45d7ff",
    "Social / AI Data":     "#4fd3c4",
}

BUCKET_C = {
    "CORE":        C["blue"],
    "GROWTH":      C["green"],
    "SPECULATIVE": C["amber"],
}

SCENARIOS = [
    {"name": "Broad market −10%",   "type": "benchmark", "shock": -0.10, "desc": "10% QQQ drawdown mapped through beta."},
    {"name": "Growth de-rating",    "type": "bucket",    "bucket": "GROWTH",      "shock": -0.18, "desc": "Duration/growth multiple compression."},
    {"name": "Speculative risk-off","type": "bucket",    "bucket": "SPECULATIVE", "shock": -0.25, "desc": "High-beta / crypto / small-cap unwind."},
    {"name": "AI multiple compress","type": "theme_kw",  "kws": ["AI"],           "shock": -0.17, "desc": "AI narrative repricing."},
    {"name": "Rates shock",         "type": "custom",    "mapping": {"GROWTH": -0.12, "SPECULATIVE": -0.16, "CORE": -0.06}, "desc": "Higher real yields pressure long-duration."},
    {"name": "Crypto crash",        "type": "tickers",   "tickers": ["COIN","MARA","HOOD"], "shock": -0.28, "desc": "Crypto sleeve reprices sharply."},
]


# ================================================================
# HELPERS
# ================================================================
def _esc(s: str) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def _fc(x: float, d: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{d}f}"

def _fp(x: float, d: int = 2) -> str:
    if pd.isna(x): return "—"
    return f"{x:+.{d}f}%"

def _fx(x: float, d: int = 2) -> str:
    if pd.isna(x): return "—"
    return f"{x:.{d}f}x"

def _col(x: float) -> str:
    return C["green"] if x >= 0 else C["red"]

def _pct_col(x: float) -> str:
    return "green" if x >= 0 else "red"

def _colors(arr, pos_col: str = None, neg_col: str = None) -> list:
    """Convert np.where result to a Python list of colour strings."""
    pc = pos_col or C["green"]
    nc = neg_col or C["red"]
    return [pc if v >= 0 else nc for v in arr]


# ================================================================
# DATA INGESTION
# ================================================================
def load_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"ticker","name","quantity","buy_price","sector","theme","risk_bucket","inception_date"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"holdings.csv missing columns: {sorted(missing)}")
    df["ticker"]     = df["ticker"].str.upper().str.strip()
    df["quantity"]   = df["quantity"].astype(float)
    df["buy_price"]  = df["buy_price"].astype(float)
    df["cost_basis"] = df["quantity"] * df["buy_price"]
    return df


def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    end_buf = (pd.Timestamp(end) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers=tickers, start=start, end=end_buf,
        auto_adjust=True, progress=False, threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data. Check connection.")

    # Normalise MultiIndex → flat ticker columns
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = [str(v) for v in raw.columns.get_level_values(0)]
        lv1 = [str(v) for v in raw.columns.get_level_values(1)]
        frames = []
        if "Close" in lv0:
            for t in tickers:
                col = ("Close", t)
                if col in raw.columns:
                    frames.append(raw[col].rename(t))
                else:
                    m = [c for c in raw.columns if c[0]=="Close" and str(c[1]).upper()==t.upper()]
                    if m: frames.append(raw[m[0]].rename(t))
        elif "Close" in lv1:
            for t in tickers:
                col = (t, "Close")
                if col in raw.columns:
                    frames.append(raw[col].rename(t))
                else:
                    m = [c for c in raw.columns if str(c[0]).upper()==t.upper() and c[1]=="Close"]
                    if m: frames.append(raw[m[0]].rename(t))
        else:
            for t in tickers:
                for col in raw.columns:
                    if "close" in str(col).lower() and t.upper() in str(col).upper():
                        frames.append(raw[col].rename(t)); break
        if not frames:
            if "Close" in raw.columns:
                frames = [raw["Close"].rename(tickers[0])]
            else:
                raise RuntimeError("Cannot extract Close prices from yfinance response.")
        close = pd.concat(frames, axis=1)
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]}) if "Close" in raw.columns else raw.copy()

    close.index = pd.to_datetime(close.index)
    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.sort_index().ffill(limit=5).dropna(how="all")

    for t in tickers:
        if t in close.columns and close[t].isna().all():
            print(f"  WARN: all prices for {t} are NaN")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        print(f"  WARN: tickers not found: {missing}")

    print(f"  Downloaded: {len(close)} sessions, {len(close.columns)} tickers")
    return close


# ================================================================
# ANALYTICS ENGINE
# ================================================================
def build_frame(prices: pd.DataFrame, holdings: pd.DataFrame) -> dict:
    px     = prices.copy()
    bench  = CFG["benchmark"]
    bench2 = CFG["bench2"]
    init   = holdings["cost_basis"].sum()

    pos_mv = pd.DataFrame(index=px.index)
    for _, r in holdings.iterrows():
        t = r["ticker"]
        if t in px.columns:
            pos_mv[t] = px[t] * r["quantity"]

    nav = pos_mv.sum(axis=1)
    qqq_units = init / float(px[bench].dropna().iloc[0])
    qqq_nav   = px[bench] * qqq_units

    if bench2 in px.columns:
        spy_units = init / float(px[bench2].dropna().iloc[0])
        spy_nav   = px[bench2] * spy_units
    else:
        spy_nav = pd.Series(dtype=float, index=px.index)

    ret   = nav.pct_change().fillna(0)
    bret  = qqq_nav.pct_change().fillna(0)
    s2ret = spy_nav.pct_change().fillna(0) if not spy_nav.empty else pd.Series(dtype=float, index=px.index)

    cum   = (1 + ret).cumprod()
    bcum  = (1 + bret).cumprod()
    s2cum = (1 + s2ret).cumprod() if not s2ret.empty else pd.Series(dtype=float, index=px.index)

    dd  = (nav  / nav.cummax()  - 1) * 100
    bdd = (qqq_nav / qqq_nav.cummax() - 1) * 100

    return {
        "prices":      px,
        "pos_mv":      pos_mv,
        "nav":         nav,
        "bench_nav":   qqq_nav,
        "bench2_nav":  spy_nav,
        "ret":         ret,
        "bret":        bret,
        "b2ret":       s2ret,
        "b100":        cum  * 100,
        "bb100":       bcum * 100,
        "b2b100":      s2cum * 100,
        "dd":          dd,
        "bdd":         bdd,
    }


def _ann(total: float, n: int, af: int = 252) -> float:
    if n <= 0: return float("nan")
    return (1 + total) ** (af / n) - 1


def _downside(r: pd.Series, mar: float, af: int) -> float:
    d = np.minimum(r - mar, 0)
    return float(np.sqrt(np.mean(d**2)) * np.sqrt(af))


def _omega(r: pd.Series, mar: float) -> float:
    d = r - mar
    g = d[d > 0].sum()
    l = -d[d < 0].sum()
    return float(g / l) if l > 0 else float("nan")


def _capture(port: pd.Series, bench: pd.Series, up: bool) -> float:
    mask = bench > 0 if up else bench < 0
    if mask.sum() == 0: return float("nan")
    b = bench[mask].mean()
    if abs(b) < 1e-12: return float("nan")
    return float(port[mask].mean() / b)


def compute_metrics(frame: dict) -> dict:
    af  = CFG["af"]
    rf  = CFG["rf"]
    rfd = rf / af

    r  = frame["ret"]
    b  = frame["bret"]
    n  = len(r)

    tr   = float(frame["nav"].iloc[-1] / frame["nav"].iloc[0] - 1)
    trb  = float(frame["bench_nav"].iloc[-1] / frame["bench_nav"].iloc[0] - 1)
    annr = _ann(tr, n, af)
    annb = _ann(trb, n, af)

    vol  = float(r.std() * math.sqrt(af))
    bvol = float(b.std() * math.sqrt(af))
    ddev = _downside(r, rfd, af)
    bddev= _downside(b, rfd, af)

    sharpe  = (annr - rf) / vol  if vol  else float("nan")
    sortino = (annr - rf) / ddev if ddev else float("nan")
    bsharpe = (annb - rf) / bvol if bvol else float("nan")
    bsortin = (annb - rf) / bddev if bddev else float("nan")

    beta  = float(r.cov(b) / b.var()) if b.var() else float("nan")
    corr  = float(r.corr(b))
    jalpha= float(annr - (rf + beta * (annb - rf))) if not math.isnan(beta) else float("nan")

    act   = r - b
    te    = float(act.std() * math.sqrt(af))
    ir    = float((act.mean() * af) / te) if te else float("nan")

    wealth = (1 + r.fillna(0)).cumprod()
    dd_s   = (wealth / wealth.cummax() - 1)
    mdd    = float(dd_s.min())
    bwealth= (1 + b.fillna(0)).cumprod()
    bmdd   = float((bwealth / bwealth.cummax() - 1).min())
    calmar = float(annr / abs(mdd)) if mdd != 0 else float("nan")

    var95  = float(np.percentile(r, 5))
    cvar95 = float(r[r <= var95].mean())
    skew   = float(r.skew())
    kurt   = float(r.kurtosis())
    treynor= float((annr - rf) / beta) if (not math.isnan(beta) and abs(beta) > 1e-12) else float("nan")
    omega  = _omega(r, rfd)
    upc    = _capture(r, b, True)
    dnc    = _capture(r, b, False)
    hit    = float((r > 0).mean())
    bhit   = float((b > 0).mean())

    dpnl  = float(frame["nav"].iloc[-1] - frame["nav"].iloc[-2])
    dret  = float(r.iloc[-1])
    tpnl  = float(frame["nav"].iloc[-1] - frame["nav"].iloc[0])

    roll  = CFG["roll_w"]
    rvol  = r.rolling(roll).std() * math.sqrt(af) * 100
    rbeta = r.rolling(roll).cov(b) / b.rolling(roll).var()
    rsh   = ((r.rolling(roll).mean() - rfd) / r.rolling(roll).std()) * math.sqrt(af)
    rexc  = ((1 + act).rolling(roll).apply(np.prod, raw=True) - 1) * 100

    mp = (1 + r).resample("ME").prod() - 1
    mb = (1 + b).resample("ME").prod() - 1
    yp = (1 + r).resample("YE").prod() - 1
    yb = (1 + b).resample("YE").prod() - 1

    return dict(
        current_nav=float(frame["nav"].iloc[-1]),
        daily_pnl=dpnl, daily_return=dret,
        total_pnl=tpnl, total_return=tr,
        bench_total_return=trb,
        alpha=tr - trb,
        ann_return=annr, ann_bench=annb,
        vol=vol, bvol=bvol, ddev=ddev, bddev=bddev,
        sharpe=sharpe, sortino=sortino, bsharpe=bsharpe, bsortino=bsortin,
        beta=beta, corr=corr, jalpha=jalpha,
        te=te, ir=ir,
        mdd=mdd, bmdd=bmdd, calmar=calmar,
        var95=var95, cvar95=cvar95,
        skew=skew, kurt=kurt, treynor=treynor, omega=omega,
        upc=upc, dnc=dnc, hit=hit, bhit=bhit,
        sessions=n,
        rvol=rvol, rbeta=rbeta, rsh=rsh, rexc=rexc,
        mp=mp, mb=mb, yp=yp, yb=yb,
    )


def compute_positions(frame: dict, holdings: pd.DataFrame) -> pd.DataFrame:
    px   = frame["prices"]
    bret = frame["bret"]
    nav  = float(frame["nav"].iloc[-1])
    init_aum = holdings["cost_basis"].sum()
    rows = []
    for _, h in holdings.iterrows():
        t = h["ticker"]
        if t not in px.columns: continue
        s = px[t].dropna()
        r = s.pct_change().dropna()
        p = float(s.iloc[-1])
        pp= float(s.iloc[-2]) if len(s) >= 2 else p
        mv= p * float(h["quantity"])
        cb= float(h["cost_basis"])
        pnl = mv - cb

        def trail(d: int) -> float:
            return float(s.iloc[-1]/s.iloc[-d-1]-1) if len(s) > d else float("nan")

        aligned = pd.concat([r, bret], axis=1, join="inner").dropna()
        aligned.columns = ["r","b"]
        beta_i = float(aligned["r"].cov(aligned["b"])/aligned["b"].var()) \
            if len(aligned) > 10 and aligned["b"].var() else float("nan")

        rows.append(dict(
            ticker=t, name=h["name"], sector=h["sector"],
            theme=h["theme"], risk_bucket=h["risk_bucket"],
            quantity=float(h["quantity"]), buy_price=float(h["buy_price"]),
            latest_price=p, market_value=mv,
            pnl=pnl, ret=pnl/cb, weight=mv/nav,
            contribution=pnl/init_aum,
            d1=p/pp-1, d5=trail(5), d21=trail(21),
            beta=beta_i,
        ))
    pos = pd.DataFrame(rows)
    pos = pos.sort_values("market_value", ascending=False).reset_index(drop=True)
    return pos


def compute_structure(pos: pd.DataFrame) -> dict:
    sector = pos.groupby("sector", as_index=False).agg(
        weight=("weight","sum"), market_value=("market_value","sum"),
        pnl=("pnl","sum"), n=("ticker","count"),
    ).sort_values("weight", ascending=False)

    theme = pos.groupby("theme", as_index=False).agg(
        weight=("weight","sum"), market_value=("market_value","sum"),
        pnl=("pnl","sum"), n=("ticker","count"),
    ).sort_values("weight", ascending=False)

    bucket = pos.groupby("risk_bucket", as_index=False).agg(
        weight=("weight","sum"), market_value=("market_value","sum"),
        pnl=("pnl","sum"), n=("ticker","count"),
    ).sort_values("weight", ascending=False)

    hhi    = float((pos["weight"]**2).sum())
    eff_n  = float(1/hhi) if hhi else float("nan")
    top5   = float(pos["weight"].head(5).sum())

    return dict(sector=sector, theme=theme, bucket=bucket,
                hhi=hhi, eff_n=eff_n, top5=top5)


def build_ledger(frame: dict) -> pd.DataFrame:
    nav = frame["nav"]
    df = pd.DataFrame({
        "date":      nav.index,
        "nav":       nav.values,
        "daily_pnl": nav.diff().fillna(0).values,
        "daily_ret": frame["ret"].values,
        "bench_ret": frame["bret"].values,
        "active":    (frame["ret"] - frame["bret"]).values,
        "dd":        frame["dd"].values / 100,
    })
    return df.tail(40).iloc[::-1].reset_index(drop=True)


def build_heatmap(mp: pd.Series) -> pd.DataFrame:
    df = mp.to_frame("r")
    df["year"]  = df.index.year
    df["month"] = df.index.strftime("%b")
    pivot = df.pivot(index="year", columns="month", values="r")
    order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in order if m in pivot.columns])
    return pivot


def build_stress(pos: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    nav  = pos["market_value"].sum()
    beta = metrics["beta"]
    rows = []
    for s in SCENARIOS:
        shock = pos[["ticker","market_value","risk_bucket","theme"]].copy()
        shock["sh"] = 0.0
        if s["type"] == "benchmark":
            shock["sh"] = beta * s["shock"]
        elif s["type"] == "bucket":
            shock.loc[shock["risk_bucket"]==s["bucket"],"sh"] = s["shock"]
        elif s["type"] == "theme_kw":
            mask = shock["theme"].str.contains("|".join(s["kws"]), case=False, na=False)
            shock.loc[mask,"sh"] = s["shock"]
        elif s["type"] == "tickers":
            shock.loc[shock["ticker"].isin(s["tickers"]),"sh"] = s["shock"]
        elif s["type"] == "custom":
            shock["sh"] = shock["risk_bucket"].map(s["mapping"]).fillna(0.0)
        total = (shock["market_value"] * shock["sh"]).sum()
        rows.append(dict(
            scenario=s["name"], desc=s["desc"],
            pnl_impact=total, ret_impact=total/nav, nav_after=nav+total,
        ))
    return pd.DataFrame(rows).sort_values("pnl_impact")


def build_forecast(frame: dict, metrics: dict):
    np.random.seed(CFG["mc_seed"])
    r   = frame["ret"].dropna()
    b   = frame["bret"].dropna()
    mu  = float(r.mean()); sig = float(r.std())
    bmu = float(b.mean()); bsig= float(b.std())
    nav0= float(frame["nav"].iloc[-1])
    horizons = {"3M": 63, "6M": 126, "12M": 252, "15Y": 252*15}
    sum_rows, path_rows = [], []
    for lbl, h in horizons.items():
        shocks  = np.random.normal(mu, sig, (CFG["mc_paths"], h))
        wealth  = nav0 * np.cumprod(1 + shocks, axis=1)
        ending  = wealth[:,-1]
        sum_rows.append(dict(
            horizon=lbl, start_nav=nav0,
            p05=float(np.percentile(ending,5)),
            p25=float(np.percentile(ending,25)),
            median=float(np.percentile(ending,50)),
            p75=float(np.percentile(ending,75)),
            p95=float(np.percentile(ending,95)),
        ))
        steps = list(range(h+1))
        nav_col = np.full((CFG["mc_paths"],1), nav0)
        all_w = np.hstack([nav_col, wealth])
        p05p  = np.percentile(all_w, 5,  axis=0).tolist()
        p95p  = np.percentile(all_w, 95, axis=0).tolist()
        bull  = (nav0 * np.cumprod(np.r_[1, np.repeat(bmu+0.75*bsig, h)])).tolist()
        base  = (nav0 * np.cumprod(np.r_[1, np.repeat(mu,            h)])).tolist()
        bear  = (nav0 * np.cumprod(np.r_[1, np.repeat(mu-0.75*sig,   h)])).tolist()
        for i, step in enumerate(steps):
            path_rows.append(dict(
                horizon=lbl, step=step,
                bull=bull[i], base=base[i], bear=bear[i],
                mc_low=p05p[i], mc_high=p95p[i],
            ))
    return pd.DataFrame(sum_rows), pd.DataFrame(path_rows)


def build_news(holdings: pd.DataFrame) -> pd.DataFrame:
    items = []
    for ticker in holdings["ticker"].tolist():
        try:
            tk   = yf.Ticker(ticker)
            news = getattr(tk, "news", None) or []
            for a in news[:5]:
                ct  = a.get("content", {}) if isinstance(a, dict) else {}
                ttl = ct.get("title") or a.get("title")
                url = ct.get("canonicalUrl", {}).get("url") or a.get("link") or a.get("url")
                src = ct.get("provider", {}).get("displayName") or a.get("publisher") or "Yahoo Finance"
                pub = ct.get("pubDate") or a.get("providerPublishTime")
                if isinstance(pub, str):
                    try:    pub = pd.to_datetime(pub, utc=True)
                    except: pub = pd.NaT
                elif pub is not None:
                    pub = pd.to_datetime(pub, unit="s", utc=True, errors="coerce")
                else:
                    pub = pd.NaT
                if ttl and url:
                    items.append(dict(ticker=ticker, title=ttl, source=src,
                                      published_at=pub, url=url))
        except Exception:
            continue
    if not items:
        return pd.DataFrame(columns=["ticker","title","source","published_at","url"])
    df = pd.DataFrame(items)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at", ascending=False)
    df["_norm"] = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"])
    return df.head(CFG["news_n"]).reset_index(drop=True)


def build_intelligence(metrics: dict, pos: pd.DataFrame, structure: dict,
                        stress: pd.DataFrame) -> list:
    top3  = pos.nlargest(3, "contribution")
    bot3  = pos.nsmallest(3, "contribution")
    dsec  = structure["sector"].iloc[0]
    wstress = stress.iloc[0]
    return [
        ("Portfolio DNA",
         f"Concentrated US equity book with high-conviction thematic growth bias. "
         f"Top 5 positions represent {structure['top5']*100:.1f}% of NAV. "
         f"HHI {structure['hhi']:.3f} → effective position count {structure['eff_n']:.1f}. "
         f"High-beta by design: β={metrics['beta']:.2f}x vs {CFG['benchmark']}."),
        ("What worked",
         f"Performance led by {', '.join(top3['ticker'].tolist())} — "
         f"top contributor {top3.iloc[0]['ticker']} at "
         f"{top3.iloc[0]['contribution']*100:+.1f}% of initial capital. "
         f"Main detractors: {', '.join(bot3['ticker'].tolist())}."),
        ("Risk lens",
         f"β={metrics['beta']:.2f}x, ann. vol {metrics['vol']*100:.1f}% "
         f"vs QQQ {metrics['bvol']*100:.1f}%. Max DD {metrics['mdd']*100:.1f}% "
         f"vs QQQ {metrics['bmdd']*100:.1f}%. "
         f"Harshest modelled shock: '{wstress['scenario']}' "
         f"({wstress['ret_impact']*100:.1f}% NAV hit)."),
        ("Benchmark lens",
         f"Alpha vs {CFG['benchmark']}: {metrics['alpha']*100:+.1f}% since inception. "
         f"IR {metrics['ir']:.2f}. Upside capture {metrics['upc']:.2f}x, "
         f"downside capture {metrics['dnc']:.2f}x. "
         f"Benchmark chosen for growth-centric opportunity cost alignment."),
        ("Concentration lens",
         f"Dominant sector: {dsec['sector']} at {dsec['weight']*100:.1f}% NAV. "
         f"Risk bucket mix: "
         + ", ".join(f"{r['risk_bucket']} {r['weight']*100:.0f}%"
                     for _, r in structure['bucket'].iterrows())
         + ". No rebalancing → winners drift to higher weights naturally."),
    ]


# ================================================================
# CHARTS
# ================================================================
def _layout(title: str, h: int = 360) -> dict:
    return dict(
        title=dict(text=title, x=0.01, font=dict(size=14, color=C["text"])),
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        margin=dict(l=55, r=20, t=55, b=45),
        font=dict(family="JetBrains Mono, monospace", size=11, color=C["text"]),
        xaxis=dict(gridcolor=C["grid"], zeroline=False, showline=False,
                   tickfont=dict(color=C["muted"])),
        yaxis=dict(gridcolor=C["grid"], zeroline=False, showline=False,
                   tickfont=dict(color=C["muted"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01,
                    bgcolor=C["card"]),
        hoverlabel=dict(bgcolor=C["card"], bordercolor=C["border"],
                        font=dict(color=C["text"])),
        hovermode="x unified", height=h,
    )


def make_charts(frame: dict, metrics: dict, pos: pd.DataFrame,
                structure: dict, heatmap: pd.DataFrame,
                stress: pd.DataFrame, fp: pd.DataFrame) -> dict:
    charts = {}

    # 1. Base-100 performance
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["b100"].index,  y=frame["b100"],
        name=CFG["portfolio_name"], line=dict(color=C["green"], width=2.5)))
    fig.add_trace(go.Scatter(x=frame["bb100"].index, y=frame["bb100"],
        name=CFG["benchmark"], line=dict(color=C["blue"], width=2)))
    if not frame["b2b100"].empty:
        fig.add_trace(go.Scatter(x=frame["b2b100"].index, y=frame["b2b100"],
            name=CFG["bench2"], line=dict(color=C["purple"], width=1.6, dash="dot")))
    fig.update_layout(**_layout("Portfolio vs Benchmark — Base 100", 400))
    charts["perf"] = fig.to_plotly_json()

    # 2. Drawdown
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["dd"].index, y=frame["dd"],
        name=CFG["portfolio_name"], fill="tozeroy",
        line=dict(color=C["red"], width=2)))
    fig.add_trace(go.Scatter(x=frame["bdd"].index, y=frame["bdd"],
        name=CFG["benchmark"], line=dict(color=C["blue"], width=1.8)))
    fig.update_layout(**_layout("Drawdown from Peak (%)"))
    charts["drawdown"] = fig.to_plotly_json()

    # 3. Rolling volatility
    rv = metrics["rvol"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv.index, y=rv,
        name="Portfolio vol", line=dict(color=C["amber"], width=2)))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Volatility (Ann. %)"))
    charts["rolling_vol"] = fig.to_plotly_json()

    # 4. Rolling beta
    rb = metrics["rbeta"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rb.index, y=rb,
        name="Rolling beta", line=dict(color=C["cyan"], width=2)))
    fig.add_hline(y=1.0, line=dict(color=C["dim"], dash="dot"))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Beta vs {CFG['benchmark']}"))
    charts["rolling_beta"] = fig.to_plotly_json()

    # 5. Rolling Sharpe
    rsh = metrics["rsh"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsh.index, y=rsh,
        name="Rolling Sharpe", line=dict(color=C["green"], width=2)))
    fig.update_layout(**_layout(f"Rolling {CFG['roll_w']}-Day Sharpe"))
    charts["rolling_sharpe"] = fig.to_plotly_json()

    # 6. Monthly bar chart  — NOTE: _colors() returns a list, not ndarray
    mp   = metrics["mp"]
    mdf  = pd.DataFrame({"date": mp.index, "val": mp.values * 100})
    fig  = go.Figure()
    fig.add_trace(go.Bar(
        x=mdf["date"], y=mdf["val"],
        name=CFG["portfolio_name"],
        marker_color=_colors(mdf["val"]),   # ← list, never ndarray
    ))
    fig.add_trace(go.Scatter(
        x=metrics["mb"].index, y=metrics["mb"].values * 100,
        name=CFG["benchmark"], line=dict(color=C["blue"], width=2),
    ))
    fig.update_layout(**_layout("Monthly Returns (%)"))
    charts["monthly_returns"] = fig.to_plotly_json()

    # 7. Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=(heatmap * 100).values.tolist(),
        x=list(heatmap.columns),
        y=[str(y) for y in heatmap.index],
        colorscale=[[0,C["red"]], [0.5,C["panel2"]], [1,C["green"]]],
        text=[[("" if pd.isna(v) else f"{v*100:+.1f}%") for v in row]
              for row in heatmap.values],
        texttemplate="%{text}",
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_layout("Monthly Return Heatmap", 300))
    charts["heatmap"] = fig.to_plotly_json()

    # 8. Top weights bar
    fig = go.Figure(data=[go.Bar(
        x=pos["weight"].head(10) * 100,
        y=pos["ticker"].head(10),
        orientation="h",
        marker_color=[BUCKET_C.get(x, C["blue"]) for x in pos["risk_bucket"].head(10)],
        text=[f"{w*100:.1f}%" for w in pos["weight"].head(10)],
        textposition="auto",
    )])
    fig.update_layout(**_layout("Top Weights (%)"))
    fig.update_yaxes(autorange="reversed")
    charts["top_weights"] = fig.to_plotly_json()

    # 9. Sector donut
    fig = go.Figure(data=[go.Pie(
        labels=structure["sector"]["sector"].tolist(),
        values=(structure["sector"]["weight"]*100).tolist(),
        hole=0.55,
        marker=dict(colors=[SC.get(s, C["blue"]) for s in structure["sector"]["sector"]]),
        sort=False,
    )])
    fig.update_layout(**_layout("Sector Allocation (%)", 360))
    charts["sector_alloc"] = fig.to_plotly_json()

    # 10. Theme donut
    theme_colors = [C["blue"],C["purple"],C["green"],C["amber"],C["cyan"],
                    "#8f9fb5","#6f80a0","#b0b9c8","#d0d6df","#5c8bd8",
                    "#7db0ff","#2ecf91","#ff9e66"]
    fig = go.Figure(data=[go.Pie(
        labels=structure["theme"]["theme"].tolist(),
        values=(structure["theme"]["weight"]*100).tolist(),
        hole=0.55,
        marker=dict(colors=theme_colors[:len(structure["theme"])]),
        sort=False,
    )])
    fig.update_layout(**_layout("Thematic Allocation (%)", 360))
    charts["theme_alloc"] = fig.to_plotly_json()

    # 11. P&L attribution  — _colors() returns a list
    fig = go.Figure(data=[go.Bar(
        x=pos["ticker"].tolist(),
        y=pos["pnl"].tolist(),
        marker_color=_colors(pos["pnl"]),   # ← list
        text=[_fc(x, 0) for x in pos["pnl"]],
        textposition="outside",
    )])
    fig.update_layout(**_layout("Position P&L Attribution ($)"))
    charts["pnl_attr"] = fig.to_plotly_json()

    # 12. Stress bar  — _colors() returns a list
    fig = go.Figure(data=[go.Bar(
        x=stress["pnl_impact"].tolist(),
        y=stress["scenario"].tolist(),
        orientation="h",
        marker_color=_colors(stress["pnl_impact"]),   # ← list
        text=[_fc(x, 0) for x in stress["pnl_impact"]],
        textposition="auto",
    )])
    fig.update_layout(**_layout("Stress Testing — Estimated P&L Impact", 360))
    fig.update_yaxes(autorange="reversed")
    charts["stress"] = fig.to_plotly_json()

    # 13. Forecast
    fp12 = fp[fp["horizon"] == "12M"]
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["mc_high"].tolist(),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["mc_low"].tolist(),
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(77,141,255,0.15)", name="Monte Carlo 5–95%"))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["bull"].tolist(),
        name="Bull", line=dict(color=C["green"], width=2)))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["base"].tolist(),
        name="Base", line=dict(color=C["blue"], width=2)))
    fig.add_trace(go.Scatter(x=fp12["step"].tolist(), y=fp12["bear"].tolist(),
        name="Bear", line=dict(color=C["red"], width=2)))
    fig.update_layout(**_layout("12M Scenario Envelope — Model-Based Paths", 360))
    charts["forecast"] = fig.to_plotly_json()

    return charts


# ================================================================
# HTML COMPONENTS
# ================================================================
def _kpi(label: str, value: str, sub: str = "", tone: str = "blue") -> str:
    return (f"<div class='kpi-card {tone}'>"
            f"<div class='kpi-label'>{_esc(label)}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>")


def _positions_table(pos: pd.DataFrame) -> str:
    rows = []
    for _, r in pos.iterrows():
        rows.append(
            f"<tr>"
            f"<td class='mono strong'>{r['ticker']}</td>"
            f"<td>{_esc(r['name'])}</td>"
            f"<td>{_esc(r['sector'])}</td>"
            f"<td>{_esc(r['theme'])}</td>"
            f"<td><span class='bucket {r['risk_bucket'].lower()}'>{_esc(r['risk_bucket'])}</span></td>"
            f"<td class='num'>{r['quantity']:,.0f}</td>"
            f"<td class='num'>{_fc(r['buy_price'],2)}</td>"
            f"<td class='num'>{_fc(r['latest_price'],2)}</td>"
            f"<td class='num'>{_fc(r['market_value'],0)}</td>"
            f"<td class='num' style='color:{_col(r['pnl'])}'>{_fc(r['pnl'],0)}</td>"
            f"<td class='num' style='color:{_col(r['ret'])}'>{_fp(r['ret']*100,1)}</td>"
            f"<td class='num'>{r['weight']*100:.1f}%</td>"
            f"<td class='num' style='color:{_col(r['contribution'])}'>{r['contribution']*100:+.1f}%</td>"
            f"<td class='num' style='color:{_col(r.d1)}'>{_fp(r['d1']*100,1)}</td>"
            f"<td class='num' style='color:{_col(r.d5)}'>{_fp(r['d5']*100,1)}</td>"
            f"<td class='num' style='color:{_col(r.d21)}'>{_fp(r['d21']*100,1)}</td>"
            f"<td class='num'>{r['beta']:.2f}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _metrics_table(m: dict, pos: pd.DataFrame) -> str:
    rows = [
        ("Current NAV",         _fc(m["current_nav"],0),                "Marked-to-market net asset value."),
        ("Total P&L",           _fc(m["total_pnl"],0),                  "Absolute P&L since inception."),
        ("Total return",        _fp(m["total_return"]*100),             "Portfolio return since inception."),
        ("Daily P&L",           _fc(m["daily_pnl"],0),                  "Latest session P&L."),
        ("Daily return",        _fp(m["daily_return"]*100),             "Latest session return."),
        ("Benchmark return",    _fp(m["bench_total_return"]*100),       f"{CFG['benchmark']} total return since inception."),
        ("Alpha vs benchmark",  _fp(m["alpha"]*100),                    "Simple excess return vs QQQ."),
        ("Annualized return",   _fp(m["ann_return"]*100),               "Compounded annualized return."),
        ("Annualized vol",      _fp(m["vol"]*100),                      "Realized annualized volatility."),
        ("Sharpe ratio",        f"{m['sharpe']:.3f}",                   f"(ann. return − rf) / vol. rf={CFG['rf']*100:.2f}%."),
        ("Sortino ratio",       f"{m['sortino']:.3f}",                  "Excess return per unit of downside deviation."),
        ("Calmar ratio",        f"{m['calmar']:.3f}",                   "Ann. return / |max drawdown|."),
        ("Max drawdown",        _fp(m["mdd"]*100),                      "Peak-to-trough portfolio drawdown."),
        ("VaR 95% (1D)",        _fp(m["var95"]*100),                    "Historical daily 5th-percentile return."),
        ("CVaR 95% (1D)",       _fp(m["cvar95"]*100),                   "Expected shortfall below VaR."),
        ("Beta",                f"{m['beta']:.3f}",                     f"CAPM sensitivity vs {CFG['benchmark']}."),
        ("Correlation",         f"{m['corr']:.3f}",                     f"Return correlation with {CFG['benchmark']}."),
        ("Jensen alpha",        _fp(m["jalpha"]*100),                   "CAPM alpha."),
        ("Tracking error",      _fp(m["te"]*100),                       "Ann. std of active returns."),
        ("Information ratio",   f"{m['ir']:.3f}",                       "Active return / tracking error."),
        ("Treynor ratio",       f"{m['treynor']:.3f}",                  "(ann. return − rf) / beta."),
        ("Omega ratio",         f"{m['omega']:.3f}",                    "Gain/loss above daily rf threshold."),
        ("Downside deviation",  _fp(m["ddev"]*100),                     "Ann. downside deviation (MAR = rf)."),
        ("Upside capture",      _fx(m["upc"]),                          "Mean return on up days vs benchmark."),
        ("Downside capture",    _fx(m["dnc"]),                          "Mean return on down days vs benchmark."),
        ("Skewness",            f"{m['skew']:.3f}",                     "Return distribution skewness."),
        ("Kurtosis",            f"{m['kurt']:.3f}",                     "Excess kurtosis of daily returns."),
        ("Hit ratio",           _fp(m["hit"]*100),                      "% positive return days."),
        ("Sessions",            str(m["sessions"]),                     "Trading days since inception."),
    ]
    return "\n".join(
        f"<tr><td>{_esc(l)}</td><td class='num strong'>{v}</td><td>{_esc(n)}</td></tr>"
        for l, v, n in rows
    )


def _monthly_table(m: dict) -> str:
    mp = m["mp"]; mb = m["mb"]
    df = pd.DataFrame({
        "Month":     mp.index.strftime("%Y-%m"),
        "Portfolio": mp.values,
        "QQQ":       mb.values,
        "Active":    (mp - mb).values,
    }).tail(12).iloc[::-1]

    yp = m["yp"]; yb = m["yb"]
    ydf = pd.DataFrame({
        "Year":      yp.index.year.astype(str),
        "Portfolio": yp.values,
        "QQQ":       yb.values,
        "Active":    (yp - yb).values,
    }).iloc[::-1]

    parts = ["<div class='split-table'>"]
    parts.append("<div><div class='mini-title'>Monthly returns</div>"
                 "<table class='data-table'><thead><tr>"
                 "<th>Month</th><th>Portfolio</th><th>QQQ</th><th>Active</th>"
                 "</tr></thead><tbody>")
    for _, r in df.iterrows():
        parts.append(
            f"<tr><td>{r['Month']}</td>"
            f"<td class='num' style='color:{_col(r['Portfolio'])}'>{_fp(r['Portfolio']*100,1)}</td>"
            f"<td class='num'>{_fp(r['QQQ']*100,1)}</td>"
            f"<td class='num' style='color:{_col(r['Active'])}'>{_fp(r['Active']*100,1)}</td></tr>"
        )
    parts.append("</tbody></table></div>")
    parts.append("<div><div class='mini-title'>Annual returns</div>"
                 "<table class='data-table'><thead><tr>"
                 "<th>Year</th><th>Portfolio</th><th>QQQ</th><th>Active</th>"
                 "</tr></thead><tbody>")
    for _, r in ydf.iterrows():
        parts.append(
            f"<tr><td>{r['Year']}</td>"
            f"<td class='num' style='color:{_col(r['Portfolio'])}'>{_fp(r['Portfolio']*100,1)}</td>"
            f"<td class='num'>{_fp(r['QQQ']*100,1)}</td>"
            f"<td class='num' style='color:{_col(r['Active'])}'>{_fp(r['Active']*100,1)}</td></tr>"
        )
    parts.append("</tbody></table></div></div>")
    return "".join(parts)


def _sector_table(structure: dict) -> str:
    rows = []
    for _, r in structure["sector"].iterrows():
        rows.append(
            f"<tr><td>{_esc(r['sector'])}</td>"
            f"<td class='num'>{r['weight']*100:.1f}%</td>"
            f"<td class='num'>{_fc(r['market_value'],0)}</td>"
            f"<td class='num' style='color:{_col(r['pnl'])}'>{_fc(r['pnl'],0)}</td>"
            f"<td class='num'>{int(r['n'])}</td></tr>"
        )
    return "\n".join(rows)


def _ledger_table(ledger: pd.DataFrame) -> str:
    rows = []
    for _, r in ledger.iterrows():
        rows.append(
            f"<tr><td>{pd.to_datetime(r['date']).strftime('%Y-%m-%d')}</td>"
            f"<td class='num'>{_fc(r['nav'],0)}</td>"
            f"<td class='num' style='color:{_col(r['daily_pnl'])}'>{_fc(r['daily_pnl'],0)}</td>"
            f"<td class='num' style='color:{_col(r['daily_ret'])}'>{_fp(r['daily_ret']*100,2)}</td>"
            f"<td class='num'>{_fp(r['bench_ret']*100,2)}</td>"
            f"<td class='num' style='color:{_col(r['active'])}'>{_fp(r['active']*100,2)}</td>"
            f"<td class='num' style='color:{_col(r['dd'])}'>{_fp(r['dd']*100,2)}</td></tr>"
        )
    return "\n".join(rows)


def _stress_table(stress: pd.DataFrame) -> str:
    rows = []
    for _, r in stress.iterrows():
        rows.append(
            f"<tr><td class='strong'>{_esc(r['scenario'])}</td>"
            f"<td>{_esc(r['desc'])}</td>"
            f"<td class='num' style='color:{_col(r.pnl_impact)}'>{_fc(r['pnl_impact'],0)}</td>"
            f"<td class='num' style='color:{_col(r.ret_impact)}'>{_fp(r['ret_impact']*100,1)}</td>"
            f"<td class='num'>{_fc(r['nav_after'],0)}</td></tr>"
        )
    return "\n".join(rows)


def _forecast_table(fs: pd.DataFrame) -> str:
    rows = []
    for _, r in fs.iterrows():
        rows.append(
            f"<tr><td class='strong'>{_esc(r['horizon'])}</td>"
            f"<td class='num'>{_fc(r['start_nav'],0)}</td>"
            f"<td class='num red'>{_fc(r['p05'],0)}</td>"
            f"<td class='num'>{_fc(r['p25'],0)}</td>"
            f"<td class='num green'>{_fc(r['median'],0)}</td>"
            f"<td class='num'>{_fc(r['p75'],0)}</td>"
            f"<td class='num amber'>{_fc(r['p95'],0)}</td></tr>"
        )
    return "\n".join(rows)


def _news_table(news: pd.DataFrame) -> str:
    if news.empty:
        return ("<tr><td colspan='4' class='empty-state'>"
                "News retrieval skipped or unavailable. Core analytics unaffected.</td></tr>")
    rows = []
    for _, r in news.iterrows():
        pub = pd.to_datetime(r["published_at"], utc=True, errors="coerce")
        pt  = pub.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(pub) else "—"
        rows.append(
            f"<tr><td class='mono strong'>{_esc(r['ticker'])}</td>"
            f"<td><a href='{_esc(str(r.url))}' target='_blank' rel='noopener noreferrer'>"
            f"{_esc(str(r['title']))}</a></td>"
            f"<td>{_esc(str(r['source']))}</td>"
            f"<td>{pt}</td></tr>"
        )
    return "\n".join(rows)


# ================================================================
# HTML GENERATION
# ================================================================
def generate_html(
    holdings, frame, metrics, pos, structure, ledger,
    heatmap, stress, fcast_sum, fcast_paths, news, intel, charts
) -> str:
    now_utc  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    init_aum = float(holdings["cost_basis"].sum())
    rf_disp  = f"{CFG['rf']*100:.2f}%"
    inc      = CFG["inception"]
    pname    = CFG["portfolio_name"]

    # Serialise all charts in one shot with the numpy-safe encoder
    chart_json = _dumps({k: v for k, v in charts.items()})

    kpi1 = "".join([
        _kpi("Current NAV",   _fc(metrics["current_nav"],0),
             f"Inception {inc}", "green"),
        _kpi("Total P&L",     _fc(metrics["total_pnl"],0),
             f"AUM₀ {_fc(init_aum,0)}",
             "green" if metrics["total_pnl"] >= 0 else "red"),
        _kpi("Total Return",  _fp(metrics["total_return"]*100),
             f"vs QQQ {_fp(metrics['bench_total_return']*100)}",
             "green" if metrics["total_return"] >= 0 else "red"),
        _kpi("Daily P&L",     _fc(metrics["daily_pnl"],0),
             _fp(metrics["daily_return"]*100),
             "green" if metrics["daily_pnl"] >= 0 else "red"),
        _kpi("Alpha vs QQQ",  _fp(metrics["alpha"]*100),
             "Simple excess return",
             "green" if metrics["alpha"] >= 0 else "red"),
    ])
    kpi2 = "".join([
        _kpi("Beta",          f"{metrics['beta']:.3f}×",
             f"ρ={metrics['corr']:.3f}", "amber"),
        _kpi("Sharpe",        f"{metrics['sharpe']:.3f}",
             f"rf={rf_disp}", "blue"),
        _kpi("Sortino",       f"{metrics['sortino']:.3f}",
             "Downside-adjusted", "blue"),
        _kpi("Max Drawdown",  _fp(metrics["mdd"]*100),
             f"vs QQQ {_fp(metrics['bmdd']*100)}", "red"),
        _kpi("Ann. Vol",      _fp(metrics["vol"]*100),
             f"vs QQQ {_fp(metrics['bvol']*100)}", "amber"),
        _kpi("Info Ratio",    f"{metrics['ir']:.3f}",
             f"TE {_fp(metrics['te']*100)}", "blue"),
        _kpi("Hit Ratio",     _fp(metrics["hit"]*100),
             f"{metrics['sessions']} sessions", "purple"),
    ])

    intel_html = "".join(
        f"<div class='intel-block'>"
        f"<div class='intel-title'>{_esc(t)}</div>"
        f"<p>{_esc(b)}</p></div>"
        for t, b in intel
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{_esc(pname)} — Analytics Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
  --bg:{C['bg']};--panel:{C['panel']};--panel2:{C['panel2']};
  --card:{C['card']};--card2:{C['card2']};
  --border:{C['border']};--border2:{C['border2']};
  --text:{C['text']};--muted:{C['muted']};--dim:{C['dim']};--grid:{C['grid']};
  --green:{C['green']};--red:{C['red']};--amber:{C['amber']};
  --blue:{C['blue']};--cyan:{C['cyan']};--purple:{C['purple']};
  --mono:'JetBrains Mono',monospace;--sans:'Inter',system-ui,sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;line-height:1.55;min-height:100vh;overflow-x:hidden}}
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:var(--panel)}}
::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:2px}}
.shell{{display:flex;min-height:100vh}}
.sidebar{{width:210px;min-width:210px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto;z-index:100;flex-shrink:0}}
.content{{flex:1;min-width:0;display:flex;flex-direction:column}}
main{{padding:20px 24px 60px}}
.sb-header{{padding:18px 16px 14px;border-bottom:1px solid var(--border)}}
.sb-name{{font-family:var(--mono);font-size:11.5px;font-weight:700;color:var(--green);line-height:1.3}}
.sb-tagline{{font-family:var(--mono);font-size:8px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-top:4px}}
.sb-pill{{display:inline-flex;align-items:center;gap:5px;background:rgba(33,208,122,0.08);border:1px solid rgba(33,208,122,0.18);border-radius:3px;padding:3px 8px;margin-top:8px;font-family:var(--mono);font-size:7.5px;color:var(--green);letter-spacing:.5px}}
.pulse{{width:5px;height:5px;border-radius:50%;background:var(--green);animation:blink 2s infinite}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.2}}}}
.sb-meta{{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:6px}}
.sb-row{{display:flex;justify-content:space-between;align-items:center}}
.sb-label{{font-family:var(--mono);font-size:7.5px;color:var(--dim);letter-spacing:1.5px;text-transform:uppercase}}
.sb-value{{font-family:var(--mono);font-size:10px;font-weight:600;color:var(--text)}}
.sb-value.green{{color:var(--green)}}.sb-value.red{{color:var(--red)}}.sb-value.amber{{color:var(--amber)}}
.sb-nav{{padding:10px 0;flex:1}}
.nav-group{{margin-bottom:4px}}
.nav-group-label{{font-family:var(--mono);font-size:7px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;padding:7px 16px 3px}}
.nav-link{{display:flex;align-items:center;gap:8px;padding:7px 16px;font-family:var(--mono);font-size:9.5px;color:var(--muted);text-decoration:none;letter-spacing:.4px;border-left:2px solid transparent;transition:all .12s}}
.nav-link:hover,.nav-link.active{{color:var(--text);background:rgba(77,141,255,.06);border-left-color:var(--blue)}}
.nav-link.active{{color:var(--green);border-left-color:var(--green)}}
.sb-footer{{padding:12px 16px;border-top:1px solid var(--border);font-family:var(--mono);font-size:7.5px;color:var(--dim);line-height:1.8}}
.topbar{{background:rgba(6,8,13,.97);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 24px;height:48px;position:sticky;top:0;z-index:90}}
.tb-left{{display:flex;align-items:center;gap:16px}}
.tb-title{{font-family:var(--mono);font-size:10.5px;font-weight:700;color:var(--text);letter-spacing:.3px}}
.tb-divider{{width:1px;height:14px;background:var(--border)}}
.tb-stat{{font-family:var(--mono);font-size:9.5px;display:flex;align-items:center;gap:5px}}
.tb-stat .lbl{{color:var(--dim)}}.tb-stat .val{{font-weight:600}}
.tb-right{{display:flex;align-items:center;gap:14px;font-family:var(--mono);font-size:8.5px;color:var(--dim)}}
.section-header{{font-family:var(--mono);font-size:8px;font-weight:700;color:var(--muted);letter-spacing:3px;text-transform:uppercase;margin:28px 0 14px;display:flex;align-items:center;gap:10px}}
.section-header::after{{content:'';flex:1;height:1px;background:var(--border)}}
.section-num{{color:var(--dim);font-size:7px;font-weight:400}}
.kpi-row{{display:grid;gap:10px;margin-bottom:10px}}
.kpi-row-5{{grid-template-columns:repeat(5,1fr)}}
.kpi-row-7{{grid-template-columns:repeat(7,1fr)}}
.kpi-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px;position:relative;overflow:hidden}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
.kpi-card.green::before{{background:var(--green)}}.kpi-card.red::before{{background:var(--red)}}
.kpi-card.blue::before{{background:var(--blue)}}.kpi-card.amber::before{{background:var(--amber)}}
.kpi-card.purple::before{{background:var(--purple)}}.kpi-card.cyan::before{{background:var(--cyan)}}
.kpi-label{{font-family:var(--mono);font-size:8px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:7px}}
.kpi-value{{font-family:var(--mono);font-size:19px;font-weight:700;line-height:1;letter-spacing:-.5px}}
.kpi-sub{{font-family:var(--mono);font-size:8.5px;color:var(--muted);margin-top:5px}}
.grid{{display:grid;gap:14px;margin-bottom:14px}}
.g2{{grid-template-columns:1fr 1fr}}.g3{{grid-template-columns:1fr 1fr 1fr}}
.g4{{grid-template-columns:1fr 1fr 1fr 1fr}}
.g65{{grid-template-columns:3fr 2fr}}.g35{{grid-template-columns:2fr 3fr}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:9px;overflow:hidden;margin-bottom:14px}}
.card-header{{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid var(--border)}}
.card-title{{font-family:var(--mono);font-size:8px;font-weight:700;color:var(--muted);letter-spacing:2px;text-transform:uppercase}}
.card-badge{{font-family:var(--mono);font-size:7.5px;color:var(--dim);padding:2px 8px;background:var(--card2);border-radius:3px;border:1px solid var(--border)}}
.card-body{{padding:14px 16px}}
.chart-wrap{{padding:4px 4px 6px}}
.table-wrap{{overflow-x:auto}}
.data-table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:10.5px}}
.data-table th{{font-size:8px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;padding:9px 12px;border-bottom:1px solid var(--border);text-align:left;background:var(--card2);font-weight:600;white-space:nowrap}}
.data-table td{{padding:7.5px 12px;border-bottom:1px solid rgba(29,42,63,.5);white-space:nowrap}}
.data-table tr:last-child td{{border-bottom:none}}
.data-table tr:hover td{{background:rgba(77,141,255,.04)}}
.num{{text-align:right}}.mono{{font-family:var(--mono)}}.strong{{font-weight:700}}
.green{{color:var(--green)}}.red{{color:var(--red)}}.amber{{color:var(--amber)}}
.bucket{{font-family:var(--mono);font-size:7.5px;padding:2px 7px;border-radius:3px;text-transform:uppercase;letter-spacing:.5px}}
.bucket.core{{background:rgba(77,141,255,.1);color:var(--blue);border:1px solid rgba(77,141,255,.2)}}
.bucket.growth{{background:rgba(33,208,122,.1);color:var(--green);border:1px solid rgba(33,208,122,.2)}}
.bucket.speculative{{background:rgba(255,190,85,.1);color:var(--amber);border:1px solid rgba(255,190,85,.2)}}
.intel-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}
.intel-block{{background:var(--card2);border:1px solid var(--border);border-radius:7px;padding:14px}}
.intel-title{{font-family:var(--mono);font-size:8px;font-weight:700;color:var(--blue);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;padding-bottom:7px;border-bottom:1px solid var(--border)}}
.intel-block p{{font-size:12px;line-height:1.85;color:var(--text)}}
.split-table{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.mini-title{{font-family:var(--mono);font-size:8px;font-weight:700;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}}
.empty-state{{font-family:var(--mono);font-size:10px;color:var(--dim);padding:20px;text-align:center}}
.disclaimer{{background:rgba(77,141,255,.04);border:1px solid rgba(77,141,255,.14);border-radius:7px;padding:12px 16px;font-size:11.5px;color:var(--muted);line-height:1.75;margin-bottom:14px}}
footer{{padding:22px 24px;border-top:1px solid var(--border);display:flex;justify-content:space-between;font-family:var(--mono);font-size:8px;color:var(--dim);line-height:1.9}}
</style>
</head>
<body>
<div class="shell">

<!-- SIDEBAR -->
<nav class="sidebar">
  <div class="sb-header">
    <div class="sb-name">{_esc(pname)}</div>
    <div class="sb-tagline">{_esc(CFG['school'])} · {_esc(CFG['school_tag'])}</div>
    <div class="sb-pill"><span class="pulse"></span>Auto-updated</div>
  </div>
  <div class="sb-meta">
    <div class="sb-row"><span class="sb-label">NAV</span><span class="sb-value {_pct_col(metrics['total_return'])}">{_fc(metrics['current_nav'],0)}</span></div>
    <div class="sb-row"><span class="sb-label">Return</span><span class="sb-value {_pct_col(metrics['total_return'])}">{_fp(metrics['total_return']*100)}</span></div>
    <div class="sb-row"><span class="sb-label">Alpha</span><span class="sb-value {_pct_col(metrics['alpha'])}">{_fp(metrics['alpha']*100)}</span></div>
    <div class="sb-row"><span class="sb-label">Beta</span><span class="sb-value amber">{metrics['beta']:.3f}×</span></div>
    <div class="sb-row"><span class="sb-label">Sharpe</span><span class="sb-value">{metrics['sharpe']:.3f}</span></div>
    <div class="sb-row"><span class="sb-label">Updated</span><span class="sb-value" style="font-size:8px">{now_utc}</span></div>
  </div>
  <div class="sb-nav">
    <div class="nav-group">
      <div class="nav-group-label">Overview</div>
      <a href="#overview"      class="nav-link active">◈ Dashboard</a>
      <a href="#positions"     class="nav-link">◈ Positions</a>
      <a href="#ledger"        class="nav-link">◈ Daily Ledger</a>
    </div>
    <div class="nav-group">
      <div class="nav-group-label">Analytics</div>
      <a href="#charts"        class="nav-link">◈ Performance Charts</a>
      <a href="#metrics"       class="nav-link">◈ Full Metrics</a>
      <a href="#monthly"       class="nav-link">◈ Monthly Returns</a>
    </div>
    <div class="nav-group">
      <div class="nav-group-label">Structure</div>
      <a href="#structure"     class="nav-link">◈ Allocation</a>
      <a href="#stress"        class="nav-link">◈ Stress Test</a>
      <a href="#scenarios"     class="nav-link">◈ Scenarios</a>
    </div>
    <div class="nav-group">
      <div class="nav-group-label">Intelligence</div>
      <a href="#intelligence"  class="nav-link">◈ Portfolio Intel</a>
      <a href="#news"          class="nav-link">◈ News Flow</a>
    </div>
  </div>
  <div class="sb-footer">
    {_esc(CFG['course'])}<br>
    Benchmark: {CFG['benchmark']}<br>
    rf = {rf_disp}<br>
    Inception: {inc}
  </div>
</nav>

<!-- CONTENT -->
<div class="content">

<!-- TOPBAR -->
<div class="topbar">
  <div class="tb-left">
    <span class="tb-title">{_esc(pname)}</span>
    <div class="tb-divider"></div>
    <div class="tb-stat"><span class="lbl">NAV</span><span class="val" style="color:var(--green)">{_fc(metrics['current_nav'],0)}</span></div>
    <div class="tb-stat"><span class="lbl">Return</span><span class="val" style="color:{_col(metrics['total_return'])}">{_fp(metrics['total_return']*100)}</span></div>
    <div class="tb-stat"><span class="lbl">Alpha</span><span class="val" style="color:{_col(metrics['alpha'])}">{_fp(metrics['alpha']*100)}</span></div>
    <div class="tb-stat"><span class="lbl">β</span><span class="val" style="color:var(--amber)">{metrics['beta']:.2f}×</span></div>
    <div class="tb-stat"><span class="lbl">Sharpe</span><span class="val">{metrics['sharpe']:.3f}</span></div>
  </div>
  <div class="tb-right">
    <span>{now_utc}</span>
    <span style="color:var(--dim)">Not investment advice</span>
  </div>
</div>

<main>

<!-- OVERVIEW -->
<div id="overview" class="section-header"><span class="section-num">01</span> Overview</div>
<div class="kpi-row kpi-row-5">{kpi1}</div>
<div class="kpi-row kpi-row-7">{kpi2}</div>

<!-- CHARTS -->
<div id="charts" class="section-header"><span class="section-num">02</span> Performance Charts</div>
<div class="card">
  <div class="card-header"><span class="card-title">Portfolio vs Benchmark — Base 100</span><span class="card-badge">inception {inc}</span></div>
  <div class="chart-wrap" id="perf"></div>
</div>
<div class="grid g2">
  <div class="card">
    <div class="card-header"><span class="card-title">Drawdown from Peak (%)</span></div>
    <div class="chart-wrap" id="drawdown"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Monthly Returns (%)</span></div>
    <div class="chart-wrap" id="monthly_returns"></div>
  </div>
</div>
<div class="grid g3">
  <div class="card">
    <div class="card-header"><span class="card-title">Rolling {CFG['roll_w']}-Day Volatility</span></div>
    <div class="chart-wrap" id="rolling_vol"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Rolling {CFG['roll_w']}-Day Beta</span></div>
    <div class="chart-wrap" id="rolling_beta"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Rolling {CFG['roll_w']}-Day Sharpe</span></div>
    <div class="chart-wrap" id="rolling_sharpe"></div>
  </div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Monthly Return Heatmap</span></div>
  <div class="chart-wrap" id="heatmap"></div>
</div>

<!-- POSITIONS -->
<div id="positions" class="section-header"><span class="section-num">03</span> Position Monitor</div>
<div class="card">
  <div class="card-header"><span class="card-title">Holdings</span><span class="card-badge">{now_utc}</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr>
        <th>Ticker</th><th>Name</th><th>Sector</th><th>Theme</th><th>Bucket</th>
        <th class="num">Qty</th><th class="num">Buy $</th><th class="num">Last $</th>
        <th class="num">Value $</th><th class="num">P&amp;L $</th><th class="num">Return</th>
        <th class="num">Weight</th><th class="num">Contrib</th>
        <th class="num">1D</th><th class="num">5D</th><th class="num">1M</th>
        <th class="num">β</th>
      </tr></thead>
      <tbody>{_positions_table(pos)}</tbody>
    </table>
  </div>
</div>
<div class="grid g2">
  <div class="card">
    <div class="card-header"><span class="card-title">Top Weights</span></div>
    <div class="chart-wrap" id="top_weights"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">P&amp;L Attribution</span></div>
    <div class="chart-wrap" id="pnl_attr"></div>
  </div>
</div>

<!-- LEDGER -->
<div id="ledger" class="section-header"><span class="section-num">04</span> Daily Ledger</div>
<div class="card">
  <div class="card-header"><span class="card-title">Daily P&amp;L — Last 40 Sessions</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr>
        <th>Date</th><th class="num">NAV</th><th class="num">Daily P&amp;L</th>
        <th class="num">Daily Ret</th><th class="num">QQQ Ret</th>
        <th class="num">Active</th><th class="num">Drawdown</th>
      </tr></thead>
      <tbody>{_ledger_table(ledger)}</tbody>
    </table>
  </div>
</div>

<!-- METRICS -->
<div id="metrics" class="section-header"><span class="section-num">05</span> Full Metrics</div>
<div class="card">
  <div class="card-header"><span class="card-title">Risk &amp; Performance Metrics</span><span class="card-badge">rf={rf_disp} · AF={CFG['af']}</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Metric</th><th class="num">Value</th><th>Notes</th></tr></thead>
      <tbody>{_metrics_table(metrics, pos)}</tbody>
    </table>
  </div>
</div>

<!-- MONTHLY -->
<div id="monthly" class="section-header"><span class="section-num">06</span> Monthly &amp; Annual Returns</div>
<div class="card">
  <div class="card-header"><span class="card-title">Return Summary by Period</span></div>
  <div class="card-body">{_monthly_table(metrics)}</div>
</div>

<!-- STRUCTURE -->
<div id="structure" class="section-header"><span class="section-num">07</span> Portfolio Structure</div>
<div class="grid g65">
  <div class="card">
    <div class="card-header"><span class="card-title">Sector Breakdown</span></div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>Sector</th><th class="num">Weight</th><th class="num">Market Value</th><th class="num">P&amp;L</th><th class="num">Positions</th></tr></thead>
        <tbody>{_sector_table(structure)}</tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Sector Allocation</span></div>
    <div class="chart-wrap" id="sector_alloc"></div>
  </div>
</div>
<div class="grid g2">
  <div class="card">
    <div class="card-header"><span class="card-title">Thematic Allocation</span></div>
    <div class="chart-wrap" id="theme_alloc"></div>
  </div>
  <div class="card card-body">
    <div class="card-header"><span class="card-title">Concentration</span></div>
    <div class="card-body">
      <table class="data-table">
        <tbody>
          <tr><td>HHI</td><td class="num strong">{structure['hhi']:.4f}</td></tr>
          <tr><td>Effective positions (1/HHI)</td><td class="num strong">{structure['eff_n']:.1f}</td></tr>
          <tr><td>Top 5 weight</td><td class="num strong">{structure['top5']*100:.1f}%</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- STRESS TEST -->
<div id="stress" class="section-header"><span class="section-num">08</span> Stress Test</div>
<div class="card">
  <div class="card-header"><span class="card-title">Stress Testing — Scenario Impact</span></div>
  <div class="chart-wrap" id="stress"></div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Stress Scenario Detail</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Scenario</th><th>Description</th><th class="num">P&amp;L Impact</th><th class="num">Return Impact</th><th class="num">NAV After</th></tr></thead>
      <tbody>{_stress_table(stress)}</tbody>
    </table>
  </div>
</div>

<!-- SCENARIOS -->
<div id="scenarios" class="section-header"><span class="section-num">09</span> Scenario Projections</div>
<div class="disclaimer"><strong>Model disclaimer:</strong> The projections below are generated via a {CFG['mc_paths']}-path Monte Carlo simulation using the portfolio's own historical return distribution. These are <strong>model-based scenario envelopes, not forecasts or guarantees</strong>. Past distribution of returns does not predict future performance.</div>
<div class="card">
  <div class="card-header"><span class="card-title">12M Scenario Envelope — Bull / Base / Bear + Monte Carlo</span><span class="card-badge">bootstrap · historical distribution</span></div>
  <div class="chart-wrap" id="forecast"></div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Projection Summary by Horizon</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Horizon</th><th class="num">Start NAV</th><th class="num red">P5 (Bear)</th><th class="num">P25</th><th class="num green">Median</th><th class="num">P75</th><th class="num amber">P95 (Bull)</th></tr></thead>
      <tbody>{_forecast_table(fcast_sum)}</tbody>
    </table>
  </div>
</div>

<!-- INTELLIGENCE -->
<div id="intelligence" class="section-header"><span class="section-num">10</span> Portfolio Intelligence</div>
<div class="card">
  <div class="card-header"><span class="card-title">Auto-generated Intelligence · {now_utc}</span></div>
  <div class="card-body"><div class="intel-grid">{intel_html}</div></div>
</div>

<!-- NEWS -->
<div id="news" class="section-header"><span class="section-num">11</span> Portfolio News</div>
<div class="card">
  <div class="card-header"><span class="card-title">Latest News · Non-blocking · Build-time snapshot</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Ticker</th><th>Headline</th><th>Source</th><th>Published</th></tr></thead>
      <tbody>{_news_table(news)}</tbody>
    </table>
  </div>
</div>

</main>

<footer>
  <div>{_esc(pname)} · {_esc(CFG['school'])} · {_esc(CFG['course'])}<br>
  Inception: {inc} · AUM₀: {_fc(init_aum,0)} · Benchmark: {CFG['benchmark']} · rf={rf_disp}</div>
  <div>Generated: {now_utc}<br>Data: Yahoo Finance via yfinance · Not investment advice</div>
</footer>

</div><!-- .content -->
</div><!-- .shell -->

<script>
const CHARTS = {chart_json};
const PCFG   = {{responsive:true,displayModeBar:false,scrollZoom:false}};
const obs = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    if (e.isIntersecting) {{
      const id = e.target.id;
      if (CHARTS[id]) {{
        try {{ Plotly.newPlot(id, CHARTS[id].data, CHARTS[id].layout, PCFG); }}
        catch(err) {{ console.warn('Chart error:', id, err); }}
      }}
      obs.unobserve(e.target);
    }}
  }});
}}, {{rootMargin:'300px'}});
Object.keys(CHARTS).forEach(id => {{
  const el = document.getElementById(id);
  if (el) obs.observe(el);
}});
const navLinks = document.querySelectorAll('.nav-link');
window.addEventListener('scroll', () => {{
  let cur = '';
  document.querySelectorAll('[id]').forEach(s => {{
    if (window.scrollY >= s.offsetTop - 130) cur = s.id;
  }});
  navLinks.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + cur));
}}, {{passive:true}});
</script>
</body>
</html>"""


# ================================================================
# ORCHESTRATION
# ================================================================
def main():
    print("Loading holdings...")
    holdings = load_holdings(ROOT / "holdings.csv")
    init_aum = float(holdings["cost_basis"].sum())
    print(f"  {len(holdings)} holdings  |  AUM₀ ${init_aum:,.2f}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tickers = holdings["ticker"].tolist() + [CFG["benchmark"], CFG["bench2"]]
    print(f"\nDownloading prices: {CFG['inception']} → {today}")
    prices = download_prices(tickers, CFG["inception"], today)

    print("\nBuilding analytics...")
    frame    = build_frame(prices, holdings)
    metrics  = compute_metrics(frame)
    pos      = compute_positions(frame, holdings)
    struct   = compute_structure(pos)
    ledger   = build_ledger(frame)
    heatmap  = build_heatmap(metrics["mp"])
    stress   = build_stress(pos, metrics)
    fsum, fp = build_forecast(frame, metrics)

    try:
        news = build_news(holdings)
        print(f"  News: {len(news)} articles")
    except Exception:
        news = pd.DataFrame(columns=["ticker","title","source","published_at","url"])
        print("  News: skipped (non-blocking)")

    print("\nBuilding charts and intelligence...")
    intel  = build_intelligence(metrics, pos, struct, stress)
    charts = make_charts(frame, metrics, pos, struct, heatmap, stress, fp)

    print("\nGenerating HTML...")
    html = generate_html(
        holdings, frame, metrics, pos, struct, ledger,
        heatmap, stress, fsum, fp, news, intel, charts,
    )
    OUT_HTML.write_text(html, encoding="utf-8")

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "portfolio_name": CFG["portfolio_name"],
            "benchmark":      CFG["benchmark"],
            "rf":             CFG["rf"],
            "inception":      CFG["inception"],
        },
        "overview": {
            "init_aum":     init_aum,
            "current_nav":  metrics["current_nav"],
            "total_pnl":    metrics["total_pnl"],
            "total_return": metrics["total_return"],
            "daily_pnl":    metrics["daily_pnl"],
            "daily_return": metrics["daily_return"],
        },
        "risk": {
            "alpha":   metrics["alpha"],
            "beta":    metrics["beta"],
            "sharpe":  metrics["sharpe"],
            "sortino": metrics["sortino"],
            "mdd":     metrics["mdd"],
            "vol":     metrics["vol"],
            "ir":      metrics["ir"],
            "te":      metrics["te"],
            "var95":   metrics["var95"],
            "cvar95":  metrics["cvar95"],
        },
        "positions": pos[[
            "ticker","name","sector","risk_bucket",
            "latest_price","market_value","pnl","ret","weight","contribution","beta",
        ]].to_dict(orient="records"),
    }
    OUT_JSON.write_text(_dumps(snapshot, ), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  HTML     → {OUT_HTML}")
    print(f"  Snapshot → {OUT_JSON}")
    print(f"  NAV:     ${metrics['current_nav']:,.2f}  |  Return: {metrics['total_return']*100:+.2f}%")
    print(f"  Sharpe:  {metrics['sharpe']:.3f}  |  Beta: {metrics['beta']:.3f}")
    print(f"{'='*60}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nBuild failed: {exc}")
        traceback.print_exc()
        raise
