from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.io as pio
import uuid
from flask import send_file
import io


app = Flask(__name__)

# ----------------------------
# Universe
# ----------------------------
TICKERS = [
    "AI.PA","AIR.PA","ALO.PA","ACA.PA","BNP.PA","EN.PA","CAP.PA","CA.PA","BN.PA","DSY.PA",
    "ENGI.PA","EL.PA","ERF.PA","RMS.PA","KER.PA","LR.PA","OR.PA","MC.PA","ML.PA","ORA.PA",
    "RI.PA","PUB.PA","RNO.PA","SAF.PA","SGO.PA","SAN.PA","SU.PA","GLE.PA","STLA.PA","STM.PA",
    "TEP.PA","HO.PA","TTE.PA","URW.AS","VIE.PA","DG.PA","VIV.PA","WLN.PA"
]

TICKER_NAMES = {
    "AI.PA":"Air Liquide","AIR.PA":"Airbus","ALO.PA":"Alstom","ACA.PA":"Crédit Agricole","BNP.PA":"BNP Paribas",
    "EN.PA":"Bouygues","CAP.PA":"Capgemini","CA.PA":"Carrefour","BN.PA":"Danone","DSY.PA":"Dassault Systèmes",
    "ENGI.PA":"Engie","EL.PA":"EssilorLuxottica","ERF.PA":"Eurofins Scientific","RMS.PA":"Hermès","KER.PA":"Kering",
    "LR.PA":"Legrand","OR.PA":"L'Oréal","MC.PA":"LVMH","ML.PA":"Michelin","ORA.PA":"Orange","RI.PA":"Pernod Ricard",
    "PUB.PA":"Publicis","RNO.PA":"Renault","SAF.PA":"Safran","SGO.PA":"Saint-Gobain","SAN.PA":"Sanofi",
    "SU.PA":"Schneider Electric","GLE.PA":"Société Générale","STLA.PA":"Stellantis","STM.PA":"STMicroelectronics",
    "TEP.PA":"Téléperformance","HO.PA":"Thales","TTE.PA":"TotalEnergies","URW.AS":"Unibail-Rodamco-Westfield",
    "VIE.PA":"Veolia","DG.PA":"Vinci","VIV.PA":"Vivendi","WLN.PA":"Worldline"
}

# ----------------------------
# Walk-forward settings
# ----------------------------
WALK_FORWARD_SPLIT = 0.70
MIN_BLOCK_DAYS = 60          # min jours par bloc OOS
MA_BUFFER_DAYS = 60          # marge de sécurité après MA_long
MIN_OOS_DAYS = 80            # min jours OOS total
MIN_IS_DAYS = 80             # min jours IS total

# ----------------------------
# 1-couple candidates around 20/100 (15)
# ----------------------------
MA_CANDIDATES_20_100 = [
    (15, 90), (15, 100), (15, 110),
    (18, 90), (18, 100), (18, 110),
    (20, 90), (20, 100), (20, 110),
    (22, 90), (22, 100), (22, 110),
    (25, 90), (25, 100), (25, 110),
]

# ----------------------------
# 3-couple candidates (15 triples)
# ----------------------------
CANDIDATE_MA_TRIPLES = [
    [(10, 90), (20, 100), (50, 200)],
    [(8, 80), (16, 96), (40, 160)],
    [(12, 84), (24, 108), (60, 200)],
    [(10, 70), (20, 90), (50, 180)],
    [(15, 90), (30, 120), (60, 200)],
    [(10, 60), (20, 100), (50, 150)],
    [(5, 50), (10, 100), (20, 200)],
    [(7, 63), (14, 112), (28, 224)],
    [(9, 81), (18, 108), (45, 180)],
    [(20, 80), (40, 120), (80, 200)],
    [(12, 60), (24, 120), (48, 200)],
    [(14, 98), (28, 126), (70, 210)],
    [(10, 100), (20, 120), (50, 250)],
    [(6, 54), (12, 108), (30, 200)],
    [(16, 64), (32, 128), (64, 256)],
]

# In-memory cache for diagnostics pages
RUN_CACHE = {}  # run_id -> dict(payload)


# ============================
# Data
# ============================
def period_to_yf(period_ui: str) -> str:
    return {"1Y":"1y","2Y":"2y","5Y":"5y","10Y":"10y","15Y":"15y"}.get(period_ui.upper(), "15y")


def get_prices_multi(tickers, period: str) -> pd.DataFrame:
    data = yf.download(tickers, period=period, progress=False, group_by="column", auto_adjust=False)
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        else:
            close = pd.DataFrame()
    else:
        close = data["Close"].to_frame()

    close = close.dropna(how="all")
    return close


# ============================
# Signals / Weights
# ============================
def compute_ma_signal(prices: pd.DataFrame, short_w: int, long_w: int) -> pd.DataFrame:
    ma_s = prices.rolling(short_w).mean()
    ma_l = prices.rolling(long_w).mean()
    raw = (ma_s > ma_l).astype(int)
    return raw.shift(1).fillna(0)  # avoid look-ahead


def normalize_weights_fully_invested(exposure: pd.DataFrame) -> pd.DataFrame:
    """
    no cash:
    - si exposition > 0 : normalise pour que la ligne somme à 1
    - si exposition == 0 : conserve les poids de la veille (fallback), et si c'est le 1er jour -> equal-weight
    """
    exposure = exposure.clip(lower=0)

    row_sum = exposure.sum(axis=1)
    weights = exposure.div(row_sum.replace(0, np.nan), axis=0)

    # fallback = poids de la veille quand pas de signal
    weights = weights.ffill()

    # si tout début de série (rien à forward-fill), on met equal-weight
    weights = weights.fillna(1.0 / exposure.shape[1])

    # sécurité numérique
    weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / exposure.shape[1])

    return weights



def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    w_prev = weights.shift(1)
    to = (weights - w_prev).abs().sum(axis=1)
    return to.fillna(0)



# ============================
# Backtests
# ============================
def backtest_trend_one_pair_no_cash(prices: pd.DataFrame, short_w: int, long_w: int):
    prices = prices.dropna(how="all")
    returns = prices.pct_change().fillna(0)

    if prices.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame()

    exposure = compute_ma_signal(prices, short_w, long_w)
    weights = normalize_weights_fully_invested(exposure)

    port_ret = (weights * returns).sum(axis=1)
    port_equity = (1 + port_ret).cumprod()

    n = prices.shape[1]
    ew = np.repeat(1.0 / n, n)
    bh_ret = (returns * ew).sum(axis=1)
    bh_equity = (1 + bh_ret).cumprod()

    res = pd.DataFrame({
        "port_ret": port_ret,
        "port_equity": port_equity,
        "bh_ret": bh_ret,
        "bh_equity": bh_equity,
    })
    return res, weights


def backtest_trend_triple_no_cash(prices: pd.DataFrame, ma_triple):
    prices = prices.dropna(how="all")
    returns = prices.pct_change().fillna(0)

    if prices.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame()

    signals = [compute_ma_signal(prices, s, l) for (s, l) in ma_triple]
    exposure = sum(signals) / len(signals)
    weights = normalize_weights_fully_invested(exposure)

    port_ret = (weights * returns).sum(axis=1)
    port_equity = (1 + port_ret).cumprod()

    n = prices.shape[1]
    ew = np.repeat(1.0 / n, n)
    bh_ret = (returns * ew).sum(axis=1)
    bh_equity = (1 + bh_ret).cumprod()

    res = pd.DataFrame({
        "port_ret": port_ret,
        "port_equity": port_equity,
        "bh_ret": bh_ret,
        "bh_equity": bh_equity,
    })
    return res, weights


# ============================
# Metrics
# ============================
def annualize_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    total = equity.iloc[-1] / equity.iloc[0] - 1
    return (1 + total) ** (252 / len(equity)) - 1

def annualize_vol(ret: pd.Series) -> float:
    return float(ret.std() * np.sqrt(252))

def sharpe(ann_ret: float, ann_vol: float) -> float:
    if ann_vol == 0 or np.isnan(ann_vol):
        return float("nan")
    return float(ann_ret / ann_vol)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())

def beta_vs_benchmark(strat_ret: pd.Series, bench_ret: pd.Series) -> float:
    var_b = np.var(bench_ret, ddof=1)
    if var_b == 0 or np.isnan(var_b):
        return float("nan")
    cov = np.cov(strat_ret, bench_ret, ddof=1)[0, 1]
    return float(cov / var_b)

def fmt_pct(x: float) -> str:
    return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.2%}"

def fmt_num(x: float, d=2) -> str:
    return "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{d}f}"


def kpis_from_res(res: pd.DataFrame):
    total = res["port_equity"].iloc[-1] / res["port_equity"].iloc[0] - 1
    ann = annualize_return(res["port_equity"])
    vol = annualize_vol(res["port_ret"])
    sh = sharpe(ann, vol)
    mdd = max_drawdown(res["port_equity"])
    return total, ann, vol, sh, mdd


def kpis_bh_from_res(res: pd.DataFrame):
    total = res["bh_equity"].iloc[-1] / res["bh_equity"].iloc[0] - 1
    ann = annualize_return(res["bh_equity"])
    vol = annualize_vol(res["bh_ret"])
    sh = sharpe(ann, vol)
    mdd = max_drawdown(res["bh_equity"])
    return total, ann, vol, sh, mdd


def split_oos_blocks(oos_df: pd.DataFrame, n_blocks=3):
    """
    Split OOS into 3 consecutive blocks.
    Returns list of DataFrames.
    """
    if oos_df is None or len(oos_df) == 0:
        return [pd.DataFrame()] * n_blocks

    if len(oos_df) < n_blocks * MIN_BLOCK_DAYS:
        return list(np.array_split(oos_df, n_blocks))

    block_size = len(oos_df) // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else len(oos_df)
        blocks.append(oos_df.iloc[start:end])
    return blocks


def stability_metrics(oos_df: pd.DataFrame):
    """
    Returns block sharpes + #valid blocks (>= MIN_BLOCK_DAYS) + mean/std across valid blocks.
    """
    blocks = split_oos_blocks(oos_df, 3)

    sharpes = []
    cagrs = []
    mdds = []
    valid = 0

    for b in blocks:
        if b is None or len(b) < MIN_BLOCK_DAYS:
            sharpes.append(np.nan)
            cagrs.append(np.nan)
            mdds.append(np.nan)
            continue

        _, ann, vol, sh, mdd = kpis_from_res(b)
        sharpes.append(sh)
        cagrs.append(ann)
        mdds.append(mdd)
        valid += 1

    sharpe_mean = float(np.nanmean(sharpes)) if valid > 0 else float("nan")
    sharpe_std = float(np.nanstd(sharpes)) if valid > 1 else float("nan")

    return {
        "block_sizes": [int(len(b)) for b in blocks],
        "valid_blocks": int(valid),
        "block_sharpes": sharpes,
        "block_cagrs": cagrs,
        "block_mdds": mdds,
        "sharpe_mean_blocks": sharpe_mean,
        "sharpe_std_blocks": sharpe_std,
    }


# ============================
# Feasibility filters (short periods)
# ============================
def enough_history(prices: pd.DataFrame, required_days: int) -> bool:
    return len(prices) >= required_days


def feasible_pairs(prices: pd.DataFrame, pairs):
    ok = []
    for s, l in pairs:
        required = l + MA_BUFFER_DAYS
        if enough_history(prices, required):
            ok.append((s, l))
    return ok


def feasible_triples(prices: pd.DataFrame, triples):
    ok = []
    for tri in triples:
        max_long = max([l for (_, l) in tri])
        required = max_long + MA_BUFFER_DAYS
        if enough_history(prices, required):
            ok.append(tri)
    return ok


# ============================
# Walk-forward selection
# ============================
def walk_forward_select_best_pair(prices: pd.DataFrame, pairs):
    prices = prices.dropna(how="all")
    if prices.empty:
        return None, pd.DataFrame(), None, 0, 0

    pairs_ok = feasible_pairs(prices, pairs)
    tested = 0
    total = len(pairs)

    dates = prices.index
    split_idx = int(len(dates) * WALK_FORWARD_SPLIT)
    split_idx = max(MIN_IS_DAYS, min(split_idx, len(dates) - MIN_OOS_DAYS))
    split_date = dates[split_idx]

    rows = []
    for (s, l) in pairs_ok:
        res, w = backtest_trend_one_pair_no_cash(prices, s, l)
        if res.empty:
            continue

        is_res = res.loc[res.index < split_date]
        oos_res = res.loc[res.index >= split_date]

        if len(is_res) < MIN_IS_DAYS or len(oos_res) < MIN_OOS_DAYS:
            continue

        _, is_ann, is_vol, is_sh, is_mdd = kpis_from_res(is_res)
        _, oos_ann, oos_vol, oos_sh, oos_mdd = kpis_from_res(oos_res)

        w_oos = w.loc[oos_res.index]
        to = turnover_from_weights(w_oos)
        to_mean = float(to.mean())
        to_p95 = float(np.nanpercentile(to, 95))

        rows.append({
            "pair": f"{s}/{l}",
            "short": s,
            "long": l,
            "is_sharpe": is_sh,
            "is_ann_return": is_ann,
            "is_ann_vol": is_vol,
            "is_mdd": is_mdd,
            "oos_sharpe": oos_sh,
            "oos_ann_return": oos_ann,
            "oos_ann_vol": oos_vol,
            "oos_mdd": oos_mdd,
            "oos_turnover_mean": to_mean,
            "oos_turnover_p95": to_p95,
        })
        tested += 1

    diag = pd.DataFrame(rows)
    if diag.empty:
        return None, pd.DataFrame(), split_date, tested, total

    diag = diag.sort_values("is_sharpe", ascending=False).reset_index(drop=True)
    best = (int(diag.loc[0, "short"]), int(diag.loc[0, "long"]))
    return best, diag, split_date, tested, total


def walk_forward_select_best_triple(prices: pd.DataFrame, triples):
    prices = prices.dropna(how="all")
    if prices.empty:
        return None, pd.DataFrame(), None, 0, 0

    triples_ok = feasible_triples(prices, triples)
    tested = 0
    total = len(triples)

    dates = prices.index
    split_idx = int(len(dates) * WALK_FORWARD_SPLIT)
    split_idx = max(MIN_IS_DAYS, min(split_idx, len(dates) - MIN_OOS_DAYS))
    split_date = dates[split_idx]

    rows = []
    for tri in triples_ok:
        res, w = backtest_trend_triple_no_cash(prices, tri)
        if res.empty:
            continue

        is_res = res.loc[res.index < split_date]
        oos_res = res.loc[res.index >= split_date]

        if len(is_res) < MIN_IS_DAYS or len(oos_res) < MIN_OOS_DAYS:
            continue

        _, is_ann, is_vol, is_sh, is_mdd = kpis_from_res(is_res)
        _, oos_ann, oos_vol, oos_sh, oos_mdd = kpis_from_res(oos_res)

        w_oos = w.loc[oos_res.index]
        to = turnover_from_weights(w_oos)
        to_mean = float(to.mean())
        to_p95 = float(np.nanpercentile(to, 95))

        label = " | ".join([f"{s}/{l}" for s, l in tri])
        rows.append({
            "triple": label,
            "is_sharpe": is_sh,
            "is_ann_return": is_ann,
            "is_ann_vol": is_vol,
            "is_mdd": is_mdd,
            "oos_sharpe": oos_sh,
            "oos_ann_return": oos_ann,
            "oos_ann_vol": oos_vol,
            "oos_mdd": oos_mdd,
            "oos_turnover_mean": to_mean,
            "oos_turnover_p95": to_p95,
        })
        tested += 1

    diag = pd.DataFrame(rows)
    if diag.empty:
        return None, pd.DataFrame(), split_date, tested, total

    diag = diag.sort_values("is_sharpe", ascending=False).reset_index(drop=True)
    best_label = diag.loc[0, "triple"]

    best_triple = None
    for tri in triples:
        label = " | ".join([f"{s}/{l}" for s, l in tri])
        if label == best_label:
            best_triple = tri
            break

    return best_triple, diag, split_date, tested, total


# ============================
# Routes
# ============================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", tickers=TICKERS, ticker_names=TICKER_NAMES)


@app.route("/trend", methods=["POST"])
def trend():
    selected = request.form.getlist("tickers")
    if not selected:
        return render_template("index.html", tickers=TICKERS, ticker_names=TICKER_NAMES,
                               message="Merci de sélectionner au moins un actif.")

    period_ui = request.form.get("period", "15Y")
    prices = get_prices_multi(selected, period_to_yf(period_ui))
    if prices.empty:
        return render_template("index.html", tickers=TICKERS, ticker_names=TICKER_NAMES,
                               message="Aucune donnée téléchargée. Réessaie plus tard.")

    # ---- Select 1-couple
    best_pair, diag1, split_date1, tested1, total1 = walk_forward_select_best_pair(prices, MA_CANDIDATES_20_100)
    if best_pair is None:
        return render_template(
            "index.html", tickers=TICKERS, ticker_names=TICKER_NAMES,
            message="Impossible de calibrer 1-couple (historique insuffisant / OOS trop court). Essaie une période plus longue."
        )

    s1, l1 = best_pair
    res1, w1 = backtest_trend_one_pair_no_cash(prices, s1, l1)

    # ---- Select 3-couple
    best_triple, diag3, split_date3, tested3, total3 = walk_forward_select_best_triple(prices, CANDIDATE_MA_TRIPLES)
    if best_triple is None:
        return render_template(
            "index.html", tickers=TICKERS, ticker_names=TICKER_NAMES,
            message="Impossible de calibrer 3-couples (historique insuffisant / OOS trop court). Essaie une période plus longue."
        )

    res3, w3 = backtest_trend_triple_no_cash(prices, best_triple)

    # Use the same split date for fair compare: take the later one, then align OOS
    split_final = max(split_date1, split_date3)

    oos1 = res1.loc[res1.index >= split_final].copy()
    oos3 = res3.loc[res3.index >= split_final].copy()

    common_idx = oos1.index.intersection(oos3.index)
    oos1 = oos1.loc[common_idx]
    oos3 = oos3.loc[common_idx]
    w1_oos = w1.loc[common_idx]
    w3_oos = w3.loc[common_idx]

    if len(oos1) < MIN_OOS_DAYS or len(oos3) < MIN_OOS_DAYS:
        return render_template(
            "index.html", tickers=TICKERS, ticker_names=TICKER_NAMES,
            message="OOS trop court après alignement. Choisis une période plus longue (5Y+ recommandé)."
        )

    # ---- KPIs OOS
    t1_total, t1_ann, t1_vol, t1_sh, t1_mdd = kpis_from_res(oos1)
    t3_total, t3_ann, t3_vol, t3_sh, t3_mdd = kpis_from_res(oos3)

    # B&H OOS (same index; use oos1)
    bh_total, bh_ann, bh_vol, bh_sh, bh_mdd = kpis_bh_from_res(oos1)

    # beta / alpha vs B&H
    beta1 = beta_vs_benchmark(oos1["port_ret"], oos1["bh_ret"])
    alpha1 = t1_ann - beta1 * bh_ann if np.isfinite(beta1) else float("nan")

    beta3 = beta_vs_benchmark(oos3["port_ret"], oos3["bh_ret"])
    alpha3 = t3_ann - beta3 * bh_ann if np.isfinite(beta3) else float("nan")

    # turnover
    to1 = turnover_from_weights(w1_oos)
    to3 = turnover_from_weights(w3_oos)
    to1_mean, to1_p95 = float(to1.mean()), float(np.nanpercentile(to1, 95))
    to3_mean, to3_p95 = float(to3.mean()), float(np.nanpercentile(to3, 95))

    # stability in 3 OOS blocks
    stab1 = stability_metrics(oos1)
    stab3 = stability_metrics(oos3)

    # equity curve plot (full period), with split line
    df_eq = pd.DataFrame({
        "Date": res1.index,
        "Trend_1couple": res1["port_equity"].values,
        "Trend_3couples": res3.reindex(res1.index)["port_equity"].values,
        "BuyHold": res1["bh_equity"].values
    }).dropna()

    total_days = int(len(df_eq))
    oos_days = int(len(common_idx))

    fig_eq = px.line(
        df_eq,
        x="Date",
        y=["Trend_1couple", "Trend_3couples", "BuyHold"],
        title=f"Equity curves — split OOS {split_final.strftime('%Y-%m-%d')}"
    )
    fig_eq.add_vline(x=split_final, line_dash="dash", opacity=0.7)
    graphJSON_equity = pio.to_json(fig_eq)

    best_triple_label = " | ".join([f"{s}/{l}" for s, l in best_triple])

    stats = {
        "period_ui": period_ui,
        "split_rule": "70% calibration / 30% test",
        "split_date": split_final.strftime("%Y-%m-%d"),
        "total_days": total_days,
        "oos_days": oos_days,

        "best_pair": f"{s1}/{l1}",
        "best_triple": best_triple_label,

        # Trend 1-couple
        "t1_total": fmt_pct(t1_total),
        "t1_ann": fmt_pct(t1_ann),
        "t1_vol": fmt_pct(t1_vol),
        "t1_sh": fmt_num(t1_sh, 2),
        "t1_mdd": fmt_pct(t1_mdd),
        "t1_beta": fmt_num(beta1, 2),
        "t1_alpha": fmt_pct(alpha1),
        "t1_to_mean": fmt_num(to1_mean, 3),
        "t1_to_p95": fmt_num(to1_p95, 3),
        "t1_sh_blocks_mean": fmt_num(stab1["sharpe_mean_blocks"], 2),
        "t1_sh_blocks_std": fmt_num(stab1["sharpe_std_blocks"], 2),
        "t1_valid_blocks": stab1["valid_blocks"],
        "t1_sh_b1": fmt_num(stab1["block_sharpes"][0], 2),
        "t1_sh_b2": fmt_num(stab1["block_sharpes"][1], 2),
        "t1_sh_b3": fmt_num(stab1["block_sharpes"][2], 2),

        # Trend 3-couples
        "t3_total": fmt_pct(t3_total),
        "t3_ann": fmt_pct(t3_ann),
        "t3_vol": fmt_pct(t3_vol),
        "t3_sh": fmt_num(t3_sh, 2),
        "t3_mdd": fmt_pct(t3_mdd),
        "t3_beta": fmt_num(beta3, 2),
        "t3_alpha": fmt_pct(alpha3),
        "t3_to_mean": fmt_num(to3_mean, 3),
        "t3_to_p95": fmt_num(to3_p95, 3),
        "t3_sh_blocks_mean": fmt_num(stab3["sharpe_mean_blocks"], 2),
        "t3_sh_blocks_std": fmt_num(stab3["sharpe_std_blocks"], 2),
        "t3_valid_blocks": stab3["valid_blocks"],
        "t3_sh_b1": fmt_num(stab3["block_sharpes"][0], 2),
        "t3_sh_b2": fmt_num(stab3["block_sharpes"][1], 2),
        "t3_sh_b3": fmt_num(stab3["block_sharpes"][2], 2),

        # Benchmark
        "bh_total": fmt_pct(bh_total),
        "bh_ann": fmt_pct(bh_ann),
        "bh_vol": fmt_pct(bh_vol),
        "bh_sh": fmt_num(bh_sh, 2),
        "bh_mdd": fmt_pct(bh_mdd),

        # coverage
        "tested1": tested1,
        "total1": total1,
        "tested3": tested3,
        "total3": total3,
        "min_block_days": MIN_BLOCK_DAYS
    }

    run_id = str(uuid.uuid4())
    RUN_CACHE[run_id] = {
        "period_ui": period_ui,
        "split_date": stats["split_date"],
        "best_pair": stats["best_pair"],
        "best_triple": stats["best_triple"],
        "diag1": diag1,
        "diag3": diag3,

        # ✅ pour le download
        "w1_oos": w1_oos,
        "w3_oos": w3_oos,
    }

    return render_template(
        "result_compare.html",
        stats=stats,
        graphJSON_equity=graphJSON_equity,
        run_id=run_id
    )

@app.route("/results/<run_id>", methods=["GET"])
def results(run_id):
    payload = RUN_CACHE.get(run_id)
    if payload is None:
        return redirect(url_for("index"))

    # Il faut que payload contienne aussi stats + graphJSON_equity
    stats = payload.get("stats")
    graphJSON_equity = payload.get("graphJSON_equity")

    if stats is None or graphJSON_equity is None:
        return redirect(url_for("index"))

    return render_template(
        "result_compare.html",
        stats=stats,
        graphJSON_equity=graphJSON_equity,
        run_id=run_id
    )

@app.route("/diagnostics_1/<run_id>", methods=["GET"])
def diagnostics_1(run_id):
    payload = RUN_CACHE.get(run_id)
    if payload is None:
        return redirect(url_for("index"))

    diag = payload["diag1"].copy()
    best_pair = payload["best_pair"]

    if diag.empty:
        return redirect(url_for("index"))

    diag["best"] = diag["pair"].apply(lambda x: "yes" if x == best_pair else "")

    table = diag.copy()
    for c in ["is_ann_return","is_ann_vol","is_mdd","oos_ann_return","oos_ann_vol","oos_mdd"]:
        table[c] = table[c].apply(fmt_pct)
    for c in ["is_sharpe","oos_sharpe"]:
        table[c] = table[c].apply(lambda x: fmt_num(x, 2))
    table["oos_turnover_mean"] = table["oos_turnover_mean"].apply(lambda x: fmt_num(x, 3))
    table["oos_turnover_p95"] = table["oos_turnover_p95"].apply(lambda x: fmt_num(x, 3))

    diag_num = payload["diag1"].copy()
    fig_is = px.bar(diag_num, x="pair", y="is_sharpe", title="1-couple — Sharpe in-sample (calibration)")
    graphJSON_is = pio.to_json(fig_is)

    fig_sc = px.scatter(diag_num, x="is_sharpe", y="oos_sharpe", text="pair",
                        title="1-couple — Robustesse (Sharpe OOS vs IS)")
    graphJSON_scatter = pio.to_json(fig_sc)

    return render_template(
        "diagnostics_1.html",
        run_id=run_id,
        period_ui=payload["period_ui"],
        split_date=payload["split_date"],
        best=best_pair,
        table=table.to_dict(orient="records"),
        graphJSON_is=graphJSON_is,
        graphJSON_scatter=graphJSON_scatter
    )


@app.route("/diagnostics_3/<run_id>", methods=["GET"])
def diagnostics_3(run_id):
    payload = RUN_CACHE.get(run_id)
    if payload is None:
        return redirect(url_for("index"))

    diag = payload["diag3"].copy()
    best_triple = payload["best_triple"]

    if diag.empty:
        return redirect(url_for("index"))

    diag["best"] = diag["triple"].apply(lambda x: "yes" if x == best_triple else "")

    table = diag.copy()
    for c in ["is_ann_return","is_ann_vol","is_mdd","oos_ann_return","oos_ann_vol","oos_mdd"]:
        table[c] = table[c].apply(fmt_pct)
    for c in ["is_sharpe","oos_sharpe"]:
        table[c] = table[c].apply(lambda x: fmt_num(x, 2))
    table["oos_turnover_mean"] = table["oos_turnover_mean"].apply(lambda x: fmt_num(x, 3))
    table["oos_turnover_p95"] = table["oos_turnover_p95"].apply(lambda x: fmt_num(x, 3))

    diag_num = payload["diag3"].copy()
    fig_is = px.bar(diag_num, x="triple", y="is_sharpe", title="3-couples — Sharpe in-sample (calibration)")
    graphJSON_is = pio.to_json(fig_is)

    fig_sc = px.scatter(diag_num, x="is_sharpe", y="oos_sharpe", text="triple",
                        title="3-couples — Robustesse (Sharpe OOS vs IS)")
    graphJSON_scatter = pio.to_json(fig_sc)

    return render_template(
        "diagnostics_3.html",
        run_id=run_id,
        period_ui=payload["period_ui"],
        split_date=payload["split_date"],
        best=best_triple,
        table=table.to_dict(orient="records"),
        graphJSON_is=graphJSON_is,
        graphJSON_scatter=graphJSON_scatter
    )

@app.route("/download_weights/<run_id>/<model>", methods=["GET"])
def download_weights(run_id, model):
    """
    Télécharge les poids au format CSV.
    model: "pair" ou "triple"
    """
    payload = RUN_CACHE.get(run_id)
    if payload is None:
        return redirect(url_for("index"))

    if model == "pair":
        w = payload.get("w1_oos")
        label = payload.get("best_pair", "best_pair")
    elif model == "triple":
        w = payload.get("w3_oos")
        label = payload.get("best_triple", "best_triple")
    else:
        return redirect(url_for("index"))

    if w is None or w.empty:
        return redirect(url_for("index"))

    # ✅ daily signal = dernier jour dispo
    last_date = w.index.max()
    w_last = w.loc[last_date].copy()

    df_out = pd.DataFrame({
        "date": [last_date] * len(w_last.index),
        "ticker": w_last.index.astype(str),
        "weight": w_last.values.astype(float),
    })

    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)

    safe_label = str(label).replace("/", "-").replace(" | ", "_").replace(" ", "")
    filename = f"weights_today_{model}_{safe_label}.csv"

    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )
