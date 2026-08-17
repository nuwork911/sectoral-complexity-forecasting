"""
=============================================================================
Sectoral complexity and directional forecasting -- analysis pipeline (v2)
=============================================================================

Yuvraj Singh -- KF7029 MSc Artificial Intelligence, Northumbria University
Supervisor: Dr Honglei Li

Run from the folder that holds merged_data.csv and ticker_sectors.xlsx:

    python pipeline.py

Outputs (all under results/):
    sci_by_sector.csv               five SCI metrics + index + band, per sector
    sci_hurst_sensitivity.csv       SCI under both Hurst sign conventions
    predictions.parquet             EVERY out-of-sample prediction (see note 1)
    performance_by_model_sector.csv pooled metrics per (model, sector)
    performance_by_fold.csv         per-fold metrics  (powers the RQ1 test)
    lstm_seed_stability.csv         per-seed LSTM metrics, mean and SD
    auc_bootstrap_ci.csv            date-block bootstrap CIs for every cell
    mcnemar_pairs.csv               all model pairs x all sectors, Holm-adjusted
    rq_summary.json                 every test statistic in one place
    robustness_*.csv                per-ticker split, Hurst min_n, lag-0 feature
    numbers.tex                     \\renewcommand macros for the thesis
    tab_*.tex                       LaTeX table bodies for the thesis

-----------------------------------------------------------------------------
WHAT CHANGED FROM v1, AND WHY (read this before you touch anything)
-----------------------------------------------------------------------------
1.  PREDICTIONS ARE PERSISTED.  v1 threw away every probability, so confidence
    intervals, McNemar and per-fold tests were impossible without a full
    re-run.  Everything downstream is now derived from predictions.parquet,
    which makes the metrics reproducible from one artefact.

2.  THE LSTM IS ACTUALLY TRAINED.  v1 ran `epochs` FULL-BATCH gradient steps
    (8 Adam updates from random init), so it measured a near-random projection
    of the features.  It is now minibatch-trained for up to LSTM_EPOCHS epochs
    with the best epoch chosen on a held-out tail of the training partition.

3.  THE LSTM IS SEEDED AND REPEATED.  torch never saw RANDOM_STATE in v1, so
    the headline RQ3 result was one unreproducible draw.  It now runs
    N_SEEDS_LSTM times; the tables carry mean and SD across seeds.

4.  COMMON EVALUATION SUPPORT.  v1 built LSTM sequences inside each test fold,
    dropping seq_len rows per ticker per fold, so the LSTM was scored on a
    different sample than the tabular models (70,610 vs 74,290 in Technology)
    and paired tests were invalid.  Sequences now reach back across the
    train/test boundary, and all headline metrics are computed on the
    intersection of keys available to every model.

5.  SEQUENCE WINDOWS END AT t.  v1 used rows t-seq_len..t-1 to predict y_t,
    so the LSTM never saw day t while the tabular models did.  Fixed.

6.  FEATURE FORMULA MATCHES THE THESIS.  v1 computed MA_w / P_t - 1; the
    methodology defines P_t / MA_w - 1.  Fixed (they are monotonically
    related, so the effect on results is small but the mismatch is not
    defensible in a viva).

7.  TWO BASELINES.  v1's majority-class baseline used the TEST labels, i.e.
    the best constant predictor in hindsight.  The train-partition baseline is
    now reported alongside it.

8.  ADAPTIVE SELECTION IS EVALUATED OUT OF SAMPLE.  v1 chose the band-to-model
    map on the same AUCs it then scored, so the "uplift" was non-negative by
    construction.  The map is now fitted on early folds and scored on late
    folds; the in-sample number is still reported, explicitly as an upper
    bound.

9.  RQ1 HAS ENOUGH BLOCKS.  v1's Friedman had 4 blocks and 11 treatments, at
    which the chi-square approximation is unreliable.  The primary test now
    uses per-fold AUC (model x fold = 20 blocks); the 4-block version is
    retained for comparability with v1.

10. MULTIPLICITY IS CORRECTED.  The four per-model Spearman correlations and
    the sixty-six McNemar tests now carry Holm-adjusted p-values.

11. UNCERTAINTY IS QUANTIFIED.  A date-block bootstrap (resampling whole
    trading dates, so cross-sectional dependence between tickers is
    preserved) gives CIs for every AUC, for the band means, for the
    LSTM-minus-logistic margins and for the RQ2 rank correlation.

12. THE TESTS PROMISED IN CHAPTER 3 ARE ALL RUN.  v1 never reported the
    Mann-Whitney U band comparison and ran McNemar once, for one pair, in one
    sector.

Note 1: predictions.parquet needs pyarrow or fastparquet.  If neither is
installed the pipeline falls back to a compressed CSV automatically.
=============================================================================
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# =========================================================================== #
# CONFIG                                                                      #
# =========================================================================== #
PRICE_CSV = "merged_data.csv"
SECTOR_MAP = "ticker_sectors.xlsx"
OUT_DIR = "results"

COL_DATE, COL_TICKER, COL_SECTOR = "date", "ticker", "Sector"
COL_OPEN, COL_HIGH, COL_LOW = "Open", "High", "Low"
COL_CLOSE, COL_VOLUME = "Close", "Volume"

# Columns that must never reach a model.  `sentiment` belongs to a separate
# project in the supervisory team and is out of scope for this study; the
# assertion in build_features() enforces that.
FORBIDDEN_FEATURES = ("sentiment",)

RANDOM_STATE = 42
N_SPLITS = 5                    # expanding-window walk-forward folds
ALPHA = 0.05
METRIC = "auc"                  # primary metric

# --- Sectoral Complexity Index -------------------------------------------- #
# +1 if a HIGHER value means MORE complex (harder to forecast).  Hurst is -1:
# stronger persistence aids forecasting and therefore lowers complexity
# (Eom et al., 2008).  HURST_SENSITIVITY re-runs the index with +1.
SCI_DIRECTION = {"volatility": +1, "sampen": +1, "hurst": -1,
                 "amihud": +1, "vol_var": +1}
SCI_ZSCORE_DDOF = 0             # population SD across the eleven sectors
SAMPEN_M, SAMPEN_R = 2, 0.2
HURST_MIN_N = 8                 # smallest R/S block
HURST_MIN_N_ROBUST = 16         # sensitivity check (R/S is biased at small n)
N_BANDS = 3
MIN_HISTORY_SCI = 250

# --- Features -------------------------------------------------------------- #
LAGS = [1, 2, 3, 5, 10]         # r_{t-1} ... r_{t-10}
INCLUDE_LAG0 = False            # r_t itself.  Main spec = False to match the
                                # documented feature set; the robustness block
                                # re-runs with True.
MA_WINDOWS = [5, 10, 20]
VOL_WINDOW = 10
RSI_WINDOW = 14
MOM_WINDOW = 10

# --- LSTM ------------------------------------------------------------------ #
SEQ_LEN = 10
LSTM_HIDDEN = 32
LSTM_EPOCHS = 10
LSTM_BATCH = 512
LSTM_LR = 1e-3
LSTM_VAL_FRAC = 0.10            # tail of the training partition, for epoch choice
N_SEEDS_LSTM = 5

# --- Inference ------------------------------------------------------------- #
BOOTSTRAP_B = 2000               # date-block bootstrap replicates
BOOTSTRAP_SEED = 7
REUSE_PREDICTIONS = True        # load results/predictions.parquet if present
                                # instead of refitting every model
RUN_ROBUSTNESS = True
QUICK_MODE = False              # True -> 3 sectors, 1 seed, 50 bootstraps

# Nemenyi q_alpha, alpha=0.05, Demsar (2006) Table 5(a), indexed by #models
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

NUMBERS: dict[str, str] = {}    # macro name -> formatted value, for numbers.tex


def _save_frame(frame: pd.DataFrame, stem: str) -> str:
    """Write a frame, preferring parquet and falling back to gzipped CSV."""
    try:
        frame.to_parquet(f"{stem}.parquet", index=False)
        return f"{stem}.parquet"
    except Exception:
        frame.to_csv(f"{stem}.csv.gz", index=False, compression="gzip")
        return f"{stem}.csv.gz"


def _load_frame(stem: str):
    """Read back whichever format _save_frame produced, or None."""
    for ext, reader in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        path = f"{stem}{ext}"
        if os.path.exists(path):
            try:
                frame = reader(path)
                frame[COL_DATE] = pd.to_datetime(frame[COL_DATE])
                return frame, path
            except Exception:
                return None, None
    return None, None


def note(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def macro(name: str, value, fmt: str = "{:.4f}") -> None:
    """Record a value for emission as a LaTeX \\renewcommand."""
    NUMBERS[name] = value if isinstance(value, str) else fmt.format(value)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)   # cuDNN RNN has no det. kernel
        torch.backends.cudnn.benchmark = False


# =========================================================================== #
# DATA                                                                        #
# =========================================================================== #
def load_data(price_csv: str = PRICE_CSV, sector_map: str = SECTOR_MAP) -> pd.DataFrame:
    df = pd.read_csv(price_csv, parse_dates=[COL_DATE])
    n_raw = len(df)

    smap = (pd.read_excel(sector_map) if str(sector_map).lower().endswith((".xlsx", ".xls"))
            else pd.read_csv(sector_map))
    smap = smap.iloc[:, :2]
    smap.columns = [COL_TICKER, COL_SECTOR]

    df = df.merge(smap, on=COL_TICKER, how="inner")
    keep = [COL_DATE, COL_TICKER, COL_SECTOR, COL_OPEN, COL_HIGH,
            COL_LOW, COL_CLOSE, COL_VOLUME]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=[COL_CLOSE])
    df = df[(df[COL_CLOSE] > 0) & (df[COL_VOLUME] > 0)]     # kills 1/0 in Amihud
    df = df.drop_duplicates([COL_TICKER, COL_DATE])
    df = df.sort_values([COL_TICKER, COL_DATE]).reset_index(drop=True)

    lengths = df.groupby(COL_TICKER).size()
    note(f"    rows {n_raw} raw -> {len(df)} clean | "
         f"{df[COL_TICKER].nunique()} tickers | {df[COL_SECTOR].nunique()} sectors")
    macro("NRowsRaw", f"{n_raw:,}")
    macro("NRowsClean", f"{len(df):,}")
    macro("NTickers", str(df[COL_TICKER].nunique()))
    macro("NSectors", str(df[COL_SECTOR].nunique()))
    macro("DateStart", df[COL_DATE].min().strftime("%-d %B %Y")
          if os.name != "nt" else df[COL_DATE].min().strftime("%d %B %Y"))
    macro("DateEnd", df[COL_DATE].max().strftime("%-d %B %Y")
          if os.name != "nt" else df[COL_DATE].max().strftime("%d %B %Y"))
    macro("MinTickerDays", str(int(lengths.min())))
    macro("MaxTickerDays", str(int(lengths.max())))
    macro("NTickersSCI", str(int((lengths >= MIN_HISTORY_SCI).sum())))
    macro("NTickersDropped", str(int((lengths < MIN_HISTORY_SCI).sum())))
    return df


# =========================================================================== #
# SECTORAL COMPLEXITY INDEX                                                   #
# =========================================================================== #
def log_returns(close) -> np.ndarray:
    r = np.diff(np.log(np.asarray(close, dtype=float)))
    return r[np.isfinite(r)]


def annualised_vol(r: np.ndarray, periods: int = 252) -> float:
    return float(np.std(r, ddof=1) * np.sqrt(periods)) if r.size >= 2 else np.nan


def sample_entropy(x, m: int = SAMPEN_M, r_frac: float = SAMPEN_R) -> float:
    """Richman and Moorman (2000) sample entropy, -ln(A/B)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    if N < m + 2:
        return np.nan
    r = r_frac * np.std(x, ddof=0)
    if r <= 0:
        return np.nan

    def _count(mm: int) -> int:
        M = N - m                      # same template count for m and m+1
        templates = np.lib.stride_tricks.sliding_window_view(x, mm)[:M]
        total = 0
        for i in range(M - 1):
            d = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            total += int(np.count_nonzero(d <= r))
        return total

    B, A = _count(m), _count(m + 1)
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan


def hurst_rs(x, min_n: int = HURST_MIN_N) -> float:
    """Hurst exponent by rescaled-range analysis on non-overlapping blocks.

    NOTE: classical R/S is upward biased at small block sizes, which is why the
    sector spread is narrow.  hurst_min_n is a config knob and the robustness
    block re-estimates the index with min_n = HURST_MIN_N_ROBUST.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    if N < 4 * min_n:
        return np.nan
    ns = np.unique(np.floor(np.logspace(np.log10(min_n),
                                        np.log10(N // 2), 20)).astype(int))
    ns = ns[ns >= min_n]
    rs, valid = [], []
    for n in ns:
        n = int(n)
        blocks = N // n
        if blocks < 1:
            continue
        vals = []
        for b in range(blocks):
            seg = x[b * n:(b + 1) * n]
            y = np.cumsum(seg - seg.mean())
            R = y.max() - y.min()
            S = seg.std(ddof=1)
            if S > 0 and R > 0:
                vals.append(R / S)
        if vals:
            rs.append(float(np.mean(vals)))
            valid.append(n)
    if len(valid) < 4:
        return np.nan
    return float(np.polyfit(np.log(valid), np.log(rs), 1)[0])


def amihud_illiquidity(r: np.ndarray, close, volume, scale: float = 1e6) -> float:
    """Amihud (2002): mean |return| per unit of dollar volume, times 1e6."""
    close = np.asarray(close, dtype=float)[1:]
    volume = np.asarray(volume, dtype=float)[1:]
    if close.size != r.size:                       # guard against ragged input
        k = min(close.size, r.size)
        close, volume, r = close[:k], volume[:k], r[:k]
    dollar = close * volume
    mask = (dollar > 0) & np.isfinite(r) & np.isfinite(dollar)
    if mask.sum() < 2:
        return np.nan
    return float(np.mean(np.abs(r[mask]) / dollar[mask]) * scale)


def volume_variability(volume) -> float:
    v = np.asarray(volume, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1) / v.mean()) if v.size >= 2 and v.mean() > 0 else np.nan


def _ticker_metrics(g: pd.DataFrame, hurst_min_n: int = HURST_MIN_N) -> pd.Series:
    close = g[COL_CLOSE].to_numpy()
    vol = g[COL_VOLUME].to_numpy()
    r = log_returns(close)
    return pd.Series({
        "volatility": annualised_vol(r),
        "sampen": sample_entropy(r),
        "hurst": hurst_rs(r, min_n=hurst_min_n),
        "amihud": amihud_illiquidity(r, close, vol),
        "vol_var": volume_variability(vol),
    })


def _index_from_raw(per_sector: pd.DataFrame, direction: dict) -> pd.Series:
    metrics = list(direction)
    z = ((per_sector[metrics] - per_sector[metrics].mean())
         / per_sector[metrics].std(ddof=SCI_ZSCORE_DDOF))
    return z.mul(pd.Series(direction)).mean(axis=1)


def compute_sci(df: pd.DataFrame, hurst_min_n: int = HURST_MIN_N,
                record: bool = True) -> pd.DataFrame:
    """Per-sector table: five raw metrics, n_tickers, SCI and complexity band."""
    counts = df.groupby(COL_TICKER)[COL_DATE].transform("size")
    long = df[counts >= MIN_HISTORY_SCI]

    per_ticker = (long.groupby([COL_SECTOR, COL_TICKER], sort=False)
                      .apply(lambda g: _ticker_metrics(g, hurst_min_n),
                             include_groups=False)
                      .reset_index())

    metrics = list(SCI_DIRECTION)
    per_sector = per_ticker.groupby(COL_SECTOR)[metrics].mean(numeric_only=True)
    per_sector["n_tickers"] = per_ticker.groupby(COL_SECTOR).size()
    per_sector["SCI"] = _index_from_raw(per_sector, SCI_DIRECTION)
    per_sector["band"] = pd.qcut(per_sector["SCI"], N_BANDS,
                                 labels=["low", "medium", "high"])
    per_sector = per_sector.sort_values("SCI")

    # sensitivity: flip the Hurst orientation
    flipped = dict(SCI_DIRECTION, hurst=+1)
    per_sector["SCI_hurst_plus"] = _index_from_raw(per_sector, flipped)
    tau, tau_p = stats.kendalltau(per_sector["SCI"], per_sector["SCI_hurst_plus"])
    if record: macro("HurstFlipTau", tau)
    if record: macro("HurstFlipTauP", tau_p, "{:.3f}")

    corr = per_sector[metrics].corr()
    if record: macro("CorrVolHurst", corr.loc["volatility", "hurst"], "{:.2f}")
    if record: macro("CorrHurstVolVar", corr.loc["hurst", "vol_var"], "{:.2f}")
    if record: macro("CorrAmihudMin", corr.loc["amihud", [c for c in metrics if c != "amihud"]].min(), "{:.2f}")
    if record: macro("CorrAmihudMax", corr.loc["amihud", [c for c in metrics if c != "amihud"]].max(), "{:.2f}")
    if record: macro("HurstMin", per_sector["hurst"].min(), "{:.3f}")
    if record: macro("HurstMax", per_sector["hurst"].max(), "{:.3f}")
    if record: macro("SCIMin", per_sector["SCI"].min(), "{:+.3f}")
    if record: macro("SCIMax", per_sector["SCI"].max(), "{:+.3f}")
    if record: macro("NLowBand", str(int((per_sector["band"] == "low").sum())))
    if record: macro("NMediumBand", str(int((per_sector["band"] == "medium").sum())))
    if record: macro("NHighBand", str(int((per_sector["band"] == "high").sum())))
    return per_sector


# =========================================================================== #
# FEATURES                                                                    #
# =========================================================================== #
def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _features_one_ticker(g: pd.DataFrame, include_lag0: bool) -> pd.DataFrame:
    g = g.sort_values(COL_DATE)
    close, vol = g[COL_CLOSE], g[COL_VOLUME]
    ret = np.log(close).diff()

    feat = pd.DataFrame(index=g.index)
    if include_lag0:
        feat["ret_lag0"] = ret                       # r_t, known at close of t
    for k in LAGS:
        feat[f"ret_lag{k}"] = ret.shift(k)
    for w in MA_WINDOWS:
        # methodology eq. (3): P_t / MA_w(P) - 1
        feat[f"close_ma{w}"] = close / close.rolling(w).mean() - 1
    feat[f"vol_{VOL_WINDOW}"] = ret.rolling(VOL_WINDOW).std()
    feat["rsi"] = _rsi(close, RSI_WINDOW) / 100
    feat[f"mom{MOM_WINDOW}"] = close / close.shift(MOM_WINDOW) - 1
    feat["vol_chg"] = vol.pct_change()
    feat["vol_ma_ratio"] = vol / vol.rolling(MA_WINDOWS[-1]).mean()

    feat[COL_DATE] = g[COL_DATE].to_numpy()
    feat[COL_TICKER] = g[COL_TICKER].to_numpy()
    feat["target"] = (ret.shift(-1) > 0).astype(int).to_numpy()   # sign of r_{t+1}
    return feat


def build_features(df: pd.DataFrame, include_lag0: bool = INCLUDE_LAG0,
                   record: bool = True) -> pd.DataFrame:
    parts = [_features_one_ticker(g, include_lag0)
             for _, g in df.groupby(COL_TICKER, sort=False)]
    out = pd.concat(parts).replace([np.inf, -np.inf], np.nan).dropna()
    tmap = df[[COL_TICKER, COL_SECTOR]].drop_duplicates()
    out = out.merge(tmap, on=COL_TICKER, how="left")
    out = out.sort_values([COL_TICKER, COL_DATE]).reset_index(drop=True)

    cols = feature_columns(out)
    bad = [c for c in cols if any(f in c.lower() for f in FORBIDDEN_FEATURES)]
    assert not bad, f"out-of-scope column reached the feature matrix: {bad}"
    assert "target" not in cols and COL_DATE not in cols
    if record:
        macro("NFeatures", str(len(cols)))
        macro("NModelRows", f"{len(out):,}")
    note(f"    feature panel {len(out):,} rows x {len(cols)} features")
    return out


def feature_columns(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns
            if c not in (COL_DATE, COL_TICKER, COL_SECTOR, "target")]


# =========================================================================== #
# MODELS                                                                      #
# =========================================================================== #
def make_sklearn(name: str, seed: int = RANDOM_STATE):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=seed,
                             n_jobs=-1, tree_method="hist")
    raise ValueError(name)


def available_models() -> list[str]:
    models = ["LogisticRegression", "RandomForest"]
    if _HAS_XGB:
        models.append("XGBoost")
    else:
        note("    [warn] xgboost not installed -> XGBoost skipped")
    if _HAS_TORCH:
        models.append("LSTM")
    else:
        note("    [warn] torch not installed -> LSTM skipped")
    return models


def _windows_for_ticker(X: np.ndarray, y: np.ndarray, keep: np.ndarray,
                        seq_len: int):
    """Windows ENDING AT t inclusive: rows t-seq_len+1 .. t predict y_t.

    v1 used rows t-seq_len .. t-1, so the LSTM never saw day t while the
    tabular models did.

    `keep` marks which end-rows to emit.  Because the whole ticker history is
    passed in, a test row near a fold boundary draws its window from
    training-period rows, so no test observation is dropped and the LSTM is
    scored on the same sample as the tabular models.
    """
    n = X.shape[0]
    if n < seq_len:
        return None, None, None
    wins = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=seq_len, axis=0)                 # (m, f, seq_len)
    wins = np.ascontiguousarray(wins.transpose(0, 2, 1))  # (m, seq_len, f)
    ends = np.arange(seq_len - 1, n)
    sel = keep[ends]
    if not sel.any():
        return None, None, None
    return wins[sel], y[ends][sel], ends[sel]


def build_sequences(frame: pd.DataFrame, cols: list[str], keep_mask: np.ndarray,
                    seq_len: int = SEQ_LEN):
    """frame must be sorted by (ticker, date) with a positional RangeIndex."""
    Xs, ys, pos = [], [], []
    arr_all = frame[cols].to_numpy(np.float32)
    y_all = frame["target"].to_numpy(np.float32)
    for _, idx in frame.groupby(COL_TICKER, sort=False).indices.items():
        idx = np.sort(idx)
        Xw, yw, ends = _windows_for_ticker(arr_all[idx], y_all[idx],
                                          keep_mask[idx], seq_len)
        if Xw is None:
            continue
        Xs.append(Xw)
        ys.append(yw)
        pos.append(idx[ends])
    if not Xs:
        return None, None, None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(pos)


def make_fold_sequences(frame: pd.DataFrame, cols: list[str],
                        train_mask: np.ndarray, test_mask: np.ndarray,
                        scaler: StandardScaler):
    """Build the train/test sequence tensors ONCE per fold.

    v1 rebuilt them inside every LSTM call; with N_SEEDS_LSTM seeds that is
    five times the work for identical arrays.
    """
    scaled = frame.copy()
    scaled[cols] = scaler.transform(frame[cols])
    Xtr, ytr, pos_tr = build_sequences(scaled, cols, train_mask)
    Xte, yte, pos_te = build_sequences(scaled, cols, test_mask)
    if Xtr is None or Xte is None or len(np.unique(ytr)) < 2:
        return None
    # inner validation split = latest LSTM_VAL_FRAC of training windows in time
    order = np.argsort(frame[COL_DATE].to_numpy()[pos_tr], kind="stable")
    n_val = max(1, int(len(order) * LSTM_VAL_FRAC))
    val_idx, tr_idx = order[-n_val:], order[:-n_val]
    if len(tr_idx) < LSTM_BATCH or len(np.unique(ytr[tr_idx])) < 2:
        tr_idx, val_idx = np.arange(len(ytr)), None
    return {"Xtr": Xtr, "ytr": ytr, "tr_idx": tr_idx, "val_idx": val_idx,
            "Xte": Xte, "yte": yte, "pos_te": pos_te}


if _HAS_TORCH:

    class SeqLSTM(nn.Module):
        def __init__(self, n_feat: int, hidden: int = LSTM_HIDDEN):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    def train_lstm_once(seq: dict, n_feat: int, seed: int) -> np.ndarray:
        """One seeded training run; returns test-set probabilities.

        v1 ran LSTM_EPOCHS *full-batch* gradient steps (eight Adam updates from
        random initialisation), so it measured a near-random projection of the
        features.  This is minibatch training with the epoch chosen on a
        held-out tail of the training partition.
        """
        set_all_seeds(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = SeqLSTM(n_feat).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=LSTM_LR)
        loss_fn = nn.BCEWithLogitsLoss()

        Xtr, ytr = seq["Xtr"], seq["ytr"]
        tr_idx, val_idx = seq["tr_idx"], seq["val_idx"]
        ds = TensorDataset(torch.from_numpy(Xtr[tr_idx]), torch.from_numpy(ytr[tr_idx]))
        gen = torch.Generator().manual_seed(seed)
        dl = DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True, generator=gen)

        def _probs(X: np.ndarray) -> np.ndarray:
            net.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 4096):
                    xb = torch.from_numpy(X[i:i + 4096]).to(dev)
                    out.append(torch.sigmoid(net(xb)).cpu().numpy())
            return np.concatenate(out)

        best_state, best_score = None, -np.inf
        for _ in range(LSTM_EPOCHS):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()
            if val_idx is None:
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                continue
            pv = _probs(Xtr[val_idx])
            score = (roc_auc_score(ytr[val_idx], pv)
                     if len(np.unique(ytr[val_idx])) > 1 else 0.5)
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
        return _probs(seq["Xte"])


# =========================================================================== #
# WALK-FORWARD EVALUATION -> long prediction table                            #
# =========================================================================== #
def run_sector(fs: pd.DataFrame, models: list[str],
               n_seeds_lstm: int = N_SEEDS_LSTM) -> pd.DataFrame:
    """Every out-of-sample prediction for one sector, as a long table."""
    fs = fs.sort_values([COL_TICKER, COL_DATE]).reset_index(drop=True)
    cols = feature_columns(fs)
    dates = np.sort(fs[COL_DATE].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(dates), start=1):
        train_mask = fs[COL_DATE].isin(dates[tr_idx]).to_numpy()
        test_mask = fs[COL_DATE].isin(dates[te_idx]).to_numpy()
        if not train_mask.any() or not test_mask.any():
            continue
        train, test = fs[train_mask], fs[test_mask]
        if train["target"].nunique() < 2:
            continue

        scaler = StandardScaler().fit(train[cols])
        train_major = float(max(train["target"].mean(), 1 - train["target"].mean()))
        # the constant predictor a decision maker could actually deploy
        train_const = int(train["target"].mean() >= 0.5)

        for name in models:
            if name == "LSTM":
                if not _HAS_TORCH or n_seeds_lstm < 1:
                    continue
                seq = make_fold_sequences(fs, cols, train_mask, test_mask, scaler)
                if seq is None:
                    continue
                pos, y_true = seq["pos_te"], seq["yte"].astype(int)
                for sd in range(n_seeds_lstm):
                    prob = train_lstm_once(seq, len(cols), seed=RANDOM_STATE + sd)
                    rows.append(pd.DataFrame({
                        COL_DATE: fs[COL_DATE].to_numpy()[pos],
                        COL_TICKER: fs[COL_TICKER].to_numpy()[pos],
                        "model": "LSTM", "seed": sd, "fold": fold,
                        "y_true": y_true, "y_prob": prob,
                        "train_major": train_major, "train_const": train_const}))
            else:
                clf = make_sklearn(name, seed=RANDOM_STATE)
                clf.fit(scaler.transform(train[cols]), train["target"])
                prob = clf.predict_proba(scaler.transform(test[cols]))[:, 1]
                rows.append(pd.DataFrame({
                    COL_DATE: test[COL_DATE].to_numpy(),
                    COL_TICKER: test[COL_TICKER].to_numpy(),
                    "model": name, "seed": 0, "fold": fold,
                    "y_true": test["target"].to_numpy(),
                    "y_prob": prob,
                    "train_major": train_major, "train_const": train_const}))

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out[COL_SECTOR] = fs[COL_SECTOR].iloc[0]
    return out


def restrict_to_common_support(preds: pd.DataFrame) -> pd.DataFrame:
    """Keep only (sector, date, ticker) keys scored by EVERY model.

    Without this, McNemar and the paired band comparisons are computed on
    slightly different samples for the LSTM.  The dropped share is reported.
    """
    n_models = preds["model"].nunique()
    key = [COL_SECTOR, COL_DATE, COL_TICKER]
    cover = preds.groupby(key, sort=False, observed=True)["model"].transform("nunique")
    before = len(preds)
    out = preds[cover == n_models].reset_index(drop=True)
    macro("CommonSupportPct", 100 * len(out) / before, "{:.2f}")
    note(f"    common support keeps {len(out):,}/{before:,} prediction rows "
         f"({100 * len(out) / before:.2f}%)")
    return out


# =========================================================================== #
# METRICS                                                                     #
# =========================================================================== #
def _metric_block(y: np.ndarray, p: np.ndarray, train_major: float,
                  train_const: np.ndarray) -> dict:
    yhat = (p >= 0.5).astype(int)
    two = len(np.unique(y)) > 1
    return {
        "accuracy": accuracy_score(y, yhat),
        "auc": roc_auc_score(y, p) if two else np.nan,
        "f1": f1_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat) if two else np.nan,
        "baseline_test": float(max(y.mean(), 1 - y.mean())),
        "baseline_train": float(np.mean(y == np.asarray(train_const))),
        "train_major": train_major,
        "n": int(len(y)),
    }


def metrics_by_foldblocked(preds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-fold metrics averaged over folds -- the primary specification.

    AUC is a ranking statistic, so computing it over the pooled predictions of
    several separately fitted folds estimates a quantity belonging to no fitted
    model, and is biased downwards when the target base rate drifts between
    folds (Section 3.7.3 of the report; see also results/foldblock_comparison).
    Folds are weighted equally because the test windows are near-identical in
    length.
    """
    per_fold = metrics_by(preds, list(keys) + ["fold"])
    metric_cols = [c for c in per_fold.columns
                   if c not in set(keys) | {"fold", "n"}]
    agg = {c: (c, "mean") for c in metric_cols}
    agg["auc_fold_sd"] = ("auc", "std")
    agg["n_folds"] = ("fold", "nunique")
    agg["n"] = ("n", "sum")
    return per_fold.groupby(list(keys), as_index=False).agg(**agg)


def metrics_by(preds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Metrics over whatever rows are passed. Use metrics_by_foldblocked for
    cell-level reporting; this pools across any folds present in `preds`."""
    out = []
    for k, g in preds.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        m = _metric_block(g["y_true"].to_numpy(), g["y_prob"].to_numpy(),
                          float(g["train_major"].mean()),
                          g["train_const"].to_numpy())
        out.append(dict(zip(keys, k), **m))
    return pd.DataFrame(out)


def collapse_lstm_seeds(per_seed: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and SD over LSTM seeds; other models pass through untouched."""
    num = ["accuracy", "auc", "f1", "precision", "recall", "mcc",
           "baseline_test", "baseline_train", "n"]
    grp = per_seed.groupby(keys, sort=False, observed=True)
    mean = grp[num].mean()
    sd = grp[num].std(ddof=1).add_suffix("_sd")
    n_seeds = grp.size().rename("n_seeds")
    return pd.concat([mean, sd, n_seeds], axis=1).reset_index()


# =========================================================================== #
# INFERENCE                                                                   #
# =========================================================================== #
def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def nemenyi_cd(k: int, n: int) -> float:
    q = _Q05.get(k)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n))) if q else np.nan


def rq1_sector_heterogeneity(fold_perf: pd.DataFrame, pooled: pd.DataFrame,
                             metric: str = METRIC) -> dict:
    """Primary test uses per-fold cells so the block count is adequate."""
    wide = (fold_perf.pivot_table(index=["model", "fold"], columns=COL_SECTOR,
                                  values=metric, observed=True)
                     .dropna(axis=0, how="any").dropna(axis=1, how="any"))
    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])

    # v1's specification, kept for comparability
    wide_v1 = pooled.pivot(index="model", columns=COL_SECTOR,
                           values=metric).dropna(axis=1)
    stat_v1, p_v1 = stats.friedmanchisquare(*[wide_v1[c].to_numpy()
                                              for c in wide_v1.columns])
    means = pooled.groupby(COL_SECTOR, observed=True)[metric].mean().sort_values()
    return {"test": "Friedman, sectors as treatments, model x fold as blocks",
            "chi2": float(stat), "p": float(p),
            "k_treatments": int(wide.shape[1]), "n_blocks": int(wide.shape[0]),
            "chi2_pooled_blocks": float(stat_v1), "p_pooled_blocks": float(p_v1),
            "n_blocks_pooled": int(wide_v1.shape[0]),
            "significant": bool(p < ALPHA),
            "sector_mean_spread": float(means.max() - means.min()),
            "sector_means": means.to_dict()}


def rq2_complexity(pooled: pd.DataFrame, sci: pd.DataFrame,
                   metric: str = METRIC) -> dict:
    perf = pooled.groupby(COL_SECTOR, observed=True)[metric].mean()
    d = sci[["SCI", "band"]].join(perf.rename("perf")).dropna(subset=["perf"])
    rho, p = stats.spearmanr(d["SCI"], d["perf"])

    lo = d.loc[d["band"] == "low", "perf"]
    hi = d.loc[d["band"] == "high", "perf"]
    u_h2 = stats.mannwhitneyu(lo, hi, alternative="greater")   # H2's direction
    u_two = stats.mannwhitneyu(lo, hi, alternative="two-sided")

    per_model, names = [], []
    for m, g in pooled.groupby("model", observed=True):
        dd = sci[["SCI"]].join(g.set_index(COL_SECTOR)[metric]).dropna()
        r, pp = stats.spearmanr(dd["SCI"], dd[metric])
        per_model.append((float(r), float(pp)))
        names.append(m)
    praw = np.array([pp for _, pp in per_model])
    padj = holm(praw)
    table = pd.DataFrame({"model": names,
                          "rho": [r for r, _ in per_model],
                          "p": praw, "p_holm": padj}).sort_values("rho", ascending=False)

    return {"test": "Spearman(SCI, mean AUC) across sectors",
            "rho": float(rho), "p": float(p),
            "negative_as_H2_predicts": bool(rho < 0),
            "significant": bool(p < ALPHA),
            "band_means": d.groupby("band", observed=True)["perf"].mean().to_dict(),
            "mannwhitney_U_H2": float(u_h2.statistic), "mannwhitney_p_H2": float(u_h2.pvalue),
            "mannwhitney_p_two_sided": float(u_two.pvalue),
            "per_model": table.to_dict("records"),
            "table": d.sort_values("SCI")}


def rq3_models(pooled: pd.DataFrame, metric: str = METRIC) -> dict:
    wide = pooled.pivot(index=COL_SECTOR, columns="model", values=metric).dropna()
    models = list(wide.columns)
    k, n = len(models), len(wide)
    if k < 3:
        stat, p = stats.wilcoxon(wide[models[0]], wide[models[1]])
        test = "Wilcoxon signed-rank"
        cd = np.nan
    else:
        stat, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in models])
        test = "Friedman + Nemenyi"
        cd = nemenyi_cd(k, n)
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    return {"test": test, "chi2": float(stat), "p": float(p),
            "significant": bool(p < ALPHA),
            "avg_ranks": ranks.to_dict(),
            "max_rank_gap": float(ranks.max() - ranks.min()),
            "critical_difference": float(cd)}


def mcnemar_all_pairs(preds: pd.DataFrame) -> pd.DataFrame:
    """Every model pair in every sector, on identical held-out observations."""
    p = preds.copy()
    p["correct"] = ((p["y_prob"] >= 0.5).astype(int) == p["y_true"]).astype(int)
    # collapse LSTM seeds to the seed-averaged probability for a single decision
    key = [COL_SECTOR, COL_DATE, COL_TICKER, "model"]
    p = (p.groupby(key, sort=False)
          .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
          .reset_index())
    p["correct"] = ((p["y_prob"] >= 0.5).astype(int) == p["y_true"]).astype(int)

    rows = []
    for sector, g in p.groupby(COL_SECTOR, sort=False):
        wide = g.pivot_table(index=[COL_DATE, COL_TICKER], columns="model",
                             values="correct")
        wide = wide.dropna()
        models = list(wide.columns)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b_ = wide[models[i]].to_numpy(), wide[models[j]].to_numpy()
                b = int(np.sum((a == 1) & (b_ == 0)))
                c = int(np.sum((a == 0) & (b_ == 1)))
                if b + c == 0:
                    pv = 1.0
                elif b + c < 25:
                    pv = float(stats.binomtest(min(b, c), b + c, 0.5,
                                               alternative="two-sided").pvalue)
                else:
                    chi = (abs(b - c) - 1) ** 2 / (b + c)
                    pv = float(stats.chi2.sf(chi, 1))
                rows.append({"sector": sector, "model_a": models[i],
                             "model_b": models[j], "n_pairs": int(len(wide)),
                             "b": b, "c": c, "p": pv})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
        out["significant_holm"] = out["p_holm"] < ALPHA
    return out.sort_values("p")


def date_block_bootstrap(preds: pd.DataFrame, sci: pd.DataFrame,
                         B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict:
    """Resample whole trading DATES within fold, not rows.

    Two-level design: dates are the resampling unit because several hundred
    tickers share each date and their errors are driven by a common market
    factor; folds are respected because each fold is a distinct fitted model and
    a pooled ranking across folds is biased (Section 3.7.3).

    Roughly five hundred tickers share each date, so their prediction errors
    are driven by a common market factor.  Resampling rows would treat them as
    independent and understate every standard error by an order of magnitude.
    Resampling dates preserves the cross-sectional dependence.
    """
    rng = np.random.default_rng(seed)
    p = (preds.groupby([COL_SECTOR, COL_DATE, COL_TICKER, "model", "fold"],
                       sort=False)
              .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
              .reset_index())

    # Dates are factorised WITHIN FOLD, because resampling is within fold: a
    # replicate must never rank one fold's predictions against another's.
    # See metrics_by_foldblocked for why pooling across folds is not used.
    blocks: dict[tuple, list] = {}
    fold_ndates: dict[int, int] = {}
    for f in np.sort(p["fold"].unique()):
        pf = p[p["fold"] == f]
        codes_all, dates_all = pd.factorize(pf[COL_DATE], sort=True)
        fold_ndates[f] = len(dates_all)
        pf = pf.assign(_dcode=codes_all)
        for (sector, model), g in pf.groupby([COL_SECTOR, "model"], sort=False):
            g = g.sort_values("_dcode", kind="stable")
            codes = g["_dcode"].to_numpy()
            uniq = np.unique(codes)
            blocks.setdefault((sector, model), []).append({
                "fold": f,
                "y": g["y_true"].to_numpy(),
                "p": g["y_prob"].to_numpy(),
                "starts": np.searchsorted(codes, uniq, side="left"),
                "ends": np.searchsorted(codes, uniq, side="right"),
                "local": uniq})

    draws: dict[tuple, list] = {k: [] for k in blocks}
    band = sci["band"].astype(str).to_dict()
    rho_draws, band_draws, margin_draws = [], [], []

    for _ in range(B):
        # One date resample per fold, shared by every cell in the replicate, so
        # the cross-sectional dependence among stocks sharing a date survives.
        picks = {f: np.bincount(rng.integers(0, n, n), minlength=n)
                 for f, n in fold_ndates.items()}
        cell_auc = {}
        for k, fold_list in blocks.items():
            per = []
            for c in fold_list:
                reps = picks[c["fold"]][c["local"]]
                take = np.nonzero(reps)[0]
                if take.size == 0:
                    continue
                idx = np.concatenate([
                    np.repeat(np.arange(c["starts"][i], c["ends"][i]), reps[i])
                    for i in take])
                y, pr = c["y"][idx], c["p"][idx]
                if len(np.unique(y)) > 1:
                    per.append(roc_auc_score(y, pr))
            a = float(np.mean(per)) if per else np.nan
            cell_auc[k] = a
            draws[k].append(a)

        df = (pd.Series(cell_auc).rename_axis([COL_SECTOR, "model"])
                .reset_index(name="auc"))
        sec_mean = df.groupby(COL_SECTOR)["auc"].mean()
        common = [s for s in sec_mean.index if s in sci.index]
        if len(common) > 2:
            r, _ = stats.spearmanr(sci.loc[common, "SCI"], sec_mean.loc[common])
            rho_draws.append(r)
        df["band"] = df[COL_SECTOR].map(band)
        bm = df.groupby(["band", "model"])["auc"].mean()
        band_draws.append(bm)
        if {"LSTM", "LogisticRegression"} <= set(df["model"]):
            margin_draws.append(bm.xs("LSTM", level="model")
                                - bm.xs("LogisticRegression", level="model"))

    def ci(v):
        v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
        if v.size < 10:
            return (np.nan, np.nan)
        return tuple(np.percentile(v, [2.5, 97.5]))

    cell_rows = []
    for (sector, model), v in draws.items():
        lo, hi = ci(v)
        v = np.asarray([x for x in v if np.isfinite(x)])
        cell_rows.append({"sector": sector, "model": model,
                          "auc_boot_mean": float(np.mean(v)) if v.size else np.nan,
                          "ci_lo": lo, "ci_hi": hi,
                          "p_two_sided_vs_half": float(2 * min((v <= 0.5).mean(),
                                                               (v >= 0.5).mean()))
                          if v.size else np.nan,
                          "excludes_half": bool(np.isfinite(lo) and (lo > 0.5 or hi < 0.5))})
    cell_df = pd.DataFrame(cell_rows)
    if len(cell_df):
        cell_df["p_holm"] = holm(cell_df["p_two_sided_vs_half"].fillna(1).to_numpy())

    band_df = pd.DataFrame()
    if band_draws:
        stacked = pd.concat(band_draws, axis=1)
        band_df = pd.DataFrame({"mean": stacked.mean(axis=1),
                                "ci_lo": stacked.quantile(.025, axis=1),
                                "ci_hi": stacked.quantile(.975, axis=1)}).reset_index()
    margin_df = pd.DataFrame()
    if margin_draws:
        st = pd.concat(margin_draws, axis=1)
        margin_df = pd.DataFrame({"mean": st.mean(axis=1),
                                  "ci_lo": st.quantile(.025, axis=1),
                                  "ci_hi": st.quantile(.975, axis=1)}).reset_index()

    rho_ci = ci(rho_draws)
    return {"cells": cell_df, "bands": band_df, "margins": margin_df,
            "rho_ci": rho_ci, "B": B}


def adaptive_selection(fold_perf: pd.DataFrame, pooled: pd.DataFrame,
                       sci: pd.DataFrame, metric: str = METRIC) -> dict:
    """Band-to-model policy, fitted on early folds and scored on late folds."""
    fp = fold_perf.merge(sci["band"].rename_axis(COL_SECTOR).reset_index(),
                         on=COL_SECTOR)
    folds = sorted(fp["fold"].unique())
    cut = max(1, len(folds) - 2)
    fit_folds, score_folds = folds[:cut], folds[cut:]

    fit = fp[fp["fold"].isin(fit_folds)]
    choice = (fit.groupby(["band", "model"], observed=True)[metric].mean()
                 .reset_index().sort_values(metric, ascending=False)
                 .drop_duplicates("band").set_index("band"))

    score = fp[fp["fold"].isin(score_folds)]
    sel = []
    for sector, g in score.groupby(COL_SECTOR, observed=True):
        b = str(sci.loc[sector, "band"])
        if b not in choice.index:
            continue
        m = choice.loc[b, "model"]
        v = g.loc[g["model"] == m, metric]
        if len(v):
            sel.append(v.mean())
    oos_adaptive = float(np.nanmean(sel)) if sel else np.nan
    fixed = score.groupby("model", observed=True)[metric].mean()
    oos_best_fixed_model = fixed.idxmax()
    oos_best_fixed = float(fixed.max())

    # in-sample version (v1's number), reported explicitly as an upper bound
    wide = pooled.pivot(index=COL_SECTOR, columns="model", values=metric)
    ch_in = (pooled.merge(sci["band"].rename_axis(COL_SECTOR).reset_index(), on=COL_SECTOR)
                   .groupby(["band", "model"], observed=True)[metric].mean()
                   .reset_index().sort_values(metric, ascending=False)
                   .drop_duplicates("band").set_index("band"))
    in_sel = [wide.loc[s, ch_in.loc[str(sci.loc[s, "band"]), "model"]]
              for s in wide.index if str(sci.loc[s, "band"]) in ch_in.index]
    return {"fit_folds": fit_folds, "score_folds": score_folds,
            "band_to_model": choice[["model", metric]].to_dict("index"),
            "oos_adaptive_mean": oos_adaptive,
            "oos_best_fixed_model": str(oos_best_fixed_model),
            "oos_best_fixed_mean": oos_best_fixed,
            "oos_uplift": oos_adaptive - oos_best_fixed,
            "in_sample_adaptive_mean": float(np.nanmean(in_sel)),
            "in_sample_best_fixed_mean": float(wide.mean().max()),
            "in_sample_best_fixed_model": str(wide.mean().idxmax()),
            "per_sector_oracle_mean": float(wide.max(axis=1).mean())}


# =========================================================================== #
# ROBUSTNESS                                                                  #
# =========================================================================== #
def robustness_per_ticker(feat: pd.DataFrame, sci: pd.DataFrame, split: float = 0.7):
    """One model per ticker on a single 70/30 time split, aggregated to sector.

    NOTE the eligibility difference from the SCI: MIN_HISTORY_SCI is applied
    here to the POST-dropna feature frame, which is about twenty-five rows
    shorter per ticker, so a ticker with 250-275 raw days qualifies for the
    index but not for this check.  That is why n can be one lower than the SCI
    column.  Both counts are reported.
    """
    cols = feature_columns(feat)
    rows = []
    for (sec, tk), g in feat.groupby([COL_SECTOR, COL_TICKER], sort=False):
        g = g.sort_values(COL_DATE)
        if len(g) < MIN_HISTORY_SCI:
            continue
        cut = int(len(g) * split)
        tr, te = g.iloc[:cut], g.iloc[cut:]
        if tr["target"].nunique() < 2 or te["target"].nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[cols])
        Xtr, Xte = sc.transform(tr[cols]), sc.transform(te[cols])
        for name, clf in [("LR", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                          ("RF", RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                        random_state=RANDOM_STATE))]:
            clf.fit(Xtr, tr["target"])
            pr = clf.predict_proba(Xte)[:, 1]
            rows.append({"sector": sec, "ticker": tk, "model": name,
                         "acc": accuracy_score(te["target"], (pr >= 0.5).astype(int)),
                         "auc": roc_auc_score(te["target"], pr)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail
    summary = (detail.groupby("sector")
                     .agg(n_tickers=("ticker", "nunique"),
                          perticker_auc=("auc", "mean"),
                          perticker_acc=("acc", "mean"),
                          pct_auc_gt_50=("auc", lambda s: (s > 0.5).mean() * 100))
                     .join(sci["band"]).sort_values("perticker_auc"))
    return detail, summary


# =========================================================================== #
# LATEX EMISSION                                                              #
# =========================================================================== #
def _fmt(v, nd=4, signed=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return f"${s}$" if signed else s


def write_tables(out: str, sci: pd.DataFrame, pooled: pd.DataFrame,
                 boot: dict, mcn: pd.DataFrame, seeds: pd.DataFrame,
                 rob_summary: pd.DataFrame, sci_flip: pd.DataFrame) -> None:
    os.makedirs(out, exist_ok=True)

    # Table: SCI and components
    lines = [r"\begin{tabular}{lrrrrrrrl}", r"\toprule",
             r"Sector & Vol. & SampEn & Hurst & Amihud & Vol.\ var. & $n$ & SCI & Band \\",
             r"\midrule"]
    for s, r in sci.iterrows():
        lines.append(f"{s} & {r.volatility:.3f} & {r.sampen:.3f} & {r.hurst:.3f} & "
                     f"{r.amihud * 1e4:.3f} & {r.vol_var:.3f} & {int(r.n_tickers)} & "
                     f"${r.SCI:+.3f}$ & {r.band} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_sci.tex", "w").write("\n".join(lines) + "\n")

    # Table: mean performance by model
    agg = (pooled.groupby("model", observed=True)[["accuracy", "auc", "f1", "mcc"]]
                 .mean().sort_values("auc", ascending=False))
    ranks = (pooled.pivot(index=COL_SECTOR, columns="model", values="auc")
                   .rank(axis=1, ascending=False).mean())
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & Accuracy & AUC & F1 & MCC & Mean rank \\", r"\midrule"]
    for m, r in agg.iterrows():
        lines.append(f"{m} & {r.accuracy:.4f} & {r.auc:.4f} & {r.f1:.4f} & "
                     f"${r.mcc:+.4f}$ & {ranks[m]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_perf_model.tex", "w").write("\n".join(lines) + "\n")

    # Table: bootstrap CIs (longtable: owns its own caption and label)
    cells = boot["cells"].merge(pooled[[COL_SECTOR, "model", "auc"]]
                               .rename(columns={COL_SECTOR: "sector"}),
                               on=["sector", "model"], how="left")
    cells = cells.sort_values("p_two_sided_vs_half")
    head = (r"Sector & Model & AUC & 95\% CI & $p_{\text{Holm}}$ & Verdict \\")
    lines = [r"\begin{longtable}{llrrrl}",
             r"\caption{Date-block bootstrap 95\% confidence intervals for the "
             r"out-of-sample AUC of every model--sector combination, with "
             r"Holm-adjusted two-sided $p$-values against an AUC of 0.5. Whole "
             r"trading dates are resampled, so cross-sectional dependence between "
             r"tickers is preserved.}\label{tab:ci}\\",
             r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\toprule", head, r"\midrule", r"\endhead",
             r"\midrule \multicolumn{6}{r}{\emph{continued on next page}}\\",
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for _, r in cells.iterrows():
        verdict = ("above chance" if r.ci_lo > 0.5 else
                   "below chance" if r.ci_hi < 0.5 else "indistinguishable")
        lines.append(f"{r.sector} & {r.model} & {r.auc:.4f} & "
                     f"$[{r.ci_lo:.4f},\\,{r.ci_hi:.4f}]$ & {r.p_holm:.3f} & {verdict} \\\\")
    lines.append(r"\end{longtable}")
    open(f"{out}/tab_ci.tex", "w").write("\n".join(lines) + "\n")

    # Table: full per-model per-sector results (longtable)
    band_order = {"low": 0, "medium": 1, "high": 2}
    full = pooled.copy()
    full["band"] = full[COL_SECTOR].map(sci["band"].astype(str).to_dict())
    full["_b"] = full["band"].map(band_order)
    full = full.sort_values(["_b", COL_SECTOR, "model"])
    head = (r"Sector & Model & Acc. & AUC & F1 & MCC & Base (test) & Base (train) \\")
    lines = [r"\begin{longtable}{llrrrrrr}",
             r"\caption{Out-of-sample performance for all model--sector "
             r"combinations. Baseline (test) is the hindsight majority class; "
             r"baseline (train) applies the training-partition majority to the "
             r"test partition. LSTM rows are means across seeds.}\label{tab:full}\\",
             r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\toprule", head, r"\midrule", r"\endhead",
             r"\midrule \multicolumn{8}{r}{\emph{continued on next page}}\\",
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for _, r in full.iterrows():
        lines.append(f"{r[COL_SECTOR]} & {r['model']} & {r.accuracy:.4f} & {r.auc:.4f} & "
                     f"{r.f1:.4f} & ${r.mcc:+.4f}$ & {r.baseline_test:.4f} & "
                     f"{r.baseline_train:.4f} \\\\")
    lines.append(r"\end{longtable}")
    open(f"{out}/tab_perf_full.tex", "w").write("\n".join(lines) + "\n")

    # Table: mean AUC by model and complexity band, with bootstrap margin row
    pb = pooled.merge(sci["band"].rename_axis(COL_SECTOR).reset_index(), on=COL_SECTOR)
    bt = pb.pivot_table(index="model", columns="band", values="auc",
                        aggfunc="mean", observed=True)
    bands = [b for b in ["low", "medium", "high"] if b in bt.columns]
    bt = bt[bands]
    lines = [r"\begin{tabular}{l" + "r" * len(bands) + "}", r"\toprule",
             "Model & " + " & ".join(b.capitalize() for b in bands) + r" \\",
             r"\midrule"]
    for m in bt.index:
        lines.append(m + " & " + " & ".join(f"{bt.loc[m, b]:.4f}" for b in bands) + r" \\")
    if {"LSTM", "LogisticRegression"}.issubset(set(bt.index)):
        lines.append(r"\midrule")
        marg = bt.loc["LSTM"] - bt.loc["LogisticRegression"]
        lines.append(r"LSTM $-$ Logistic & " +
                     " & ".join(f"${marg[b]:+.4f}$" for b in bands) + r" \\")
        if len(boot.get("margins", pd.DataFrame())):
            mi = boot["margins"].set_index("band")
            cells_ci = []
            for b in bands:
                if b in mi.index:
                    cells_ci.append(f"$[{mi.loc[b, 'ci_lo']:+.4f},\\,{mi.loc[b, 'ci_hi']:+.4f}]$")
                else:
                    cells_ci.append("--")
            lines.append(r"\quad 95\% CI & " + " & ".join(cells_ci) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_band.tex", "w").write("\n".join(lines) + "\n")

    # Table: LSTM seed stability
    if len(seeds):
        lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
                 r"Sector & Mean AUC & SD & Min & Max \\", r"\midrule"]
        for s, g in seeds.groupby(COL_SECTOR, observed=True):
            lines.append(f"{s} & {g.auc.mean():.4f} & {g.auc.std(ddof=1):.4f} & "
                         f"{g.auc.min():.4f} & {g.auc.max():.4f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{out}/tab_lstm_seeds.tex", "w").write("\n".join(lines) + "\n")

    # Table: McNemar, every pair in every sector (longtable)
    if len(mcn):
        head = (r"Sector & Model A & Model B & $b$ & $c$ & $p$ & "
                r"$p_{\text{Holm}}$ \\")
        lines = [r"\begin{longtable}{lllrrrr}",
                 r"\caption{McNemar tests of paired predictions for every model "
                 r"pair in every sector, on the common evaluation support. "
                 r"$b$ and $c$ are the discordant counts; $p$-values are "
                 r"Holm-adjusted across the whole family.}\label{tab:mcnemar}\\",
                 r"\toprule", head, r"\midrule", r"\endfirsthead",
                 r"\toprule", head, r"\midrule", r"\endhead",
                 r"\midrule \multicolumn{7}{r}{\emph{continued on next page}}\\",
                 r"\endfoot", r"\bottomrule", r"\endlastfoot"]
        for _, r in mcn.iterrows():
            lines.append(f"{r.sector} & {r.model_a} & {r.model_b} & {int(r.b)} & "
                         f"{int(r.c)} & {r.p:.4f} & {r.p_holm:.4f} \\\\")
        lines.append(r"\end{longtable}")
        open(f"{out}/tab_mcnemar.tex", "w").write("\n".join(lines) + "\n")

    # Table: per-ticker robustness
    if len(rob_summary):
        lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
                 r"Sector & Band & $n$ & Per-ticker AUC & Per-ticker acc. & "
                 r"\% AUC $>0.5$ & Pooled AUC \\", r"\midrule"]
        pooled_auc = pooled.groupby(COL_SECTOR, observed=True)["auc"].mean()
        for s, r in rob_summary.iterrows():
            lines.append(f"{s} & {r.band} & {int(r.n_tickers)} & {r.perticker_auc:.4f} & "
                         f"{r.perticker_acc:.4f} & {r.pct_auc_gt_50:.1f} & "
                         f"{pooled_auc.get(s, float('nan')):.4f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{out}/tab_perticker.tex", "w").write("\n".join(lines) + "\n")

    # Table: Hurst sign sensitivity
    lines = [r"\begin{tabular}{lrrl}", r"\toprule",
             r"Sector & SCI ($d_{\text{Hurst}}=-1$) & SCI ($d_{\text{Hurst}}=+1$) & Band \\",
             r"\midrule"]
    for s, r in sci_flip.iterrows():
        lines.append(f"{s} & ${r.SCI:+.3f}$ & ${r.SCI_hurst_plus:+.3f}$ & {r.band} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_hurst_sensitivity.tex", "w").write("\n".join(lines) + "\n")


def write_numbers(out: str) -> None:
    with open(f"{out}/numbers.tex", "w") as f:
        f.write("% Auto-generated by pipeline.py -- do not edit by hand.\n")
        f.write(f"% {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(r"\renewcommand{\NumbersAreStale}{false}" + "\n")
        for k, v in sorted(NUMBERS.items()):
            f.write("\\renewcommand{\\%s}{%s}\n" % (k, v))
    note(f"    wrote {out}/numbers.tex with {len(NUMBERS)} macros")


_ROB_SUMMARY = pd.DataFrame()


def _run_robustness_block(df, feat, sci, pooled, models, n_seeds):
    """Isolated so a failure here cannot cost the main results."""
    global _ROB_SUMMARY
    if not RUN_ROBUSTNESS:
        return
    rob_detail, _ROB_SUMMARY = robustness_per_ticker(feat, sci)
    rob_summary = _ROB_SUMMARY
    if len(rob_detail):
        rob_detail.to_csv(f"{OUT_DIR}/robustness_perticker.csv", index=False)
        rob_summary.to_csv(f"{OUT_DIR}/robustness_by_sector.csv")
        macro("PerTickerAuc", rob_detail["auc"].mean())
        macro("PooledAuc", pooled["auc"].mean())
        macro("PerTickerPctAbove",
              100 * (rob_detail.groupby("ticker")["auc"].mean() > 0.5).mean(), "{:.0f}")
        macro("PerTickerBest", rob_detail["auc"].max(), "{:.3f}")
        macro("NTickersRobust", str(rob_detail["ticker"].nunique()))

    # Hurst min_n sensitivity
    sci_h = compute_sci(df, hurst_min_n=HURST_MIN_N_ROBUST, record=False)
    tau, _ = stats.kendalltau(sci["SCI"], sci_h.reindex(sci.index)["SCI"])
    sci_h.to_csv(f"{OUT_DIR}/robustness_sci_hurst_minn.csv")
    macro("HurstMinNTau", tau)
    macro("HurstMinNRobust", str(HURST_MIN_N_ROBUST))

    # lag-0 feature sensitivity, tabular models only (cheap)
    feat0 = build_features(df, include_lag0=True, record=False)
    tab = [m for m in models if m != "LSTM"]
    rows = []
    for sector, fs in feat0.groupby(COL_SECTOR, sort=False):
        p = run_sector(fs, tab, n_seeds_lstm=0)
        if len(p):
            rows.append(p)
    if rows:
        p0 = pd.concat(rows, ignore_index=True)
        m0 = metrics_by(p0, ["model", COL_SECTOR])
        m0.to_csv(f"{OUT_DIR}/robustness_lag0.csv", index=False)
        base = pooled[pooled["model"].isin(tab)]["auc"].mean()
        macro("LagZeroAuc", m0["auc"].mean())
        macro("LagZeroBaseAuc", base)
        macro("LagZeroDelta", m0["auc"].mean() - base, "{:+.4f}")



# =========================================================================== #
# MAIN                                                                        #
# =========================================================================== #
def main() -> None:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(RANDOM_STATE)

    n_seeds = 1 if QUICK_MODE else N_SEEDS_LSTM
    n_boot = 50 if QUICK_MODE else BOOTSTRAP_B

    note("environment")
    note(f"    python {platform.python_version()} | numpy {np.__version__} | "
         f"pandas {pd.__version__} | xgboost {'yes' if _HAS_XGB else 'NO'} | "
         f"torch {'yes' if _HAS_TORCH else 'NO'}")
    macro("SeedValue", str(RANDOM_STATE))
    macro("NFolds", str(N_SPLITS))
    macro("NSeedsLSTM", str(n_seeds))
    macro("BootstrapB", str(n_boot))
    macro("SeqLen", str(SEQ_LEN))
    macro("LSTMEpochs", str(LSTM_EPOCHS))
    macro("LSTMBatch", str(LSTM_BATCH))
    macro("LSTMHidden", str(LSTM_HIDDEN))
    macro("MinHistorySCI", str(MIN_HISTORY_SCI))

    note("1/8  loading data")
    df = load_data()
    if QUICK_MODE:
        keep = df[COL_SECTOR].drop_duplicates().head(3)
        df = df[df[COL_SECTOR].isin(keep)]
        note(f"    QUICK_MODE: {list(keep)}")

    note("2/8  Sectoral Complexity Index")
    sci = compute_sci(df)
    sci.to_csv(f"{OUT_DIR}/sci_by_sector.csv")
    sci[["SCI", "SCI_hurst_plus", "band"]].to_csv(f"{OUT_DIR}/sci_hurst_sensitivity.csv")
    print(sci[["SCI", "band"]].round(4).to_string())

    note("3/8  features")
    feat = build_features(df)

    note("4/8  walk-forward forecasting")
    models = available_models()
    if len(models) < 4:
        note("    [WARN] fewer than four models available. Macros for the "
             "missing model will NOT be written to numbers.tex, so the thesis "
             "would silently keep its previous values. Fix the install first.")
    macro("NModels", str(len(models)))
    macro("ModelList", ", ".join(models))
    parts_dir = f"{OUT_DIR}/pred_parts"
    os.makedirs(parts_dir, exist_ok=True)
    preds, cache_path = (_load_frame(f"{OUT_DIR}/predictions")
                         if REUSE_PREDICTIONS else (None, None))
    if preds is not None:
        note(f"    reusing {len(preds):,} cached predictions from {cache_path}")
        note("    (delete results/predictions.* or set REUSE_PREDICTIONS=False "
             "to refit)")
    else:
        # One checkpoint per sector.  On a free-tier runtime that can disconnect
        # mid-run, this is the difference between losing four minutes and losing
        # two hours: re-running picks up from the first sector without a part
        # file on disk.
        all_preds = []
        sectors = list(feat.groupby(COL_SECTOR, sort=False))
        for i, (sector, fs) in enumerate(sectors, 1):
            stem = f"{parts_dir}/" + "".join(
                ch if ch.isalnum() else "_" for ch in str(sector))
            part, part_path = (_load_frame(stem) if REUSE_PREDICTIONS
                               else (None, None))
            if part is not None:
                note(f"    [{i}/{len(sectors)}] {sector}  -- checkpoint found, "
                     f"skipping ({len(part):,} rows)")
                all_preds.append(part)
                continue
            note(f"    [{i}/{len(sectors)}] {sector}  ({len(fs):,} rows)")
            part = run_sector(fs, models, n_seeds_lstm=n_seeds)
            if not len(part):
                note(f"    [warn] {sector} produced no predictions")
                continue
            written = _save_frame(part, stem)
            note(f"        checkpoint -> {written}")
            all_preds.append(part)

        if not all_preds:
            raise SystemExit("no predictions produced; nothing to analyse")
        preds = pd.concat(all_preds, ignore_index=True)
        preds = restrict_to_common_support(preds)
        written = _save_frame(preds, f"{OUT_DIR}/predictions")
        note(f"    saved {len(preds):,} predictions -> {written}")

    note("5/8  metrics")
    # PRIMARY SPECIFICATION: fold-blocked (report Section 3.7.3). AUC is a
    # ranking statistic, so computing it over the pooled predictions of several
    # separately fitted folds estimates a quantity belonging to no fitted model,
    # and is biased downwards when the target base rate drifts between folds.
    per_seed = metrics_by_foldblocked(preds, ["model", COL_SECTOR, "seed"])
    per_seed.to_csv(f"{OUT_DIR}/performance_by_model_sector_seed.csv", index=False)
    lstm_seeds = per_seed[per_seed["model"] == "LSTM"]
    lstm_seeds.to_csv(f"{OUT_DIR}/lstm_seed_stability.csv", index=False)

    pooled = collapse_lstm_seeds(per_seed, ["model", COL_SECTOR])
    pooled.to_csv(f"{OUT_DIR}/performance_by_model_sector.csv", index=False)
    pooled.to_csv(f"{OUT_DIR}/performance_foldblocked.csv", index=False)

    # Retained only to support the comparison in Appendix F of the report.
    across = collapse_lstm_seeds(
        metrics_by(preds, ["model", COL_SECTOR, "seed"]), ["model", COL_SECTOR])
    across.to_csv(f"{OUT_DIR}/performance_pooled.csv", index=False)
    cmp_df = (across[[COL_SECTOR, "model", "auc"]].rename(columns={"auc": "pooled"})
              .merge(pooled[[COL_SECTOR, "model", "auc"]]
                     .rename(columns={"auc": "fold_blocked"}),
                     on=[COL_SECTOR, "model"]))
    cmp_df["gap"] = cmp_df["fold_blocked"] - cmp_df["pooled"]
    cmp_df["side_flip"] = (cmp_df["pooled"] < .5) != (cmp_df["fold_blocked"] < .5)
    cmp_df.to_csv(f"{OUT_DIR}/foldblock_comparison.csv", index=False)
    macro("AggMeanAbsGap", cmp_df["gap"].abs().mean())
    macro("AggSideFlips", str(int(cmp_df["side_flip"].sum())))
    macro("AggWorstGap", cmp_df["gap"].max(), "{:+.4f}")
    macro("PooledSpread", across.groupby(COL_SECTOR)["auc"].mean()
                                .pipe(lambda s: s.max() - s.min()))
    for _lbl, _fr in (("RealEstatePooled", across), ("RealEstateFoldBlocked", pooled)):
        _v = _fr.loc[_fr[COL_SECTOR] == "Real Estate", "auc"]
        if len(_v):
            macro(_lbl, _v.mean())
    _tab = (cmp_df.reindex(cmp_df["gap"].abs().sort_values(ascending=False).index)
                  .head(8))
    with open(f"{OUT_DIR}/tab_aggregation.tex", "w") as _f:
        _f.write("\\begin{tabular}{llrrrc}\n\\toprule\n")
        _f.write("Sector & Model & Pooled & Fold-blocked & Gap & Side change "
                 "\\\\\n\\midrule\n")
        for _, _r in _tab.iterrows():
            _f.write(f"{_r[COL_SECTOR]} & {_r['model']} & {_r['pooled']:.4f} & "
                     f"{_r['fold_blocked']:.4f} & ${_r['gap']:+.4f}$ & "
                     f"{'yes' if _r['side_flip'] else 'no'} \\\\\n")
        _f.write("\\bottomrule\n\\end{tabular}\n")

    fold_seed = metrics_by(preds, ["model", COL_SECTOR, "fold", "seed"])
    fold_perf = collapse_lstm_seeds(fold_seed, ["model", COL_SECTOR, "fold"])
    fold_perf.to_csv(f"{OUT_DIR}/performance_by_fold.csv", index=False)

    print(pooled.groupby("model", observed=True)[["accuracy", "auc", "f1", "mcc"]]
                .mean().round(4).to_string())

    beat_test = int((pooled["accuracy"] > pooled["baseline_test"]).sum())
    beat_train = int((pooled["accuracy"] > pooled["baseline_train"]).sum())
    macro("NCells", str(len(pooled)))
    macro("NBeatBaselineTest", str(beat_test))
    macro("NBeatBaselineTrain", str(beat_train))
    macro("NAucAboveHalf", str(int((pooled["auc"] > 0.5).sum())))
    macro("NMccPositive", str(int((pooled["mcc"] > 0).sum())))
    macro("AucMin", pooled["auc"].min(), "{:.4f}")
    macro("AucMax", pooled["auc"].max(), "{:.4f}")
    for m, g in pooled.groupby("model", observed=True):
        tag = m.replace("Regression", "Reg")
        macro(f"Mean{tag}Auc", g["auc"].mean())
        macro(f"Mean{tag}Acc", g["accuracy"].mean())
        macro(f"Mean{tag}FOne", g["f1"].mean())
        macro(f"Mean{tag}Mcc", g["mcc"].mean(), "{:+.4f}")
    if len(lstm_seeds):
        sd = lstm_seeds.groupby(COL_SECTOR, observed=True)["auc"].std(ddof=1)
        macro("LstmSeedSdMean", sd.mean())
        macro("LstmSeedSdMax", sd.max())

    note("6/8  hypothesis tests")
    rq1 = rq1_sector_heterogeneity(fold_perf, pooled)
    rq2 = rq2_complexity(pooled, sci)
    rq3 = rq3_models(pooled)
    mcn = mcnemar_all_pairs(preds)
    mcn.to_csv(f"{OUT_DIR}/mcnemar_pairs.csv", index=False)

    macro("RQoneChiSq", rq1["chi2"], "{:.2f}")
    macro("RQonePval", rq1["p"], "{:.3f}")
    macro("RQoneBlocks", str(rq1["n_blocks"]))
    macro("RQoneTreatments", str(rq1["k_treatments"]))
    macro("RQoneChiSqOld", rq1["chi2_pooled_blocks"], "{:.2f}")
    macro("RQonePvalOld", rq1["p_pooled_blocks"], "{:.3f}")
    macro("RQoneSpread", rq1["sector_mean_spread"])
    macro("RQtwoRho", rq2["rho"], "{:+.3f}")
    macro("RQtwoPval", rq2["p"], "{:.3f}")
    macro("RQtwoMWU", rq2["mannwhitney_U_H2"], "{:.1f}")
    macro("RQtwoMWPval", rq2["mannwhitney_p_H2"], "{:.3f}")
    macro("RQtwoMWPvalTwo", rq2["mannwhitney_p_two_sided"], "{:.3f}")
    for b, v in rq2["band_means"].items():
        macro(f"BandMean{str(b).capitalize()}", v)
    for rec in rq2["per_model"]:
        tag = rec["model"].replace("Regression", "Reg")
        macro(f"Rho{tag}", rec["rho"], "{:+.3f}")
        macro(f"Rho{tag}P", rec["p"], "{:.3f}")
        macro(f"Rho{tag}PHolm", rec["p_holm"], "{:.3f}")
    macro("RQthreeChiSq", rq3["chi2"], "{:.2f}")
    macro("RQthreePval", rq3["p"], "{:.3f}")
    macro("RQthreeCD", rq3["critical_difference"], "{:.2f}")
    macro("RQthreeMaxGap", rq3["max_rank_gap"], "{:.2f}")
    for m, r in rq3["avg_ranks"].items():
        macro(f"Rank{m.replace('Regression', 'Reg')}", r, "{:.2f}")
    macro("NMcNemarTests", str(len(mcn)))
    macro("NMcNemarSig", str(int(mcn["significant_holm"].sum()) if len(mcn) else 0))

    note("7/8  date-block bootstrap  (this is the slow part)")
    boot = date_block_bootstrap(preds, sci, B=n_boot)
    boot["cells"].to_csv(f"{OUT_DIR}/auc_bootstrap_ci.csv", index=False)
    boot["bands"].to_csv(f"{OUT_DIR}/auc_bootstrap_bands.csv", index=False)
    boot["margins"].to_csv(f"{OUT_DIR}/auc_bootstrap_margins.csv", index=False)
    cells = boot["cells"]
    macro("NCellsAboveChance", str(int((cells["ci_lo"] > 0.5).sum())))
    macro("NCellsBelowChance", str(int((cells["ci_hi"] < 0.5).sum())))
    macro("NCellsCIExcludeHalf", str(int(cells["excludes_half"].sum())))
    macro("NCellsHolmSig", str(int((cells["p_holm"] < ALPHA).sum())))
    macro("RhoCILo", boot["rho_ci"][0], "{:+.3f}")
    macro("RhoCIHi", boot["rho_ci"][1], "{:+.3f}")
    if len(boot["margins"]):
        for b, r in boot["margins"].set_index("band").iterrows():
            macro(f"Margin{str(b).capitalize()}", r["mean"], "{:+.4f}")
            macro(f"Margin{str(b).capitalize()}CILo", r["ci_lo"], "{:+.4f}")
            macro(f"Margin{str(b).capitalize()}CIHi", r["ci_hi"], "{:+.4f}")

    ad = adaptive_selection(fold_perf, pooled, sci)
    macro("AdaptiveOOS", ad["oos_adaptive_mean"])
    macro("AdaptiveOOSFixed", ad["oos_best_fixed_mean"])
    macro("AdaptiveOOSFixedModel", ad["oos_best_fixed_model"])
    macro("AdaptiveOOSUplift", ad["oos_uplift"], "{:+.4f}")
    macro("AdaptiveInSample", ad["in_sample_adaptive_mean"])
    macro("AdaptiveInSampleFixed", ad["in_sample_best_fixed_mean"])
    macro("AdaptiveInSampleFixedModel", ad["in_sample_best_fixed_model"])
    macro("AdaptiveOracle", ad["per_sector_oracle_mean"])
    _lb = (pooled.assign(band=pooled[COL_SECTOR].map(sci["band"].astype(str)))
                 .query("model == 'LSTM'").groupby("band")["auc"].mean())
    if {"low", "medium", "high"} <= set(_lb.index):
        for b in ("low", "medium", "high"):
            macro(f"LstmBand{b.capitalize()}", _lb[b])
        macro("LstmBandRise", _lb["high"] - _lb["low"], "{:+.4f}")

    for b, v in ad["band_to_model"].items():
        macro(f"BestModel{str(b).capitalize()}", v["model"])

    note("8/8  robustness")
    rob_summary = pd.DataFrame()
    try:
        _run_robustness_block(df, feat, sci, pooled, models, n_seeds)
        rob_summary = _ROB_SUMMARY
    except Exception as e:
        note(f"    [WARN] robustness block failed ({type(e).__name__}: {e}). "
             f"Main results and LaTeX artefacts are unaffected; the robustness "
             f"macros keep their fallback values.")

    note("writing LaTeX artefacts")
    write_tables(OUT_DIR, sci, pooled, boot, mcn, lstm_seeds, rob_summary,
                 sci[["SCI", "SCI_hurst_plus", "band"]])
    write_numbers(OUT_DIR)

    summary = {"rq1": rq1, "rq2": {k: v for k, v in rq2.items() if k != "table"},
               "rq3": rq3, "adaptive": ad,
               "bootstrap": {"B": n_boot, "rho_ci": boot["rho_ci"],
                             "cells_excluding_half": int(cells["excludes_half"].sum())},
               "mcnemar_significant": int(mcn["significant_holm"].sum()) if len(mcn) else 0,
               "runtime_minutes": round((time.time() - t0) / 60, 1),
               "config": {"n_splits": N_SPLITS, "n_seeds_lstm": n_seeds,
                          "include_lag0": INCLUDE_LAG0, "seq_len": SEQ_LEN,
                          "lstm_epochs": LSTM_EPOCHS, "bootstrap_B": n_boot,
                          "random_state": RANDOM_STATE, "quick_mode": QUICK_MODE}}
    with open(f"{OUT_DIR}/rq_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"RQ1  chi2={rq1['chi2']:.2f}  p={rq1['p']:.4f}  "
          f"({rq1['n_blocks']} blocks x {rq1['k_treatments']} sectors)")
    print(f"     v1 specification: chi2={rq1['chi2_pooled_blocks']:.2f}  "
          f"p={rq1['p_pooled_blocks']:.4f}  ({rq1['n_blocks_pooled']} blocks)")
    print(f"RQ2  rho={rq2['rho']:+.3f}  p={rq2['p']:.4f}  "
          f"bootstrap 95% CI [{boot['rho_ci'][0]:+.3f}, {boot['rho_ci'][1]:+.3f}]")
    print(f"     Mann-Whitney low>high p={rq2['mannwhitney_p_H2']:.4f}  "
          f"(two-sided {rq2['mannwhitney_p_two_sided']:.4f})")
    print(f"RQ3  chi2={rq3['chi2']:.2f}  p={rq3['p']:.4f}  CD={rq3['critical_difference']:.2f}  "
          f"max gap={rq3['max_rank_gap']:.2f}")
    print(f"     McNemar significant after Holm: "
          f"{int(mcn['significant_holm'].sum()) if len(mcn) else 0}/{len(mcn)}")
    print(f"CI   cells whose 95% CI excludes 0.5: {int(cells['excludes_half'].sum())}"
          f"/{len(cells)}  (above {int((cells['ci_lo'] > 0.5).sum())}, "
          f"below {int((cells['ci_hi'] < 0.5).sum())})")
    print(f"ADPT out-of-sample {ad['oos_adaptive_mean']:.4f} vs best fixed "
          f"{ad['oos_best_fixed_mean']:.4f} -> {ad['oos_uplift']:+.4f}   "
          f"(in-sample {ad['in_sample_adaptive_mean']:.4f})")
    print("=" * 70)
    note(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
