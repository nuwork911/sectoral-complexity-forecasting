"""fig_walk_forward.pdf -- expanding-window protocol diagram.

Fold boundaries are reproduced from the FEATURE PANEL's trading dates using the
same TimeSeriesSplit(N_SPLITS) call run_sector uses, then cross-checked against
the test windows actually present in predictions.parquet.

Why not derive everything from the predictions: that table holds test rows only,
so it cannot see where a fold's training window begins.

Writes to figures/ (TOP LEVEL) -- thesis.tex wants figures/fig_walk_forward.pdf,
not results/figures/.
"""
from __future__ import annotations
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np, pandas as pd
from matplotlib.patches import Patch
from sklearn.model_selection import TimeSeriesSplit

import pipeline as P

RES_DIR = "results_full"
OUT_DIR = "figures"
TRAIN_C, TEST_C = "#8c9bab", "#b04545"
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})

def panel_folds():
    df = P.load_data()
    feat = P.build_features(df, record=False)
    ds = pd.to_datetime(pd.Series(feat[P.COL_DATE].unique())).sort_values()
    ds = ds.reset_index(drop=True)
    dates = ds.to_list()          # list of pd.Timestamp -- formats and subtracts cleanly
    print(f"feature panel: {len(dates)} trading dates, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    rows = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=P.N_SPLITS).split(dates), 1):
        rows.append({"fold": fold,
                     "train_start": dates[tr[0]], "train_end": dates[tr[-1]],
                     "train_days": len(tr),
                     "test_start": dates[te[0]], "test_end": dates[te[-1]],
                     "test_days": len(te)})
    return pd.DataFrame(rows).set_index("fold"), dates

def make_figure(w):
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(w)
    fig, ax = plt.subplots(figsize=(8.4, 0.72 * n + 1.6))
    for i, (fold, r) in enumerate(w.iterrows()):
        y = n - 1 - i
        ax.barh(y, r["train_end"] - r["train_start"], left=r["train_start"],
                height=0.52, color=TRAIN_C)
        ax.barh(y, r["test_end"] - r["test_start"], left=r["test_start"],
                height=0.52, color=TEST_C)
        mid = r["train_start"] + (r["train_end"] - r["train_start"]) / 2
        ax.text(mid, y, f"{r['train_days']:,} train", ha="center", va="center",
                color="white", fontsize=8.5)
        ax.annotate(f"{r['test_days']} test", xy=(r["test_end"], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=TEST_C)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Fold {f}" for f in reversed(list(w.index))])
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("trading date")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=TRAIN_C, label="training window (expanding)"),
                       Patch(color=TEST_C, label="out-of-sample test window")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.set_title("Expanding-window walk-forward protocol", pad=10)
    fig.savefig(f"{OUT_DIR}/fig_walk_forward.pdf"); plt.close(fig)
    print(f"  wrote {OUT_DIR}/fig_walk_forward.pdf")

if __name__ == "__main__":
    w, dates = panel_folds()
    show = w.copy()
    for c in ("train_start", "train_end", "test_start", "test_end"):
        show[c] = show[c].dt.strftime("%Y-%m-%d")
    print("\nfold boundaries:"); print(show.to_string())

    # real checks this time
    assert (w["train_end"] < w["test_start"]).all(), "training overlaps its test window"
    assert w["train_days"].is_monotonic_increasing, "training does not expand"
    assert (w["train_days"] > 0).all(), "a fold has no training data"
    assert w["test_days"].std() < 2, "test windows unequal"

    # cross-check against what was actually predicted
    pr = pd.read_parquet(f"{RES_DIR}/predictions.parquet")
    dc = "date" if "date" in pr.columns else "Date"
    pr[dc] = pd.to_datetime(pr[dc])
    act = pr.groupby("fold")[dc].agg(["min", "max", "nunique"])
    print("\ntest windows in predictions.parquet (union over sectors):")
    print(act.to_string())
    for f in w.index:
        d = abs((act.loc[f, "min"] - w.loc[f, "test_start"]).days)
        assert d <= 5, f"fold {f}: derived test_start off by {d} days"
    print("\nderived folds agree with the predictions to within 5 days")
    make_figure(w)
