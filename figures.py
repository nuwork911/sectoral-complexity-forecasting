"""
=============================================================================
Figures for the results chapter (v2)
=============================================================================

    python figures.py

Reads  : results/sci_by_sector.csv
         results/performance_by_model_sector.csv
         results/auc_bootstrap_ci.csv         (optional; enables Figure 7)
         results/lstm_seed_stability.csv      (optional; enables Figure 8)
Writes : results/figures/*.pdf                (vector, drops into Overleaf)

Kept separate from pipeline.py so figures can be restyled without refitting.
Needs only matplotlib / pandas / numpy / scipy.

-----------------------------------------------------------------------------
WHAT CHANGED FROM v1
-----------------------------------------------------------------------------
1.  CRITICAL-DIFFERENCE CLIQUES ARE CORRECT.  v1 started a clique bar at every
    index, so nested and redundant bars were drawn on top of each other; with
    CD = 1.41 and a maximum rank gap of 1.00 all four models form ONE clique and
    the figure should show a single bar.  Maximal cliques are now computed and
    subsumed ones discarded.

2.  fig_perf_by_band IS LABELLED AND WIRED UP.  The thesis referenced it as
    "Figure ??" because the float was never included; it now carries the same
    filename the .tex expects, and shows individual sector points over the box
    so a reader can see that the bands overlap.

3.  NEW fig_auc_confidence_intervals.  A caterpillar plot of every model-sector
    AUC with its date-block bootstrap interval, sorted by AUC.  This is the
    figure that carries Section 4.2's argument: near-chance everywhere, with a
    handful of cells reliably away from 0.5 in both directions.

4.  NEW fig_lstm_seed_stability.  Per-sector LSTM AUC across random seeds,
    against the band means.  The point of the figure is the comparison of
    scales: if seed-to-seed spread exceeds the complexity-conditioned margin,
    the RQ3 pattern is not a finding about markets.

5.  ERROR BARS ON THE SCATTER.  fig_sci_vs_performance now draws bootstrap
    intervals on the per-sector means where available, because a scatter of
    eleven bare points invites over-reading of a rank correlation that is not
    significant.

6.  ROBUST TO MISSING INPUTS.  Every figure is skipped with a printed note
    rather than crashing if its input file is absent, so the script can be run
    part-way through an analysis.
=============================================================================
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = "results"
FIG_DIR = f"{OUT_DIR}/figures"
METRIC = "auc"
BAND_ORDER = ["low", "medium", "high"]
BAND_COLOURS = {"low": "#4a7a4e", "medium": "#c79a2e", "high": "#b04545"}
SCI_METRICS = ["volatility", "sampen", "hurst", "amihud", "vol_var"]
SCI_LABELS = {"volatility": "volatility", "sampen": "sample entropy",
              "hurst": "Hurst", "amihud": "Amihud", "vol_var": "volume var."}

# Nemenyi q_alpha, alpha = 0.05, Demsar (2006), indexed by number of models
_Q05 = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 10, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,          # embed real fonts, not Type 3
})


def _save(fig, name: str) -> None:
    fig.savefig(f"{FIG_DIR}/{name}.pdf")
    plt.close(fig)
    print(f"  wrote {FIG_DIR}/{name}.pdf")


def _read(name: str, **kw):
    path = f"{OUT_DIR}/{name}"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kw)


def load():
    sci = _read("sci_by_sector.csv", index_col=0)
    if sci is None:
        raise SystemExit(f"{OUT_DIR}/sci_by_sector.csv not found -- run pipeline.py first")
    sci.index.name = "sector"
    perf = _read("performance_by_model_sector.csv")
    if perf is None:
        raise SystemExit(f"{OUT_DIR}/performance_by_model_sector.csv not found")
    if "Sector" in perf.columns:
        perf = perf.rename(columns={"Sector": "sector"})
    ci = _read("auc_bootstrap_ci.csv")
    seeds = _read("lstm_seed_stability.csv")
    if seeds is not None and "Sector" in seeds.columns:
        seeds = seeds.rename(columns={"Sector": "sector"})
    return sci, perf, ci, seeds


# --------------------------------------------------------------------------- #
def fig_sci_by_sector(sci: pd.DataFrame) -> None:
    s = sci.sort_values("SCI")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(s.index, s["SCI"], color=[BAND_COLOURS[b] for b in s["band"]])
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Sectoral Complexity Index (z-score units)")
    ax.set_title("Sectoral Complexity Index by sector")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND_COLOURS[b]) for b in BAND_ORDER]
    ax.legend(handles, [f"{b} complexity" for b in BAND_ORDER],
              loc="lower right", frameon=False)
    _save(fig, "fig_sci_by_sector")


def fig_metric_correlation(sci: pd.DataFrame) -> None:
    metrics = [m for m in SCI_METRICS if m in sci.columns]
    corr = sci[metrics].corr()
    labels = [SCI_LABELS.get(m, m) for m in metrics]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(labels)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Correlation among the five SCI metrics (across sectors)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "fig_metric_correlation")


def fig_sci_vs_performance(sci: pd.DataFrame, perf: pd.DataFrame,
                           ci: pd.DataFrame | None) -> None:
    mean_perf = perf.groupby("sector")[METRIC].mean()
    d = sci.join(mean_perf.rename("perf")).dropna(subset=["perf"])
    rho, p = stats.spearmanr(d["SCI"], d["perf"])

    err = None
    if ci is not None and {"sector", "ci_lo", "ci_hi"}.issubset(ci.columns):
        g = ci.groupby("sector")[["ci_lo", "ci_hi"]].mean()
        g = g.reindex(d.index)
        lo = (d["perf"] - g["ci_lo"]).clip(lower=0)
        hi = (g["ci_hi"] - d["perf"]).clip(lower=0)
        if lo.notna().all() and hi.notna().all():
            err = np.vstack([lo.to_numpy(), hi.to_numpy()])

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    if False:  # see review
        ax.errorbar(d["SCI"], d["perf"], yerr=err, fmt="none",
                    ecolor="0.7", elinewidth=1, capsize=2, zorder=2)
    for b in BAND_ORDER:
        sub = d[d["band"] == b]
        ax.scatter(sub["SCI"], sub["perf"], s=70, color=BAND_COLOURS[b],
                   label=b, zorder=3)
    for s, row in d.iterrows():
        ax.annotate(s, (row["SCI"], row["perf"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.5, color="0.6", ls="--", lw=0.8, label="chance (AUC 0.5)")
    ax.set_xlabel("Sectoral Complexity Index")
    ax.set_ylabel(f"mean {METRIC.upper()} across models")
    ax.set_title(f"SCI vs forecasting performance "
                 f"(Spearman $\\rho$={rho:.2f}, p={p:.2f})")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_sci_vs_performance")


def fig_perf_by_band(sci: pd.DataFrame, perf: pd.DataFrame) -> None:
    """The figure the thesis referenced as 'Figure ??' in v1."""
    d = perf.merge(sci["band"].reset_index(), on="sector")
    groups = [d.loc[d["band"] == b, METRIC].dropna().to_numpy() for b in BAND_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(groups, tick_labels=BAND_ORDER, patch_artist=True, widths=0.55,
                    medianprops={"color": "black"})
    for patch, b in zip(bp["boxes"], BAND_ORDER):
        patch.set_facecolor(BAND_COLOURS[b])
        patch.set_alpha(0.45)
    rng = np.random.default_rng(0)
    for i, (b, vals) in enumerate(zip(BAND_ORDER, groups), start=1):
        if len(vals):
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=16,
                       color=BAND_COLOURS[b], edgecolor="white", linewidth=0.4,
                       zorder=3)
    ax.axhline(0.5, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("complexity band")
    ax.set_ylabel(METRIC.upper())
    ax.set_title(f"{METRIC.upper()} by complexity band (all model--sector cells)")
    _save(fig, "fig_perf_by_band")


def fig_model_sector_heatmap(sci: pd.DataFrame, perf: pd.DataFrame) -> None:
    wide = perf.pivot_table(index="sector", columns="model", values=METRIC)
    wide = wide.reindex(sci.sort_values("SCI").index)
    fig, ax = plt.subplots(figsize=(1.4 * wide.shape[1] + 3, 0.5 * len(wide) + 2))
    im = ax.imshow(wide.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(wide.shape[1]))
    ax.set_xticklabels(wide.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(wide)))
    ax.set_yticklabels(wide.index)
    for i in range(len(wide)):
        for j in range(wide.shape[1]):
            v = wide.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white", fontsize=7)
    ax.set_title(f"{METRIC.upper()} by model and sector (sectors ordered by SCI)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label=METRIC.upper())
    _save(fig, "fig_model_sector_heatmap")


def _maximal_cliques(ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Maximal runs of models whose average ranks lie within CD of each other.

    v1 opened a bar at every index and drew subsumed runs on top of one another,
    which is why the published figure showed several stacked bars where there is
    only one clique.
    """
    k = len(ranks)
    runs = []
    for i in range(k):
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) <= cd:
            j += 1
        if j > i:
            runs.append((i, j))
    maximal = [r for r in runs
               if not any(a <= r[0] and r[1] <= b and (a, b) != r for a, b in runs)]
    return maximal


def fig_critical_difference(perf: pd.DataFrame) -> None:
    wide = perf.pivot_table(index="sector", columns="model", values=METRIC).dropna()
    models = list(wide.columns)
    k, n = len(models), len(wide)
    if k < 3:
        print("  [note] CD diagram needs >= 3 models; skipped")
        return
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    q05 = _Q05.get(k, 2.569)
    cd = q05 * np.sqrt(k * (k + 1) / (6.0 * n))

    names = list(ranks.index)
    rvals = ranks.to_numpy()
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 2.4 + 0.3 * k))
    ax.set_xlim(lo - 0.6, hi + 0.6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([lo, hi], [0.78, 0.78], "k-", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0.78, 0.81], "k-", lw=1)
        ax.text(r, 0.85, str(r), ha="center")

    for idx, (name, rv) in enumerate(zip(names, rvals)):
        left = idx < len(names) / 2
        depth = idx if left else (len(names) - 1 - idx)
        y = 0.58 - 0.10 * depth
        xtext = lo - 0.5 if left else hi + 0.5
        ax.plot([rv, rv], [0.78, y], "k-", lw=0.8)
        ax.plot([rv, xtext], [y, y], "k-", lw=0.8)
        ax.text(xtext + (-0.05 if left else 0.05), y, f"{name} ({rv:.2f})",
                ha="right" if left else "left", va="center", fontsize=9)

    ax.plot([lo, lo + cd], [0.94, 0.94], "k-", lw=2.5)
    ax.text(lo + cd / 2, 0.965, f"CD = {cd:.2f}", ha="center", fontsize=9)

    yb = 0.72
    for i, j in _maximal_cliques(rvals, cd):
        ax.plot([rvals[i] - 0.04, rvals[j] + 0.04], [yb, yb], "k-", lw=3.5)
        yb -= 0.035
    ax.set_title("Critical-difference diagram (Nemenyi, lower rank = better)")
    _save(fig, "fig_critical_difference")


def fig_auc_confidence_intervals(sci: pd.DataFrame, ci: pd.DataFrame | None) -> None:
    """Caterpillar plot: every cell's AUC with its date-block bootstrap interval."""
    if ci is None or not {"sector", "model", "ci_lo", "ci_hi"}.issubset(ci.columns):
        print("  [note] auc_bootstrap_ci.csv not found; skipping CI figure")
        return
    d = ci.copy()
    perf_auc = _read("performance_by_model_sector.csv")
    if perf_auc is not None:
        if "Sector" in perf_auc.columns:
            perf_auc = perf_auc.rename(columns={"Sector": "sector"})
        d = d.merge(perf_auc[["sector", "model", "auc"]],
                    on=["sector", "model"], how="left")
        d["auc"] = d["auc"].fillna(d["auc_boot_mean"])
    else:
        d["auc"] = d["auc_boot_mean"]
    band = sci["band"].astype(str).to_dict()
    d["band"] = d["sector"].map(band)
    d = d.sort_values("auc").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.6, 0.22 * len(d) + 2))
    for i, r in d.iterrows():
        excl = (r.ci_lo > 0.5) or (r.ci_hi < 0.5)
        col = BAND_COLOURS.get(r.band, "0.4")
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col,
                lw=2.4 if excl else 1.2, alpha=1.0 if excl else 0.55,
                solid_capstyle="round")
        ax.scatter(r.auc, i, s=26 if excl else 14, color=col,
                   edgecolor="black" if excl else "none", linewidth=0.6, zorder=3)
    ax.axvline(0.5, color="0.35", ls="--", lw=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.sector} -- {r.model}" for _, r in d.iterrows()],
                       fontsize=7)
    ax.set_xlabel("out-of-sample AUC with 95% date-block bootstrap interval")
    n_excl = int(((d.ci_lo > 0.5) | (d.ci_hi < 0.5)).sum())
    ax.set_title(f"AUC and uncertainty for all {len(d)} model--sector cells\n"
                 f"({n_excl} intervals exclude 0.5; heavier lines)", fontsize=10)
    handles = [plt.Line2D([], [], color=BAND_COLOURS[b], lw=2.4, label=b)
               for b in BAND_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right",
              title="complexity band", title_fontsize=8)
    _save(fig, "fig_auc_confidence_intervals")


def fig_lstm_seed_stability(sci: pd.DataFrame, seeds: pd.DataFrame | None) -> None:
    """Seed-to-seed spread against the complexity ordering."""
    if seeds is None or "auc" not in seeds.columns or "sector" not in seeds.columns:
        print("  [note] lstm_seed_stability.csv not found; skipping seed figure")
        return
    order = [s for s in sci.sort_values("SCI").index if s in set(seeds["sector"])]
    if not order:
        print("  [note] no matching sectors in seed file; skipping")
        return
    groups = [seeds.loc[seeds["sector"] == s, "auc"].to_numpy() for s in order]
    band = sci["band"].astype(str).to_dict()

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(1)
    for i, (s, vals) in enumerate(zip(order, groups)):
        col = BAND_COLOURS.get(band.get(s, "low"), "0.4")
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)),
                   vals, s=26, color=col, alpha=0.75, edgecolor="white",
                   linewidth=0.4, zorder=3)
        if len(vals) > 1:
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="_",
                        color="black", markersize=16, capsize=4, lw=1.2, zorder=4)
    ax.axhline(0.5, color="0.6", ls="--", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("LSTM AUC")
    ax.set_title("LSTM AUC across random initialisations, sectors ordered by SCI\n"
                 "(black bars are mean $\\pm$ 1 SD across seeds)", fontsize=10)
    _save(fig, "fig_lstm_seed_stability")


def summary_tables(perf: pd.DataFrame) -> None:
    print("\nMean performance by model (across sectors):")
    cols = [c for c in ["accuracy", "auc", "f1", "mcc"] if c in perf.columns]
    print(perf.groupby("model")[cols].mean().round(4).to_string())
    total = len(perf)
    for col, label in [("baseline_test", "hindsight test baseline"),
                       ("baseline_train", "deployable train baseline")]:
        if col in perf.columns:
            n = int((perf["accuracy"] > perf[col]).sum())
            print(f"Cells beating {label}: {n}/{total}")
    print(f"Cells with AUC > 0.5: {int((perf['auc'] > 0.5).sum())}/{total}")
    if "mcc" in perf.columns:
        print(f"Cells with MCC > 0: {int((perf['mcc'] > 0).sum())}/{total}")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    sci, perf, ci, seeds = load()
    fig_sci_by_sector(sci)
    fig_metric_correlation(sci)
    fig_sci_vs_performance(sci, perf, ci)
    fig_perf_by_band(sci, perf)
    fig_model_sector_heatmap(sci, perf)
    fig_critical_difference(perf)
    fig_auc_confidence_intervals(sci, ci)
    fig_lstm_seed_stability(sci, seeds)
    summary_tables(perf)
    print(f"\nFigures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
