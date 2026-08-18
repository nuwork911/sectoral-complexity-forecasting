# Sectoral complexity and machine-learning forecasts of daily stock price direction

Analysis code and generated results for an MSc dissertation examining whether the
structural complexity of an industry sector explains how well machine-learning
models forecast the direction of its daily price movements, and whether
complexity can guide model selection in a decision-support setting.

**Headline result is negative.** Across 44 model–sector combinations spanning
589,884 daily observations for 523 S&P 500 constituents over 2020–2025, no
configuration beats a hindsight majority-class baseline. Performance does differ
across sectors by a statistically reliable margin, but one spanning barely more
than one AUC point. A composite complexity index computed ex ante is not
associated with that variation in the direction hypothesised. A
complexity-conditioned model-selection policy, fitted and evaluated out of
sample, selects the same model in every complexity band and yields no gain.

## Contents

| Path | Description |
|---|---|
| `pipeline.py` | End-to-end analysis: loading, complexity index, features, walk-forward evaluation of four models, hypothesis tests, bootstrap, robustness checks. Emits the LaTeX macros and table bodies the report reads. |
| `figures.py` | All results figures, generated from stored results without refitting. |
| `make_walk_forward_fig.py` | The walk-forward protocol diagram, with fold boundaries read from the stored predictions. |
| `results/` | Every generated artefact: predictions, per-fold and per-sector performance, bootstrap intervals, paired tests, robustness outputs, and the machine-generated `numbers.tex`. |
| `ticker_sectors.xlsx` | Ticker-to-sector mapping. |

## Data

The price panel (`merged_data.csv`, 523 tickers, 589,886 rows, January 2020 to
December 2025) is **not distributed here.** It was provided for this project and
includes a column belonging to a separate study, so it is not mine to
redistribute. The panel is reconstructible from any daily OHLCV source; the
columns `pipeline.py` requires are documented at the top of that file, and
`ticker_sectors.xlsx` supplies the sector mapping.

`results/predictions.parquet` contains every out-of-sample prediction keyed by
sector, date, ticker, model, seed and fold, so every reported statistic can be
recomputed and checked without access to the source panel or a refit.

## Reproducing

```bash
pip install -r requirements.txt
python pipeline.py      # ~70 min on a T4; reuses cached predictions if present
python figures.py
python make_walk_forward_fig.py
```

A single seed (42) is propagated to Python, NumPy, scikit-learn, XGBoost and
PyTorch. The LSTM is **not** bit-reproducible on GPU, because cuDNN's recurrent
kernels have no deterministic path; it is therefore reported as a mean over five
seeds, with the seed-to-seed spread published alongside
(`results/lstm_seed_stability.csv`).

## Two methodological notes

**AUC is computed within fold, then averaged across folds.** A walk-forward
protocol fits one model per fold, so pooling predictions across folds and
computing a single AUC evaluates a ranking that mixes the outputs of several
differently calibrated models. Where the target's base rate drifts between folds
this is biased downwards. On these data the bias averages 0.0030 AUC but reaches
0.0227 in the worst cell and changes which side of chance 11 of 44 cells fall on
— enough to manufacture a spurious sub-chance sectoral outlier.
`results/foldblock_comparison.csv` quantifies it. This is a common practice and
worth checking in other walk-forward work.

**Bootstrap intervals resample whole trading dates, within fold.** Several
hundred tickers share each trading date and their errors are driven partly by a
common market factor, so resampling observations would understate every standard
error substantially.

## Caveats

Predictive information only — no transaction costs are modelled and no economic
evaluation is attempted. Hyperparameters are held at documented defaults rather
than tuned. Models are pooled within sectors, imposing one parameter set on all
constituents. The complexity index is computed over the whole sample period, so
it is ex ante with respect to the models but not with respect to time. A null
result for this index is not a null result for sectoral complexity as a
construct.

## Licence

Code released under the MIT Licence. The generated results in `results/` are
released under CC BY 4.0. Neither covers the source price panel, which is not
included.
