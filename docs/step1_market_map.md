# The market map: method, results and limits

This document explains what the code does, what it found on one year of S&P 500 data, and how far
those findings can be trusted. The numbers come from the files in [`output/step1/`](../output/step1);
the auto-generated version with every table is [`step1_report.md`](../output/step1/step1_report.md).

## The question

If we ignore sector labels and only look at how stock prices move day by day, which stocks end up
together, and does that match the sectors people use to describe the market?

## Method

1. **Data.** Dividend- and split-adjusted daily closes from Yahoo Finance for the current S&P 500
   members, one year (`2025-10-16` to `2026-10-16`, end exclusive). Sector labels (GICS) come from
   Wikipedia's list of S&P 500 companies.
2. **Returns.** Daily log return of each stock, `r(t) = ln(P(t) / P(t-1))`. Nothing is filled in.
   Only stocks with a return on **every** day are used (498 of 503). The other five are listed with
   the reason in [`excluded_stocks.csv`](../output/step1/excluded_stocks.csv); they are recent
   listings, spin-offs or tickers that changed.
3. **Correlation.** Pearson correlation of the daily returns of every pair of stocks. The matrix
   is written `C`, and `C[i,j]` is the correlation of stock `i` with stock `j`. (Some texts write
   it `ρ(i,j)`; it is the same number.)
4. **Distance.** `d[i,j] = sqrt(2 (1 - C[i,j]))`. It is 0 for perfectly correlated stocks, about
   1.41 for uncorrelated ones and 2 for perfectly opposite ones. For standardized return series
   this is the straight-line (Euclidean) distance between the two series up to a constant factor,
   so it is a true distance and clustering methods that need one are valid. This is checked on
   every run (the correlation matrix must be positive semi-definite), and Ward linkage is skipped
   if the check fails. The unit tests confirm the identity.
5. **Hierarchical clustering.** Agglomerative: start with every stock alone and repeatedly merge
   the two closest groups. How "closest" is measured for groups is the *linkage*. Four were
   compared, none chosen silently:
   - **single**: closest pair of members. Tends to chain stocks into one giant cluster.
   - **complete**: farthest pair of members. Compact, balanced groups.
   - **average**: mean over all member pairs.
   - **Ward**: merge that increases within-group variance the least (needs Euclidean distance).
6. **Choosing the linkage and the number of clusters `k`.** A rule fixed *before* looking at the
   results: pick the linkage and `k` (between 5 and 30) with the highest **mean silhouette**. The
   silhouette of a stock compares its average distance to its own cluster with its distance to
   the nearest other cluster (range -1 to 1; above about 0.5 is clear structure, near 0 is no
   structure). It uses returns only. **Sector labels are never used to build or choose clusters**;
   they are used afterwards to describe and compare.
7. **Sensitivity.**
   - Silhouette and agreement with sectors for every `k` from 2 to 40, for every linkage.
   - Agreement between linkages (adjusted Rand index, ARI: 1 = identical groupings, about 0 =
     unrelated).
   - Bootstrap: resample the trading days with replacement 100 times (seed 0), rebuild the tree,
     and compare with the full-sample clusters by ARI.

## Results (one year, 498 stocks, 250 trading days)

### There is no natural number of clusters

| linkage | tree fidelity (cophenetic corr.) | best k | silhouette | largest cluster | ARI vs sectors |
|---|---|---|---|---|---|
| single | 0.42 | 5 | -0.02 | 99% of stocks | 0.00 |
| complete | 0.57 | 30 | 0.09 | 9% | 0.28 |
| average | 0.71 | 23 | 0.08 | 30% | 0.25 |
| ward | 0.55 | 30 | 0.08 | 7% | 0.26 |

- **Every silhouette is below 0.1.** By the usual rules of thumb that is "no substantial
  structure": stocks are not sorted into clearly separated groups; the market is a continuum with
  denser neighbourhoods.
- The silhouette of complete linkage keeps rising with `k`, even beyond the allowed range (see the
  figure, which goes to `k = 40`), so the rule picks the boundary (`k = 30`). That is a convenient
  cut, **not** a discovered number of groups. The
  [sensitivity figure](../output/step1/k_sensitivity.png) shows the whole curve.
- **Single linkage is unusable here** (one cluster holds 99% of the stocks): chaining, as expected.
- Complete and Ward agree moderately at `k = 30` (ARI 0.56); average linkage produces a very
  uneven cut (a 30% cluster and several single-stock clusters).
- **Stability:** across 100 resamples of the days, the rebuilt 30-cluster grouping agrees with the
  full-sample one with a median ARI of 0.50 (5th to 95th percentile 0.44 to 0.56). So roughly half
  of the grouping is robust to which days are in the sample and the rest is not. A resample
  contains only about 63% of the distinct days, so this check is on the harsh side; and it
  assumes days are independent, which is only approximately true.

### What the clusters look like

![Correlation matrix ordered by cluster](../output/step1/correlation_heatmap.png)

The picture is the correlation matrix with stocks ordered by the tree. Dark red blocks on the
diagonal are groups that move together; black lines separate the 30 clusters; the coloured strips
show each stock's sector. Several blocks are clearly visible, and several large areas are only
faintly structured.

- **Some clusters are almost pure sectors or industries.** Banks, asset managers and brokers (36
  stocks, all Financials), insurers (21, all Financials), electric and multi-utilities (26 of 29
  are Utilities, the rest tower REITs), oil and gas producers (11 of 17 are Energy), a second
  all-Energy group of 10 (storage and transportation, equipment and services, exploration) and
  telecoms and cable (5, all Communication Services). Health Care splits into three clusters of
  12 each: life-science tools, biotech and pharma, and medical devices.
- **Others are mixed.** The two largest clusters (45 and 43 stocks) contain stocks from five and
  seven sectors, and Information Technology is spread over at least four clusters (22, 19, 15 and
    8 of its stocks in its four biggest).
- Overall agreement with GICS sectors is **moderate, not strong**: ARI about 0.26 to 0.28 at
  `k = 30`. Across all `k` it peaks at 0.33 (Ward, `k = 13`), 0.31 (complete, `k = 15`) and 0.29
  (average, `k = 37`); see the right panel of the sensitivity figure. Sectors are a partial
  description of co-movement in this window, not a full one.

### Sectors and correlation

| stock pairs | mean correlation |
|---|---|
| all pairs | 0.118 |
| same sector | 0.252 |
| different sector | 0.101 |
| same cluster | 0.437 |
| different cluster | 0.102 |

Two stocks in the same sector are on average much more correlated than two in different sectors,
but the average level is low overall. ("Same cluster" being the highest is by construction, because
the clusters were built to put correlated stocks together; it is not an independent finding.)
Within-sector correlation is highest for Utilities (0.51) and Energy (0.46) and lowest for
Information Technology (0.18) and Communication Services (0.16); see the
[sector correlation figure](../output/step1/sector_correlation.png). Energy is the only sector
that is about uncorrelated or slightly negatively correlated with most others.

### Strongest and weakest pairs

- The strongest pairs are share classes of one company (GOOGL/GOOG 0.996, FOXA/FOX 0.985,
  NWSA/NWS 0.961), then close industry peers: NSC/UNP (railroads) 0.94, CPT/MAA (apartment REITs)
  0.92, DHI/PHM (homebuilders) 0.91, AMAT/LRCX (chip equipment) 0.91. All are in the same sector
  and the same cluster, which is a useful sanity check on the method.
- The most negative correlation is -0.58 (ADP and TER). With 123,753 pairs, a few extreme values
  are expected from noise alone, so individual negative pairs should not be over-interpreted.

## Limits

- **One year only.** 250 observations are few for 498 stocks. There are more stocks than days, so
  the correlation matrix is mathematically singular (valid for clustering, but each correlation
  is noisy). Results describe this window and may differ in another.
- **Survivorship bias.** The universe is today's index membership applied to the past year.
- **Stocks without a full year of history are left out** (5 of 503).
- **Correlation is not causation.** Clusters say which stocks moved together, not why.
- **Choices matter.** The number of clusters, the linkage, the distance definition and the window
  all change the result; the silhouette picked the edge of the allowed range, so the 30-cluster
  picture is one reasonable view, not the truth.
- **Market-wide moves** share part of every stock's return. This project makes no attempt to
  remove them; clusters are based on raw correlations.
- **Not investment advice.**

## Reproducing it

```powershell
market-structure step1 --start 2025-10-16 --end 2026-10-16
```

Yahoo Finance may revise adjusted prices later, so a re-run can differ slightly in the last
decimals. The bootstrap uses a fixed seed (`--seed`). The run writes the exact settings to
[`run_summary.json`](../output/step1/run_summary.json).

## What each part of the code does

| file | role |
|---|---|
| [`data.py`](../src/market_structure/data.py) | S&P 500 list with sectors; Yahoo prices; saved copies on disk |
| [`returns.py`](../src/market_structure/returns.py) | log returns; which stocks have a complete history |
| [`clustering.py`](../src/market_structure/clustering.py) | correlation, distance, Euclidean check, linkages, silhouette, ARI, bootstrap |
| [`summaries.py`](../src/market_structure/summaries.py) | membership, cluster summaries, sector tables, pairs |
| [`plotting.py`](../src/market_structure/plotting.py) | the five figures |
| [`step1.py`](../src/market_structure/step1.py) | runs everything end to end |
| [`report.py`](../src/market_structure/report.py) | writes tables, figures and the generated report |
| [`cli.py`](../src/market_structure/cli.py) | the `market-structure` command |
