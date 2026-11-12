# pumpfun-top5-forecast-01 — Monte Carlo rank forecaster

Forecasts the 5 largest pump.fun coins by market cap at each of the house's
seven answer reads, from the 10-minute market-cap snapshots in `/input`.

## Method

Pump.fun's newest coins churn violently: in the practice history the top 5
changes almost completely every 10 minutes, 1-4 of the next top 5 are coins
that did not exist in the previous mark, and at the last practice read 3 of
the top 5 were coins the history never named. Pure persistence (the house
baseline) is therefore far too confident at a 7-13 hour horizon.

This entry instead:

- builds every coin's log10-cap trajectory from the history marks;
- estimates each rank band's per-10-minute drift and spread from the
  history's own consecutive-mark transitions (shrunk toward priors);
- simulates each candidate coin's log-cap at every answer read as a
  two-state mixture: with a per-step survival probability that grows with
  the coin's size and age it holds near its peak (rising slowly), otherwise
  it fades to the ~28 SOL floor every dumped coin ends at;
- adds a newborn process: coins born after the history cut, drawn at the
  rate and sizes at which new coins actually entered the top 5;
- ranks the simulated caps, and averages the place probabilities and the
  per-place size histograms over the seven reads (the one forecast is scored
  at every read, so the mixture is the honest single forecast).

Node 22, standard library only, deterministic (fixed-seed RNG, no clock, no
network, no `Math.random`).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Local validation (practice history, held-out marks as pseudo-answer reads)

| split | this entry | house baseline |
|---|---:|---:|
| cut 20:30, reads 20:40+20:50 | -29.28 | -33.48 |
| cut 20:40, read 20:50 | -23.77 | -29.87 |
| last practice read, order only | -21.83 | -23.29 |

Log score (higher is better, 0 best). The entry beats the baseline on every
split, including the hardest 70-minute-out read where 3 of the top 5 were
coins the history never named.

## License

MIT. See LICENSE.
