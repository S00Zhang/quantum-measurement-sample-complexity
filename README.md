# Sample Complexity for Generative Learning of Quantum Measurement Data

Code and data for

> Shuhan Zhang and Yi-Zhuang You, *Sample Complexity for Generative Learning of Quantum
> Measurement Data*, [arXiv:2610.38028](https://arxiv.org/abs/2610.38028) (2026).

The paper develops a bias–variance framework for learning a quantum measurement distribution
from a finite number of measurement records and tests it on computational-basis Born
distributions of transverse-field Ising ground states, learned by local autoregressive models
with a smoothed histogram learner. This repository contains the Python package `qmsc` with the
exact free-fermion and exact-diagonalization calculations, the learner, the exact finite-sample
estimation cost of Proposition 1 and the two learning campaigns, the result files behind every
figure, table and quoted number, the scripts that regenerate the figures from the result files,
and an audit script that recomputes every number quoted in the paper.

## Installation

Python 3.10 or later with `numpy`, `scipy` and `matplotlib`. From the repository root

```
pip install -e .
```

Regenerating the figures additionally needs a LaTeX installation with Computer Modern (the
figures are typeset with `text.usetex`) and `poppler-utils` (`pdfinfo`, `pdffonts`, `pdftoppm`)
for the PNG previews and the figure checks.

## Layout

```
qmsc/                      the package
  tfim_chain.py            exact diagonalization of the open chain, Born distribution, representation bias (Lemma 1)
  bulk_entropy.py          exact block entropies of the infinite chain (Pfaffian / Walsh-Hadamard route)
  critical_tail.py         Monte Carlo test of the critical tail b(R) ~ kappa/R to R = 96
  bulk_bias.py             per-site bias b_g(R) of the infinite chain for g = 0.5, 1, 1.5
  open_chain.py            free-fermion probabilities, cells and exact sampler of the open 50-site chain
  histogram_learner.py     the smoothed histogram learner with exact population errors
  estimation_cost.py       the exact finite-sample estimation cost of Proposition 1
  bulk_campaign.py         learning campaign at a bulk site of the infinite chain
  finite_chain_campaign.py learning campaign on the open 50-site chain
  template_fits.py         fits of the Table I templates to the bulk optima
  plotting/                figure modules (bulk.py, finite_chain.py), the figure style and the figure checks
scripts/                   command-line entry points (see below)
data/                      result files (JSON) that the figures, tables and quoted numbers are drawn from
figures/                   written by scripts/make_figures.py (not tracked)
reproduce.sh               regenerates the figures and runs the audit
```

`data/` and `figures/` are located relative to the repository root, which is the parent of the
package for an editable install, otherwise the current working directory or the directory named
by the environment variable `QMSC_ROOT`.

## Reproducing the figures and the quoted numbers

```
./reproduce.sh
```

runs `scripts/make_figures.py`, which writes the seven figures of the paper to `figures/`
(`fig2_bulk_bias.pdf` to `fig8_finite_chain_scaling.pdf`, Fig. 1 is a schematic drawn in the
manuscript) together with PNG previews, and then `scripts/audit_numbers.py`, which recomputes
every number quoted in the paper from `data/` and ends with `AUDIT RESULT: ALL PASS`.
Every figure is drawn at the width it is included at, and `qmsc.plotting.figcheck` checks
font sizes, legend placement, legend–data overlap and panel letters on every save.

| Paper | Figure function (`qmsc.plotting`) | Result files |
|---|---|---|
| Fig. 2, per-site bias of the infinite chain | `bulk.fig_bulk_bias` | `bulk_bias.json`, `bulk_site_campaign.json` |
| Fig. 3, estimation cost and context occupancy at a bulk site | `bulk.fig_estimation_cost` | `bulk_site_campaign.json` |
| Fig. 4, bias–variance competition at a bulk site | `bulk.fig_capacity_selection` | `bulk_site_campaign.json` |
| Fig. 5, optimal error and radius at a bulk site | `bulk.fig_scaling` | `bulk_site_campaign.json`, `bulk_bias.json` |
| Table II, leading-order exponents | `scripts/audit_numbers.py` | `bulk_campaign_statistics.json`, `template_fits.json` |
| Fig. 6, critical-tail test | `bulk.fig_critical_tail` | `bulk_bias.json`, `critical_tail_mc.json` |
| Fig. 7, finite-chain validation and selection | `finite_chain.fig_finite_chain_validation` | `finite_chain_campaign_n50_g{0.5,1,1.5}.json` |
| Fig. 8, finite-chain sample-size dependence | `finite_chain.fig_finite_chain_scaling` | `finite_chain_campaign_n50_g{0.5,1,1.5}.json` |
| App. B 3, finite-chain bias and boundary effects | `scripts/audit_numbers.py` | `finite_chain_bias_n20_g*.json`, `critical_tail_n50.json` |
| App. D, ln ln N corrections and the template fits | `scripts/audit_numbers.py` | `lnlnN_coefficients_check.json`, `template_fits.json` |

## Regenerating the result files

The scripts below produce the files in `data/` from scratch, in the order of the paper. The
deterministic steps take seconds to minutes. The two learning campaigns and the Monte Carlo
tail test are the expensive steps and are seeded, so a rerun with the stated parameters
reproduces the stored files.

| Script | Writes | Reads |
|---|---|---|
| `compute_bulk_entropies.py --lmax 20` | `bulk_block_entropies.json` | — |
| `run_critical_tail_test.py` | `critical_tail_mc.json` | `bulk_block_entropies.json` |
| `compute_bulk_bias.py` | `bulk_bias.json` | `bulk_block_entropies.json`, `critical_tail_mc.json` |
| `run_bulk_campaign.py --rmax 17 --nmax 100000000` | `bulk_site_campaign.json` | `bulk_bias.json`, `bulk_block_entropies.json` |
| `fit_templates.py` | `template_fits.json` | `bulk_site_campaign.json` |
| `run_finite_chain_campaign.py --g 0.5` (and `1`, `1.5`) | `finite_chain_campaign_n50_g*.json` | — |
| `compute_statistics.py` | `bulk_campaign_statistics.json`, `finite_chain_n50_statistics.json` | the two campaigns |
| `check_finite_chain_critical_tail.py` | `critical_tail_n50.json` | `finite_chain_campaign_n50_g1.json`, `bulk_bias.json` |
| `compute_finite_chain_bias.py --n 20 --g 0.25 0.5 0.75 0.9 1 1.1 1.25 1.5 2 --no-direct-above 16` | `finite_chain_bias_n20_g*.json` | — |
| `check_cell_law.py` | `cell_law_check.json` | — |
| `check_lnlnN_coefficients.py` | `lnlnN_coefficients_check.json` | — |

Every module that produces data carries its own validation against exact diagonalization or
brute force, run before any number is used: `python -m qmsc.open_chain` runs the free-fermion
ladder at n = 6 to 16, `python -m qmsc.estimation_cost` cross-checks the exact estimation cost
against a Monte Carlo of the learner at n = 12, `compute_bulk_entropies.py` checks the Pfaffian
route against exact diagonalization, `run_critical_tail_test.py` calibrates the conditioning
engine on exact diagonalization, and `run_finite_chain_campaign.py --validate16` runs the whole
50-site pipeline at n = 16 against exact diagonalization.

## Result files

All information quantities are in nats. Sample sizes N, context radii R and couplings g are
stored with the data they index.

**Infinite chain**

- `bulk_block_entropies.json` — exact block entropies H_ℓ of consecutive bulk spins for
  g = 1 and 1.5 (ℓ ≤ 20), the one-step conditional entropies h_R = H_{R+1} − H_R, the
  compensated doubling differences 2R (h_R − h_{2R}) and the reference value κ = 0.0589 nats
  of Lau and Grassberger.
- `critical_tail_mc.json` — the Monte Carlo part of the critical-tail test (App. B 2):
  compensated doubling differences 2R D_R at g = 1 from averages of exact conditional
  divergences with their standard errors (R = 4 to 96, two measurement bases), the gapped
  control at g = 1.5 and the calibration of the conditioning engine at R = 4.
- `bulk_bias.json` — the per-site bias b_g(R) = h_R − h_∞ of Sec. IV for g = 0.5, 1, 1.5:
  the block entropies at g = 0.5 (ℓ ≤ 22), the gapped biases with their trusted range, and the
  critical bias reconstructed from the exact (2R ≤ 19) and Monte Carlo doubling differences
  with its tail fractions.
- `bulk_site_campaign.json` — the bulk-site learning campaign of Secs. V and VI: for each
  coupling, the bias b(R) for R ≤ 17, the measured estimation excess `V_hat` of the histogram
  learner for 20 training realizations × 13 sample sizes (10² to 10⁸) × 18 radii, the number
  of visited contexts, the exact estimation cost `V_exact` of Proposition 1 on the same grid
  and on a dense sample-size grid, the expected occupancy `d_occ`, and the measured, exactly
  predicted and parameter-count optimal radii and errors.
- `bulk_campaign_statistics.json` — the validation statistics of the bulk campaign (702 grid
  points, fractions within two and three standard errors), the optimal radii, and the
  least-squares slopes and exponents with bootstrap errors quoted in Table II.
- `template_fits.json` — the full Table I templates fitted to the 13 bulk optima: row III at
  g = 1 (μ, κ, β with bootstrap errors and the sensitivity to the entropy-rate uncertainty) and
  row I at g = 0.5 and 1.5 (App. D 2).
- `lnlnN_coefficients_check.json` — the numerical check of the ln ln N coefficients of rows II
  and III of Table I and of the correction factors of the optimal error (App. D 1).

**50-site chain**

- `finite_chain_campaign_n50_g0.5.json`, `..._g1.json`, `..._g1.5.json` — the finite-chain
  campaign of App. C 2: the window conditional entropies, the representation bias B(R) from
  13-spin windows with the Monte Carlo tail beyond W = 12 and its standard error, the measured
  estimation excess for 20 training realizations × 9 sample sizes (10² to 10⁶) × 13 radii, the
  visited cells, the exact estimation cost, and the optima.
- `finite_chain_n50_statistics.json` — the numbers quoted from that campaign (optimum counts,
  radius slopes, error exponents, bias fits over the visited radii, normalized cost).
- `critical_tail_n50.json` — the compensated doubling difference 2R D_R at the centre of the
  50-site chain against the infinite-chain value (App. B 3).

**Exact diagonalization**

- `finite_chain_bias_n20_g*.json` — the exact bias B(R) by both routes and the site-resolved
  conditional mutual information on a 20-site chain for nine couplings from g = 0.25 to 2
  (App. B 3).
- `cell_law_check.json` — exact binomial sums for the per-cell error of the add-1/2 estimate
  (Lemma 2).

## License

The code is released under the MIT License (`LICENSE`). The result files in `data/` are
released under the Creative Commons Attribution 4.0 International License (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite the paper when using this code or these data.
