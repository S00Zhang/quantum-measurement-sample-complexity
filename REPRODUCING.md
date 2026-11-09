# Reproducing and extending the calculations

`./reproduce.sh` (see the README) regenerates the figures and checks the quoted numbers from the
result files in `data/`. This page is for rerunning the calculations that produced those files,
or adapting them.

## Package contents

```
qmsc/
  tfim_chain.py             exact diagonalization of the open chain, Born distribution, representation bias (Lemma 1)
  bulk_entropy.py           exact block entropies of the infinite chain (Pfaffian / Walsh-Hadamard route)
  critical_tail.py          Monte Carlo test of the critical tail b(R) ~ kappa/R to R = 96
  bulk_bias.py              per-site bias b_g(R) of the infinite chain for g = 0.5, 1, 1.5
  open_chain.py             free-fermion probabilities, cells and exact sampler of the open 50-site chain
  histogram_learner.py      the smoothed histogram learner with exact population errors
  estimation_cost.py        the exact finite-sample estimation cost of Proposition 1
  bulk_campaign.py          learning campaign at a bulk site of the infinite chain
  finite_chain_campaign.py  learning campaign on the open 50-site chain
  template_fits.py          fits of the Table I templates to the bulk optima
  paths.py                  locations of data/ and figures/
  plotting/
    bulk.py                 Figs. 2 to 6 and the bulk-campaign statistics
    finite_chain.py         Figs. 7 and 8 and the finite-chain statistics
    style.py                the publication style shared by the figure modules
    figcheck.py             automated checks of the figures (fonts, legends, panel letters, inclusion width)
```

Every module starts with a docstring that states the quantity it computes, the method, the
conventions (bit order, units) and the section of the paper it belongs to.

`data/` and `figures/` are located relative to the repository root, which is the parent of the
package for an editable install (`pip install -e .`), otherwise the current working directory
or the directory named by the environment variable `QMSC_ROOT`.

## Regenerating the result files

The scripts below produce the files in `data/` from scratch, in the order of the paper. The
deterministic steps take seconds to minutes. The two learning campaigns and the Monte Carlo
tail test are the expensive steps. They are seeded, so a rerun with the stated parameters
reproduces the stored files.

| Script (in `scripts/`) | Writes | Reads |
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

Each script's docstring (`python scripts/<name>.py --help` for the arguments) states what it
computes. The bulk campaign with the paper's parameters draws 10⁸ records per seed and needs
about 1 GB of memory. The 50-site campaign writes a checkpoint after every seed and resumes
from it. Reduced runs (`--seeds 2 --nmax 10000`, with `--tail 100000` for the 50-site campaign, or
`--quick` for the tail test) exercise the whole pipeline in minutes.

## Validation

Every module that produces data carries its own validation against exact diagonalization or
brute force, run before any number is used:

- `python -m qmsc.open_chain` runs the free-fermion ladder of the open chain at n = 6 to 16
  (covariance matrix, cells, bias, sampler).
- `python -m qmsc.estimation_cost` cross-checks the exact estimation cost against a Monte Carlo
  of the learner at n = 12.
- `scripts/compute_bulk_entropies.py` checks the Pfaffian route against exact diagonalization
  before computing the block entropies.
- `scripts/run_critical_tail_test.py` calibrates the conditioning engine on exact
  diagonalization and checks its R = 4 estimate against the exact value.
- `scripts/run_finite_chain_campaign.py --validate16` runs the whole 50-site pipeline at n = 16
  against exact diagonalization (output in `validation/`).
- `scripts/compute_finite_chain_bias.py` computes the bias by two independent routes and records
  their agreement (10⁻¹⁰ or better in the stored files).

`python -m qmsc.plotting.figcheck` runs the PDF-level checks on the written figures.

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
