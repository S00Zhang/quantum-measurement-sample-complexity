# Sample Complexity for Generative Learning of Quantum Measurement Data

Code and data for

> Shuhan Zhang and Yi-Zhuang You, *Sample Complexity for Generative Learning of Quantum
> Measurement Data*, [arXiv:2610.38028](https://arxiv.org/abs/2610.38028) (2026).

The paper develops a bias–variance framework for learning a quantum measurement distribution
from a finite number of measurement records and tests it on computational-basis Born
distributions of transverse-field Ising ground states, learned by local autoregressive models
with a smoothed histogram learner. This repository contains the Python package `qmsc` (the
exact calculations, the learner, the exact finite-sample estimation cost and the two learning
campaigns), the result files behind every figure, table and quoted number, the script that
regenerates the figures from them, and an audit script that recomputes every number quoted in
the paper.

## Installation

Python 3.10 or later with `numpy`, `scipy` and `matplotlib`. From the repository root

```
pip install -e .
```

## Reproducing the figures and the quoted numbers

```
./reproduce.sh
```

regenerates Figs. 2–8 from the included result files and checks the numerical values reported
in the paper. The command writes the figures to `figures/` and does not rerun the simulations.
It needs a LaTeX installation with Computer Modern (the figures are typeset with `text.usetex`)
and `poppler-utils` (`pdfinfo`, `pdffonts`, `pdftoppm`). The audit alone,
`python scripts/audit_numbers.py`, needs neither and ends with `AUDIT RESULT: ALL PASS`.

| Paper | Figure function (`qmsc.plotting`) | Result files in `data/` |
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

Fig. 1 is a schematic drawn in the manuscript. All information quantities in the result files
are in nats, and sample sizes, context radii and couplings are stored with the data they index.

## Layout

```
qmsc/            the package: exact diagonalization and free-fermion calculations, the learner,
                 the exact estimation cost, the two learning campaigns, the figure modules
scripts/         command-line entry points: make_figures.py, audit_numbers.py, one driver per result file
data/            result files (JSON) of the simulations and the exact calculations
figures/         written by scripts/make_figures.py (not tracked)
reproduce.sh     the command above
REPRODUCING.md   how to rerun the simulations and what each result file contains
```

To rerun or adapt the calculations themselves, see [REPRODUCING.md](REPRODUCING.md): it lists
the module-by-module contents of the package, the script and parameters that produced each
result file, the validation checks against exact diagonalization, and the contents of every
file in `data/`.

## License

The code is released under the MIT License (`LICENSE`). The result files in `data/` are
released under the Creative Commons Attribution 4.0 International License (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite the paper when using this code or these data.
