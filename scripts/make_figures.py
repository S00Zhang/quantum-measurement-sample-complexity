"""Regenerate the seven figures of the paper from the result files in data/ into figures/.

Usage (repository root, after ``pip install -e .``):  python scripts/make_figures.py
Requires a LaTeX installation with Computer Modern (the figures are typeset with text.usetex) and
poppler-utils for the PNG previews.
"""
from qmsc.plotting import bulk, finite_chain


def main():
    D = bulk.load()
    bulk.fig_bulk_bias()                        # Fig. 2
    bulk.fig_estimation_cost(D)                 # Fig. 3
    bulk.fig_capacity_selection(D, N_show=1e6)  # Fig. 4
    bulk.fig_scaling()                          # Fig. 5
    bulk.fig_critical_tail()                    # Fig. 6
    F, fits = finite_chain._load()
    finite_chain.fig_finite_chain_validation(F)        # Fig. 7
    finite_chain.fig_finite_chain_scaling(F, fits)     # Fig. 8
    print("done")


if __name__ == "__main__":
    main()
