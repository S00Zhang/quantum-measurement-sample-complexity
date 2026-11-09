"""Learning campaign on the open 50-site chain (Appendix C 2, Figs. 7 and 8).

Writes data/finite_chain_campaign_n50_g{g}.json, one coupling at a time, with a checkpoint after every seed.
--validate16 runs the same pipeline at n = 16 against exact diagonalization (output in validation/).

Usage: python scripts/run_finite_chain_campaign.py --g 0.5 [--seeds 20] [--nmax 1000000]
       python scripts/run_finite_chain_campaign.py --validate16 [--nmax 100000]
"""
from qmsc.finite_chain_campaign import main


if __name__ == "__main__":
    main()
