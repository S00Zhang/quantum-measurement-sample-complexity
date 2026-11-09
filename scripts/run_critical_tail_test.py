"""Monte Carlo test of the critical tail b(R) ~ kappa/R to R = 96 (Appendix B 2, Fig. 6).

Reads data/bulk_block_entropies.json for the exact small-R cross-check and writes data/critical_tail_mc.json.
--quick runs a reduced plan for a smoke test.

Usage: python scripts/run_critical_tail_test.py [--quick]
"""
from qmsc.critical_tail import main


if __name__ == "__main__":
    main()
