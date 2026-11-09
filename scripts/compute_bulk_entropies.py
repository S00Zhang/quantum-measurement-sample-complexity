"""Exact block entropies of the infinite chain at g = 1 and 1.5 (Appendix B 1).

Validates the Pfaffian / Walsh-Hadamard route against exact diagonalization, then writes
data/bulk_block_entropies.json.  The paper uses --lmax 20 (the cost doubles with every added site).

Usage: python scripts/compute_bulk_entropies.py [--lmax 20] [--skip-validate]
"""
from qmsc.bulk_entropy import main


if __name__ == "__main__":
    main()
