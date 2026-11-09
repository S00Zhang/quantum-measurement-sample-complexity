"""Per-site representation bias of the infinite chain for g = 0.5, 1, 1.5 (Sec. IV, Fig. 2).

Reads data/bulk_block_entropies.json and data/critical_tail_mc.json, computes the g = 0.5 block entropies
(reused from an existing data/bulk_bias.json unless BULK_RECOMPUTE=1) and writes data/bulk_bias.json.

Usage: python scripts/compute_bulk_bias.py
"""
from qmsc.bulk_bias import main


if __name__ == "__main__":
    main()
