"""Learning campaign at a bulk site of the infinite chain (Secs. V and VI, Figs. 3 to 5).

Reads data/bulk_bias.json and data/bulk_block_entropies.json and writes data/bulk_site_campaign.json.  The
paper's run is --rmax 17 --nmax 100000000 --seeds 20 (the 1e8 draws of one seed take about 1 GB of memory);
the defaults give a reduced campaign to N = 1e6.

Usage: python scripts/run_bulk_campaign.py [--rmax 17] [--nmax 100000000] [--seeds 20] [--gs 0.5,1,1.5]
"""
from qmsc.bulk_campaign import main


if __name__ == "__main__":
    main()
