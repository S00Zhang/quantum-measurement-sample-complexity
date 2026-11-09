"""Statistics quoted in the text and captions, derived from the campaign result files.

Writes data/bulk_campaign_statistics.json (validation z-scores, optima, slopes and exponents of the bulk-site
campaign) and data/finite_chain_n50_statistics.json (the same for the 50-site chain).

Usage: python scripts/compute_statistics.py
"""
from qmsc.plotting import bulk, finite_chain


def main():
    D = bulk.load()
    bulk.campaign_statistics(D)
    F, fits = finite_chain._load()
    finite_chain.campaign_statistics(F, fits)


if __name__ == "__main__":
    main()
