"""Fits of the Table I templates to the bulk-site campaign (Sec. VI B, Table II, Appendix D 2).

Reads data/bulk_site_campaign.json and writes data/template_fits.json.

Usage: python scripts/fit_templates.py
"""
from qmsc.template_fits import main


if __name__ == "__main__":
    main()
