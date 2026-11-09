#!/usr/bin/env bash
# Regenerates the seven figures of the paper from the result files in data/ and recomputes every
# quoted number.  Requires the package (pip install -e .), a LaTeX installation with Computer
# Modern (the figures are typeset with text.usetex) and poppler-utils (pdfinfo, pdffonts, pdftoppm).
set -euo pipefail
cd "$(dirname "$0")"
echo "== figures -> figures/"
python scripts/make_figures.py
echo "== PDF-level figure checks"
python -m qmsc.plotting.figcheck
echo "== number audit"
python scripts/audit_numbers.py | tail -1
