"""Publication styling shared by the figure modules.

The figures are typeset with LaTeX (``text.usetex``) in Computer Modern, the font of the REVTeX manuscript,
with 9 pt axis labels and panel letters and 8 pt tick labels and legends.  Every figure is drawn at the width
it is included at (``\\textwidth`` = 510 pt = 7.06 in or ``\\columnwidth`` = 246 pt = 3.40 in), so the fonts
are not rescaled by ``\\includegraphics``.  ``save`` audits the figure with :mod:`qmsc.plotting.figcheck` and
writes a tight-cropped PDF to the ``figures/`` directory of the repository.
"""
import os
import re
import subprocess

import matplotlib

from qmsc.paths import FIGURES_DIR

TEXTWIDTH = 510.0 / 72.27  # revtex4-2 two-column \textwidth in inches
COLUMNWIDTH = 246.0 / 72.27
LABEL_PT, TICK_PT, LEGEND_PT = 9.0, 8.0, 8.0
MARKER_SCALE = 0.6  # marker sizes relative to the screen drafts the figures were designed on


def apply():
    """Set the matplotlib defaults.  Call once at import time, before any figure is created."""
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
            "font.family": "serif",
            "font.size": LABEL_PT,
            "axes.labelsize": LABEL_PT,
            "axes.titlesize": LABEL_PT,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "legend.fontsize": LEGEND_PT,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "savefig.dpi": 300,
        }
    )


def num(x):
    """A sample size as a LaTeX power of ten (1e6 -> 10^{6}, 3e4 -> 3\\times10^{4})."""
    import math

    x = float(x)
    if x <= 0:
        return f"{x:g}"
    k = int(math.floor(math.log10(x) + 1e-9))
    m = x / 10**k
    if abs(m - 1) < 1e-6:
        return rf"10^{{{k}}}"
    return rf"{m:g}\times10^{{{k}}}"


def ms(size):
    """Marker size in points, from the draft size."""
    return max(1.8, round(size * MARKER_SCALE, 1))


def mew(width):
    """Marker edge width, scaled with the marker."""
    return round(width * 0.75, 2)


def figsize(w, h, width=TEXTWIDTH, min_h=None, paper_h=None):
    """Figure size at the inclusion width: height ``paper_h`` if given, else the draft aspect ratio (not below
    ``min_h``).  ``w`` and ``h`` are the draft dimensions the layouts were designed at."""
    if paper_h is not None:
        return (width, paper_h)
    hh = h * width / w
    if min_h is not None:
        hh = max(hh, min_h)
    return (width, hh)


_LETTER = re.compile(r"^\s*\(([a-z])\)")


def set_title(ax, title, size=None, color="#0b0b0b"):
    """Panel label above the axes: ``title`` is a string, of which only a leading panel letter "(a)" is kept, or
    a (draft, label) pair whose second element is used as given."""
    if title is None:
        return
    draft, label = (title, None) if isinstance(title, str) else title
    if label is None:
        m = _LETTER.match(draft)
        label = f"({m.group(1)})" if m else ""
    if label:
        ax.set_title(label, loc="left", fontsize=LABEL_PT, color=color, pad=4)


def save(fig, name):
    """Audit the figure, then write ``figures/<name>.pdf`` (tight crop) and a PNG preview next to it.
    ``FIGCHECK=0`` skips the audit."""
    if os.environ.get("FIGCHECK", "1") != "0":
        from qmsc.plotting import figcheck

        figcheck.audit(fig, name)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf = os.path.join(FIGURES_DIR, name + ".pdf")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    try:
        subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", pdf, os.path.join(FIGURES_DIR, name)], check=True)
    except Exception as e:  # the preview is a convenience only
        print(f"  [figures] preview not written ({e})")
    print(f"  [figures] wrote figures/{name}.pdf")
