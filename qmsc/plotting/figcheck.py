"""
Automated checks of the publication figures.

Two entry points.

1. audit(fig, name)  — called by :func:`qmsc.plotting.style.save` before the PDF is written.
   Inspects the live matplotlib Figure and prints one line per finding:
     * LEGEND/DATA     a legend box covers plotted points or line segments of its own axes
     * TEXT/DATA       an in-axes annotation covers plotted points or segments
     * LEGEND/LEGEND   two panels carry legends with the same label set (redundant; key colours once)
     * FONT            a text element whose size differs from the style constants (9 pt labels/titles,
                       8 pt ticks/legends)
     * AXIS LABEL      a panel without an x or y label (shared labels must be stated in the caption)
     * PANEL LETTER    a multi-panel figure whose panel has no "(a)"-style title or text
     * FIG WIDTH       the figure is not drawn at a REVTeX inclusion width (\\textwidth or \\columnwidth)
   Set FIGCHECK=0 to skip the audit.

2. python -m qmsc.plotting.figcheck [figures/*.pdf]  — PDF-level checks on the written files (poppler):
     * page width against the two inclusion widths (pdfinfo)
     * fonts embedded and Computer Modern (pdffonts): CM*/SFRM*/CMR* expected; DejaVu, Helvetica,
       Arial, Bitstream, Nimbus flagged (a non-CM font means text.usetex was off or a label bypassed
       the style)

The checks are geometric and conservative: a finding is a prompt to look at the rendered PNG, not a
verdict.  Legend/data overlap is tested in display coordinates after a draw, so log axes, twin axes
and insets are handled by the transforms themselves.
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np

TEXTWIDTH_IN = 510.0 / 72.27
COLUMNWIDTH_IN = 246.0 / 72.27
LABEL_PT, TICK_PT, LEGEND_PT = 9.0, 8.0, 8.0
PAD_PX = 2.0           # tolerance around a legend/text box, in display pixels
SUBSAMPLE = 8          # points per line segment when testing segment/box intersection


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------
def _inside(pts, bbox, pad=PAD_PX):
    """Boolean mask: which display-space points fall inside bbox (padded)."""
    if pts is None or len(pts) == 0:
        return np.zeros(0, bool)
    x, y = pts[:, 0], pts[:, 1]
    return (x >= bbox.x0 - pad) & (x <= bbox.x1 + pad) & (y >= bbox.y0 - pad) & (y <= bbox.y1 + pad)


def _line_display_points(ax, line):
    """Display-space points of a Line2D, with segments subsampled when a line is drawn."""
    xy = np.asarray(line.get_xydata(), float)
    if xy.size == 0:
        return np.zeros((0, 2))
    ok = np.isfinite(xy).all(axis=1)
    xy = xy[ok]
    if xy.shape[0] == 0:
        return np.zeros((0, 2))
    disp = line.get_transform().transform(xy)
    if line.get_linestyle() not in ("None", "none", "", " ") and line.get_linewidth() > 0 and len(disp) > 1:
        t = np.linspace(0, 1, SUBSAMPLE, endpoint=False)[1:]
        seg = disp[:-1, None, :] + (disp[1:, None, :] - disp[:-1, None, :]) * t[None, :, None]
        disp = np.vstack([disp, seg.reshape(-1, 2)])
    return disp


def _collection_display_points(coll):
    offs = np.asarray(coll.get_offsets(), float)
    if offs.size == 0:
        return np.zeros((0, 2))
    return coll.get_offset_transform().transform(offs)


def _data_points(ax):
    """Display-space points of the data artists; reference lines drawn in axes coordinates
    (axhline/axvline, blended transforms) are not data and are excluded."""
    pts = [_line_display_points(ax, ln) for ln in ax.lines if ln.get_visible() and ln.get_transform() == ax.transData]
    pts += [_collection_display_points(c) for c in ax.collections if c.get_visible() and hasattr(c, "get_offsets")]
    pts = [p for p in pts if len(p)]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _axes_label(fig, ax):
    try:
        k = list(fig.axes).index(ax)
    except ValueError:
        k = -1
    t = ax.get_title(loc="left") or ax.get_title()
    return f"axes[{k}]" + (f" {t!r}" if t else "")


# ----------------------------------------------------------------------------------------------
# live-figure audit
# ----------------------------------------------------------------------------------------------
def audit(fig, name="figure", quiet=False):
    """Run all live checks; return the list of findings (strings). Prints unless quiet."""
    findings = []
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception as e:                                    # pragma: no cover
        return [f"[figcheck] could not draw {name}: {e}"]

    axes = [ax for ax in fig.axes if ax.get_visible()]
    multi = len(axes) > 1
    legend_label_sets = []

    # FIG WIDTH
    w_in = fig.get_size_inches()[0]
    if not (abs(w_in - TEXTWIDTH_IN) / TEXTWIDTH_IN < 0.02 or abs(w_in - COLUMNWIDTH_IN) / COLUMNWIDTH_IN < 0.02):
        findings.append(f"FIG WIDTH      {name}: drawn at {w_in:.2f} in; REVTeX widths are {TEXTWIDTH_IN:.2f} (textwidth) "
                        f"or {COLUMNWIDTH_IN:.2f} (columnwidth) — fonts will be rescaled by includegraphics")

    for ax in axes:
        tag = _axes_label(fig, ax)
        data = _data_points(ax)

        # LEGEND/DATA
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            bbox = leg.get_window_extent(renderer)
            n_in = int(_inside(data, bbox).sum())
            if n_in:
                findings.append(f"LEGEND/DATA    {name} {tag}: legend box covers {n_in} plotted points/segments "
                                f"(loc={leg._loc!r}); move it, shrink it, or key the colours in another panel/caption")
            legend_label_sets.append((tag, tuple(t.get_text() for t in leg.get_texts())))

        # TEXT/DATA (annotations and free text inside the axes, excluding axis labels/titles)
        for txt in ax.texts:
            if not txt.get_visible() or not txt.get_text().strip():
                continue
            tb = txt.get_window_extent(renderer)
            n_in = int(_inside(data, tb, pad=0.5).sum())
            if n_in:
                findings.append(f"TEXT/DATA      {name} {tag}: text {txt.get_text()!r} covers {n_in} plotted points/segments")

        # FONT sizes
        for t, expect, what in ([(ax.xaxis.label, LABEL_PT, "x label"), (ax.yaxis.label, LABEL_PT, "y label")]
                                + [(ax.title, LABEL_PT, "title"), (ax._left_title, LABEL_PT, "left title")]):
            if t.get_text().strip() and abs(t.get_fontsize() - expect) > 0.26:
                findings.append(f"FONT           {name} {tag}: {what} {t.get_fontsize():.1f} pt (style: {expect:g} pt)")
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            if lab.get_text().strip() and abs(lab.get_fontsize() - TICK_PT) > 0.26:
                findings.append(f"FONT           {name} {tag}: tick label {lab.get_fontsize():.1f} pt (style: {TICK_PT:g} pt)")
                break
        if leg is not None:
            for t in leg.get_texts():
                if abs(t.get_fontsize() - LEGEND_PT) > 0.26:
                    findings.append(f"FONT           {name} {tag}: legend text {t.get_fontsize():.1f} pt (style: {LEGEND_PT:g} pt)")
                    break
        for txt in ax.texts:
            if txt.get_text().strip() and abs(txt.get_fontsize() - LEGEND_PT) > 0.26 and abs(txt.get_fontsize() - LABEL_PT) > 0.26:
                findings.append(f"FONT           {name} {tag}: annotation {txt.get_text()!r} at {txt.get_fontsize():.1f} pt "
                                f"(style: {LEGEND_PT:g} or {LABEL_PT:g} pt)")

        # AXIS LABEL
        if len(data) and not ax.get_xlabel().strip():
            findings.append(f"AXIS LABEL     {name} {tag}: no x label (if shared across panels, say so in the caption)")
        if len(data) and not ax.get_ylabel().strip():
            shared = any(o is not ax and o.get_ylabel().strip() and o.get_yscale() == ax.get_yscale()
                         and abs(o.get_position().y0 - ax.get_position().y0) < 0.02 for o in axes)
            if not shared:   # a labelled panel in the same row with the same scale counts as the shared label
                findings.append(f"AXIS LABEL     {name} {tag}: no y label (if shared across panels, say so in the caption)")

        # PANEL LETTER
        if multi and len(data):
            texts = [ax.get_title(loc="left"), ax.get_title()] + [t.get_text() for t in ax.texts]
            if not any(re.match(r"^\s*\(?\(?[a-z]\)", s or "") for s in texts):
                findings.append(f"PANEL LETTER   {name} {tag}: no '(a)'-style panel letter")

    # LEGEND/LEGEND
    seen = {}
    for tag, labels in legend_label_sets:
        if labels in seen:
            findings.append(f"LEGEND/LEGEND  {name}: {tag} repeats the legend of {seen[labels]} ({', '.join(labels)}); key once")
        else:
            seen[labels] = tag

    if not quiet:
        if findings:
            for f in findings:
                print("  [figcheck] " + f)
        else:
            print(f"  [figcheck] {name}: no findings")
    return findings


# ----------------------------------------------------------------------------------------------
# PDF-level checks
# ----------------------------------------------------------------------------------------------
BAD_FONTS = re.compile(r"DejaVu|Helvetica|Arial|Bitstream|Nimbus|Liberation|Times", re.I)
CM_FONTS = re.compile(r"CM[A-Z]*\d|SFRM|SFTT|SFSS|CMR|CMMI|CMSY|CMEX|MSBM|MSAM|LMRoman|LMMath", re.I)


def check_pdf(path):
    out = []
    try:
        info = subprocess.run(["pdfinfo", path], capture_output=True, text=True, check=True).stdout
        m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
        if m:
            w_in = float(m.group(1)) / 72.0
            W = min((TEXTWIDTH_IN, COLUMNWIDTH_IN), key=lambda W: abs(w_in - W))
            scale = W / w_in            # what \includegraphics[width=W] multiplies the fonts by
            if abs(scale - 1) < 0.05:
                note = f" (includegraphics rescales by {scale:.3f}: 9 pt -> {9*scale:.1f} pt; bbox_inches='tight' crop)" if abs(scale - 1) > 0.01 else ""
                out.append(f"ok  width {w_in:.2f} in for a {W:.2f} in slot{note}")
            else:
                out.append(f"XX  width {w_in:.2f} in: nearest slot {W:.2f} in rescales fonts by {scale:.2f}")
    except Exception as e:
        out.append(f"?? pdfinfo failed: {e}")
    try:
        fonts = subprocess.run(["pdffonts", path], capture_output=True, text=True, check=True).stdout.splitlines()[2:]
        rows = [re.match(r"^(\S+)\s+(.+?)\s+(\S+)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", ln) for ln in fonts if ln.strip()]
        names = [r.group(1) for r in rows if r]
        emb = [r.group(1) for r in rows if r and r.group(4) == "no"]        # the 'emb' column
        bad = [n for n in names if BAD_FONTS.search(n)]
        noncm = [n for n in names if not CM_FONTS.search(n.split("+")[-1])]
        if not names:
            out.append("XX no fonts (text outlined? or empty)")
        else:
            out.append(("ok " if not bad and not noncm else "XX ") + f"fonts: {', '.join(sorted(set(n.split('+')[-1] for n in names)))}"
                       + (f" — non-Computer-Modern: {', '.join(sorted(set(noncm)))}" if noncm else ""))
        if emb:
            out.append(f"XX fonts not embedded: {', '.join(emb)}")
    except Exception as e:
        out.append(f"?? pdffonts failed: {e}")
    return out


def main(paths):
    if not paths:
        from qmsc.paths import FIGURES_DIR

        paths = sorted(glob.glob(os.path.join(FIGURES_DIR, "*.pdf")))
    bad = 0
    for p in paths:
        res = check_pdf(p)
        bad += sum(r.startswith("XX") for r in res)
        print(os.path.basename(p))
        for r in res:
            print("   " + r)
    print("\nFIGCHECK (pdf):", "ALL OK" if bad == 0 else f"{bad} finding(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
