"""Figures and statistics of the bulk-site study (Secs. IV to VI of the paper).

Everything is read from the stored result files, nothing is sampled here:

  bulk_bias.json             exact block entropies and the per-site bias b_g(R) of the infinite chain,
  bulk_site_campaign.json    the learning campaign at a bulk site (measured and exact estimation excess, optima),
  critical_tail_mc.json      the Monte Carlo doubling differences of the critical-tail test.

  fig_bulk_bias              Fig. 2  (figures/fig2_bulk_bias.pdf), per-site bias against the context radius
  fig_estimation_cost        Fig. 3  (fig3_estimation_cost.pdf), measured and exact estimation excess against N,
                             and the occupancy of the contexts
  fig_capacity_selection     Fig. 4  (fig4_capacity_selection.pdf), bias, estimation cost and their sum against R
  fig_scaling                Fig. 5  (fig5_scaling.pdf), optimal error and radius against N with the templates
  fig_critical_tail          Fig. 6  (fig6_critical_tail.pdf), the compensated doubling difference 2R D_R at g = 1
  campaign_statistics        the validation statistics, optima, slopes and exponents quoted in the text
                             (data/bulk_campaign_statistics.json)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from qmsc.plotting import style as PS  # noqa: E402

PS.apply()
import numpy as np  # noqa: E402

from qmsc.paths import DATA_DIR as RESULTS  # noqa: E402

G_COLORS = {"0.5": "#2a78d6", "1": "#eb6834", "1.5": "#1baf7a"}


TEXT, MUTED = ("#0b0b0b", "#52514e")


R_CMAP = plt.cm.viridis


XN = "sample size $N$"


XR = "context radius $R$"


R_OK = {"0.5": 10, "1": 17, "1.5": 13}


def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    PS.set_title(ax, title, PS.LABEL_PT, color=TEXT)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
    ax.grid(True, which="minor", color="#f1f0eb", lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=PS.TICK_PT)


def load():
    camp = json.load(open(os.path.join(RESULTS, "bulk_site_campaign.json")))
    out = {"N": np.array(camp["N_grid"], float), "R_MAX": camp["R_MAX"], "seeds": camp["seeds"], "g": {}}
    for g in ("0.5", "1", "1.5"):
        rec = camp["g"][g]
        Vh = np.array(rec["V_hat"])
        vis = np.array(rec["visited"])
        out["g"][g] = {
            "b": np.array(rec["b"]),
            "Vh": Vh,
            "Vm": np.nanmean(Vh, 0),
            "Vse": np.nanstd(Vh, 0, ddof=1) / np.sqrt(Vh.shape[0]),
            "vis": np.nanmean(vis, 0),
            "Ve": np.array(rec["V_exact"]),
            "docc": np.array(rec["d_occ"]),
            "R_meas": np.array(rec["R_star_measured"]),
            "R_ex": np.array(rec["R_star_exact"]),
        }
    return out


KAPPA = 0.0589


_BULK = None


_CAMP = None


def exact_bulk_bias(g):
    """Infinite-chain b_g(R) over the trusted range, with the continuation ratio (gapped) or None
    (critical, continued as kappa/R). Gapped: bulk_bias.json (exact block entropies, R <= R_ok).
    Critical: exact h_R - h_inf for R <= 15 with h_inf anchored by the doubling sums, as stored by
    bulk_site_campaign.py when available, else the doubling-sum points are not used here."""
    global _BULK, _CAMP
    if _BULK is None:
        _BULK = json.load(open(os.path.join(RESULTS, "bulk_bias.json")))
    if g == "1":
        if _CAMP is None:
            path = os.path.join(RESULTS, "bulk_site_campaign.json")
            _CAMP = json.load(open(path)) if os.path.exists(path) else {}
        if "g" in _CAMP and "1" in _CAMP["g"]:
            b = np.array(_CAMP["g"]["1"]["b"])
            return (b, None)
        return (None, None)
    rec = _BULK["g"][g]
    b = np.array(rec["b"])
    R_ok = rec["R_ok"]
    return (b[: R_ok + 1], b[R_ok] / b[R_ok - 1])


def bulk_bias(g, R):
    R = np.asarray(R, float)
    b, ratio = exact_bulk_bias(g)
    if g == "1":
        kap = np.where(R > 0, KAPPA / np.maximum(R, 1e-12), np.inf)
        if b is None:
            return kap
        Ri = R.astype(int)
        R_ok = len(b) - 1
        return np.where(Ri <= R_ok, b[np.minimum(Ri, R_ok)], kap)
    Ri = R.astype(int)
    R_ok = len(b) - 1
    return np.where(Ri <= R_ok, b[np.minimum(Ri, R_ok)], b[R_ok] * ratio ** np.maximum(Ri - R_ok, 0))


_FITS = None


def bias_fits():
    """Exponent-table fits of the gapped bulk bias: {g: (alpha, lnA, lo, hi)} with alpha the mean of
    ln[b(R)/b(R+1)] over the radii the bulk-site optimizer visits, capped at the trusted range R_ok,
    and lnA the least-squares anchor on that window (the dashed lines of Fig. 2)."""
    global _FITS
    if _FITS is None:
        d = bulk_bias_data()
        camp = json.load(open(os.path.join(RESULTS, "bulk_site_campaign.json")))
        _FITS = {}
        for g in ("0.5", "1.5"):
            bb = np.array(d["g"][g]["b"])
            Rm = np.array(camp["g"][g]["R_star_measured"])
            lo, hi = (int(Rm.min()), min(int(Rm.max()), int(d["g"][g]["R_ok"])))
            alpha = float(np.mean(np.log(bb[lo:hi] / bb[lo + 1 : hi + 1])))
            lnA = float(np.mean(np.log(bb[lo : hi + 1]) + alpha * np.arange(lo, hi + 1)))
            _FITS[g] = (alpha, lnA, lo, hi)
    return _FITS


def template_optimum(g, N, R_max=64):
    """Per-site optimum of b_g(R) + 2^R/(2N) over integer R in [0, R_max]: returns (e*, R*), with the
    infinite-chain bias continued geometrically (gapped) or by kappa/R (critical) beyond the exact range."""
    R = np.arange(0, R_max + 1)
    b = bulk_bias(g, R)
    N = np.atleast_1d(np.asarray(N, float))
    tot = b[None, :] + 2.0 ** R[None, :] / (2.0 * N[:, None])
    k = np.argmin(tot, axis=1)
    return (tot[np.arange(len(N)), k], R[k])


def local_exponent(g, lo=1000000.0, hi=1000000000000.0):
    """Log–log slope of the template optimum between lo and hi (least squares on 61 points)."""
    N = np.logspace(np.log10(lo), np.log10(hi), 61)
    e, _ = template_optimum(g, N)
    return float(np.polyfit(np.log(N), np.log(e), 1)[0])


def bulk_bias_data():
    return json.load(open(os.path.join(RESULTS, "bulk_bias.json")))


def fig_bulk_bias(R_show=10):
    """Fig. 2: per-site bias of the infinite chain at R = 1..R_show for the three couplings on one grid.
    Gapped: exact block-entropy differences (bulk_bias.json; at g = 0.5 the rounding floor of the
    block entropies reaches ~5 % of b at R = 12). Critical: exact h_R minus the h_inf anchored by the
    doubling sums (bulk_site_campaign.json), error bar = anchor uncertainty 1e-4 nats. Lines:
    e^{-alpha R} with alpha the mean of ln[b(R)/b(R+1)] over the radii the bulk-site optimizer visits
    (the alpha of the exponent table), anchored by least squares on that window, and kappa/R with the
    Lau-Grassberger value (no fit). The legend quotes alpha."""
    from matplotlib.lines import Line2D

    d = bulk_bias_data()
    camp = json.load(open(os.path.join(RESULTS, "bulk_site_campaign.json")))
    fig, ax = plt.subplots(1, 1, figsize=PS.figsize(5.2, 4.0, width=PS.COLUMNWIDTH * 1.072, paper_h=2.85))
    R = np.arange(1, R_show + 1)
    Rl = np.linspace(0.6, R_show + 0.6, 200)
    handles, fits = ([], {})
    for g in ("0.5", "1", "1.5"):
        if g == "1":
            b = np.array(camp["g"]["1"]["b"])[R]
            ax.errorbar(
                R,
                b,
                yerr=0.0001,
                fmt="o",
                ms=PS.ms(5.0),
                color=G_COLORS[g],
                mfc="white",
                mew=PS.mew(1.3),
                capsize=PS.ms(2),
                zorder=5,
            )
            ax.plot(Rl, KAPPA / Rl, "--", lw=1.1, color=G_COLORS[g], zorder=3)
            label = "$g=1$: $\\kappa/R$, $\\kappa=0.0589$"
        else:
            bb = np.array(d["g"][g]["b"])
            b = bb[R]
            ax.plot(R, b, "o", ms=PS.ms(5.0), color=G_COLORS[g], mfc="white", mew=PS.mew(1.3), zorder=5)
            Rm = np.array(camp["g"][g]["R_star_measured"])
            lo, hi = (int(Rm.min()), int(Rm.max()))
            hi = min(hi, int(d["g"][g]["R_ok"]))
            alpha = float(np.mean(np.log(bb[lo:hi] / bb[lo + 1 : hi + 1])))
            lnA = float(np.mean(np.log(bb[lo : hi + 1]) + alpha * np.arange(lo, hi + 1)))
            ax.plot(Rl, np.exp(lnA - alpha * Rl), "--", lw=1.1, color=G_COLORS[g], zorder=3)
            fits[g] = (alpha, lo, hi)
            label = f"$g={g}$: $e^{{-\\alpha R}}$, $\\alpha={alpha:.2f}$"
        handles.append(Line2D([], [], ls="--", lw=1.1, color=G_COLORS[g], label=label))
    handles.insert(
        0,
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=PS.ms(5.0),
            color=MUTED,
            mfc="white",
            mew=PS.mew(1.3),
            label="population bias",
        ),
    )
    ax.set_xlim(0.5, R_show + 0.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1.5)
    ax.set_xticks(range(0, R_show + 1, 2))
    style(ax, "context radius $R$", "per-site bias $b_g(R)$ [nats]")
    ax.legend(
        handles=handles, fontsize=PS.LEGEND_PT, frameon=False, loc="lower left", handlelength=1.7, labelspacing=0.3
    )
    fig.tight_layout(rect=(0, 0, 1, 1))
    print("  fig_bulk_bias fits (alpha, R window):", {g: (round(v[0], 3), v[1], v[2]) for g, v in fits.items()})
    PS.save(fig, "fig2_bulk_bias")
    plt.close(fig)


def fig_scaling():
    """Fig. 5: measured optima of the bulk-site campaign (bulk_site_campaign.json) with the exact finite-N
    prediction (Proposition 1 with the exact bulk cells, R <= 15) and the asymptotic templates of Table I."""
    camp = json.load(open(os.path.join(RESULTS, "bulk_site_campaign.json")))
    Ng = np.array(camp["N_grid"], float)
    Nd = np.array(camp["N_dense"], float)
    N_TOP = 10000000000.0
    topt = template_optimum
    Nt = np.logspace(2, np.log10(N_TOP), 401)
    R_MAX = camp["R_MAX"]
    fig, axes = plt.subplots(1, 2, figsize=PS.figsize(10.4, 4.0, paper_h=2.2))
    ax = axes[0]
    rows = {}
    for g in ("0.5", "1", "1.5"):
        rec = camp["g"][g]
        b = np.array(rec["b"])
        Vd = np.array(rec["V_exact_dense"])
        tot = b[None, :] + Vd
        k = np.argmin(tot, axis=1)
        e_ex = tot[np.arange(len(Nd)), k]
        okmask = k < R_MAX
        et, Rt = topt(g, Nt)
        ax.plot(Nt, et, "--", lw=1.1, color=G_COLORS[g], alpha=0.9, zorder=2)
        ax.plot(Nd[okmask], e_ex[okmask], "-", lw=1.8, color=G_COLORS[g], zorder=3, label=f"$g={g}$")
        em = np.array(rec["E_star_measured"], float)
        ax.plot(Ng, em, "o", ms=PS.ms(5.5), color=G_COLORS[g], mfc="white", mew=PS.mew(1.4), zorder=5)
        rows[g] = (Nd[okmask], k[okmask], Rt, np.array(rec["R_star_measured"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-09, 0.2)
    if N_TOP > 100000000.0:
        ax.axvspan(100000000.0, N_TOP * 1.5, color="#000000", alpha=0.06, lw=0, zorder=0)
    e1, _ = topt("1", N_TOP / 10**1.3)
    ax.text(
        N_TOP / 10**1.3,
        1.6 * float(e1[0]),
        "row III, $\\mu=1$",
        color=G_COLORS["1"],
        fontsize=PS.LEGEND_PT,
        ha="center",
        va="bottom",
    )
    gapped_label = ax
    style(
        ax,
        XN,
        "$\\mathcal E^\\star(N)$ per site [nats]",
        "(a) e*(N) at a bulk site of the infinite chain: points = measured, solid = exact prediction, dashed = asymptotic-approximation template",
    )
    ax.set_xticks([10.0**k for k in range(2, int(round(np.log10(N_TOP))) + 1, 2)])
    from matplotlib.lines import Line2D

    hs = [Line2D([], [], lw=1.8, color=G_COLORS[g], label=f"$g={g}$") for g in ("0.5", "1", "1.5")]
    hs_style = [
        Line2D(
            [], [], ls="none", marker="o", ms=PS.ms(5.5), color=MUTED, mfc="white", mew=PS.mew(1.4), label="measured"
        ),
        Line2D([], [], lw=1.8, color=MUTED, label="exact prediction"),
        Line2D(
            [],
            [],
            lw=1.1,
            ls="--",
            color=MUTED,
            label="asymptotic prediction",
        ),
    ]
    ax.legend(handles=hs, fontsize=PS.LEGEND_PT, frameon=False, loc="lower left", handlelength=1.8, labelspacing=0.3)
    ax = axes[1]
    for g in ("0.5", "1", "1.5"):
        Ndk, k, Rt, Rm = rows[g]
        ax.plot(Nt, Rt, "--", lw=1.1, color=G_COLORS[g], drawstyle="steps-mid", alpha=0.9, zorder=2)
        ax.plot(Ndk, k, "-", lw=1.8, color=G_COLORS[g], drawstyle="steps-mid", zorder=3)
        ax.plot(Ng, Rm, "o", ms=PS.ms(5.5), color=G_COLORS[g], mfc="white", mew=PS.mew(1.4), zorder=5)
    ax.set_xscale("log")
    if N_TOP > 100000000.0:
        ax.axvspan(100000000.0, N_TOP * 1.5, color="#000000", alpha=0.06, lw=0, zorder=0)
    _, Rt1 = topt("1", N_TOP**0.83)
    ax.text(
        N_TOP**0.76,
        float(Rt1[0]) + 1.2,
        "row III, $\\mu=1$",
        color=G_COLORS["1"],
        fontsize=PS.LEGEND_PT,
        ha="center",
        va="bottom",
    )
    _, Rt15 = topt("1.5", N_TOP**0.95)
    ax.text(
        N_TOP**0.97,
        float(Rt15[0]) - 3.3,
        "$\\simeq\\ln N/(\\alpha+\\beta)$",
        color=MUTED,
        fontsize=PS.LEGEND_PT,
        ha="right",
        va="top",
    )
    style(ax, XN, "$R^\\star(N)$", "(b) R*(N), same conventions")
    ax.legend(
        handles=hs_style, fontsize=PS.LEGEND_PT, frameon=False, loc="upper left", handlelength=1.8, labelspacing=0.3
    )
    ax.set_xticks([10.0**k for k in range(2, int(round(np.log10(N_TOP))) + 1, 2)])
    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=1.2)
    axa = gapped_label
    xs = np.array([N_TOP / 10.0, N_TOP])
    ym = np.sqrt(topt("0.5", xs)[0] * topt("1.5", xs)[0])
    d0, d1 = axa.transData.transform(np.column_stack([xs, ym]))
    ang = float(np.degrees(np.arctan2(d1[1] - d0[1], d1[0] - d0[0])))
    xc = N_TOP / 10**0.75
    yc = float(np.sqrt(topt("0.5", xc)[0][0] * topt("1.5", xc)[0][0]))
    axa.text(
        xc,
        yc,
        "$\\propto N^{-\\alpha/(\\alpha+\\beta)}$",
        color=MUTED,
        fontsize=PS.LEGEND_PT,
        ha="center",
        va="center",
        rotation=ang,
        rotation_mode="anchor",
    )
    PS.save(fig, "fig5_scaling")
    plt.close(fig)


R_COLORS = {4: plt.cm.viridis(0.05), 8: plt.cm.viridis(0.35), 12: plt.cm.viridis(0.55), 16: plt.cm.viridis(0.75)}


def rcol(R):
    return R_COLORS.get(R, R_CMAP(0.08 + 0.85 * R / 16))


def fig_estimation_cost(D, R_show=(4, 8, 16)):
    """Fig. 3: (a) measured and exact per-site estimation excess against N at g = 0.5 for the radii R_show,
    with the parameter-count law 2^R/(2N); (b) the expected fraction of visited contexts."""
    N = D["N"]
    fig, axes = plt.subplots(1, 2, figsize=PS.figsize(10.4, 4.0))
    ax = axes[0]
    r = D["g"]["0.5"]
    for R in R_show:
        col = rcol(R)
        ax.errorbar(
            N,
            r["Vm"][:, R],
            yerr=r["Vse"][:, R],
            fmt="o",
            ms=PS.ms(4),
            color=col,
            mfc="white",
            mew=PS.mew(1.1),
            capsize=PS.ms(2),
            label=f"$R={R}$",
        )
        ax.plot(N, r["Ve"][:, R], "-", lw=1.6, color=col)
        ax.plot(N, 2.0**R / (2.0 * N), "--", lw=1.0, color=col, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3e-08, 3000.0)
    ax.set_xticks([100.0, 10000.0, 1000000.0, 100000000.0])
    style(
        ax,
        XN,
        "$v_g(R,N)$ [nats]",
        ("(a) g = 0.5, bulk site: measured (points), exact (solid), 2^R/(2N) (dashed)", "(a) $g=0.5$"),
    )
    ax.legend(
        fontsize=PS.LEGEND_PT,
        frameon=False,
        loc="upper right",
        ncol=3,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    ax = axes[1]
    for g in ("0.5", "1", "1.5"):
        r = D["g"][g]
        for R, ls in ((4, "-"), (8, ":"), (16, "-.")):
            ax.plot(N, r["docc"][:, R] / 2.0**R, ls, lw=1.5, color=G_COLORS[g], label=f"$g={g}$" if R == 4 else None)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([100.0, 10000.0, 1000000.0, 100000000.0])
    style(ax, XN, "$\\mathbb{E}[\\#\\mathrm{visited\\ cells}]/2^R$", "(b)")
    ax.legend(
        fontsize=PS.LEGEND_PT,
        frameon=False,
        loc="upper center",
        ncol=3,
        handlelength=1.3,
        columnspacing=0.9,
        borderaxespad=0.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 1))
    PS.save(fig, "fig3_estimation_cost")
    plt.close(fig)


YLIM_DECOMP = (3e-07, 3.0)


def _decomp_panel(ax, D, g, iN, letter, show_legend):
    """One panel of Fig. 4: bias, exact estimation cost, their sum and the measured error against R at one
    coupling and sample size.  Returns the radius of the measured minimum."""
    N = D["N"]
    Rg = np.arange(D["R_MAX"] + 1)
    r = D["g"][g]
    b = r["b"]
    ok = R_OK[g]
    Ve = r["Ve"][iN]
    Fm = b[None, :] + r["Vh"][:, iN, :]
    Fmean = np.nanmean(Fm, 0)
    Fse = np.nanstd(Fm, 0, ddof=1) / np.sqrt(Fm.shape[0])
    col = G_COLORS[g]
    ax.plot(Rg[: ok + 1], b[: ok + 1], "-", lw=2.0, color="black", zorder=3, label="$b_g(R)$")
    if ok < D["R_MAX"]:
        ratio = b[ok] / b[ok - 1]
        Rc = Rg[ok:]
        ax.plot(Rc, b[ok] * ratio ** (Rc - ok), ":", color="black", lw=1.6, zorder=3)
    ax.plot(Rg, Ve, "--", lw=1.4, color=MUTED, zorder=3, label="$v_g(R,N)$, exact")
    ax.plot(Rg, b + Ve, "-", lw=1.6, color=col, zorder=4, label="$b_g+v_g$")
    ax.errorbar(
        Rg,
        Fmean,
        yerr=Fse,
        fmt="o",
        ms=PS.ms(5.0),
        color=col,
        mfc="white",
        mew=PS.mew(1.4),
        capsize=PS.ms(2),
        zorder=5,
        label="measured $\\hat f_g(R,N)$",
    )
    kp = int(np.argmin(b + Ve))
    k = int(np.argmin(Fmean))
    ax.plot([Rg[kp]], [(b + Ve)[kp]], "o", ms=PS.ms(8), color=col, mfc=col, zorder=6)
    ax.set_yscale("log")
    ax.set_xticks(range(0, 18, 3))
    ax.set_ylim(YLIM_DECOMP)
    style(
        ax,
        XR,
        "error per site [nats]" if letter == "a" else "",
        (f"({letter}) g = {g}, N = {N[iN]:.0e}", f"({letter}) $g={g}$, $N=10^{{{int(round(np.log10(N[iN])))}}}$"),
    )
    if show_legend:
        ax.legend(fontsize=PS.LEGEND_PT, frameon=False, loc="upper right", handlelength=2.0, labelspacing=0.3)
    print(
        f"  decomposition g = {g}, N = {N[iN]:.0e}: predicted R* = {Rg[kp]} (measured minimum at {Rg[k]}), b = {b[kp]:.3e}, v = {Ve[kp]:.3e}, b/v = {b[kp] / Ve[kp]:.2f}, measured f = {Fmean[kp]:.3e} +- {Fse[kp]:.1e}"
    )
    return k


def fig_capacity_selection(D, N_show=1e6, critical_panel=True):
    """Fig. 4: the decomposition panels at g = 0.5 and g = 1 for the sample size closest to N_show."""
    N = D["N"]
    iN = int(np.argmin(np.abs(N - N_show)))
    if critical_panel:
        fig, axes = plt.subplots(1, 2, figsize=PS.figsize(10.4, 4.0))
        _decomp_panel(axes[0], D, "0.5", iN, "a", True)
        _decomp_panel(axes[1], D, "1", iN, "b", False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=PS.figsize(5.2, 4.0, width=PS.COLUMNWIDTH * 1.072, paper_h=2.7))
        _decomp_panel(ax, D, "0.5", iN, "a", True)
    fig.tight_layout(rect=(0, 0, 1, 1))
    PS.save(fig, "fig4_capacity_selection")
    plt.close(fig)


def fig_critical_tail():
    """Fig. 6 (Appendix B 2): the compensated doubling difference 2R D_R of the critical chain against R, exact
    for R <= 9 and Monte Carlo (critical_tail_mc.json) beyond, with the Lau-Grassberger value."""
    d = bulk_bias_data()
    d9 = json.load(open(os.path.join(RESULTS, "critical_tail_mc.json")))
    fig, ax = plt.subplots(1, 1, figsize=PS.figsize(5.2, 4.0, width=PS.COLUMNWIDTH * 1.072, paper_h=2.6))
    De = d["g"]["1"]["D_exact"]
    Re = np.array(sorted((int(k) for k in De)))
    ve = np.array([2 * r * De[str(r)] for r in Re])
    ax.axhline(KAPPA, ls="--", lw=1.1, color=MUTED, zorder=2)
    ax.plot(Re, ve, "o", ms=PS.ms(5.0), color=G_COLORS["1"], mfc=G_COLORS["1"], mew=PS.mew(1.2), zorder=6)
    Zm = d9["Z_g1"]
    Rmc = np.array([z["R"] for z in Zm])
    kmc = np.array([z["kappa"] for z in Zm])
    kse = np.array([z["kappa_se"] for z in Zm])
    ax.errorbar(
        Rmc,
        kmc,
        yerr=kse,
        fmt="o",
        ms=PS.ms(5.0),
        color=G_COLORS["1"],
        mfc="white",
        mew=PS.mew(1.3),
        capsize=PS.ms(2),
        zorder=5,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.8, 130)
    ax.set_ylim(0.04, 0.066)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["$1$", "$10$", "$100$"])
    ax.set_yticks([0.04, 0.045, 0.05, 0.055, 0.06, 0.065])
    style(ax, "context radius $R$", "$2R\\,D_R$ [nats]")
    hs = [
        Line2D([], [], ls="none", marker="o", ms=PS.ms(5.0), color=G_COLORS["1"], mfc=G_COLORS["1"], label="exact"),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=PS.ms(5.0),
            color=G_COLORS["1"],
            mfc="white",
            mew=PS.mew(1.3),
            label="Monte Carlo",
        ),
        Line2D([], [], ls="--", lw=1.1, color=MUTED, label="$\\kappa=0.0589$"),
    ]
    ax.legend(handles=hs, fontsize=PS.LEGEND_PT, frameon=False, loc="lower right", handlelength=1.7, labelspacing=0.3)
    fig.tight_layout(rect=(0, 0, 1, 1))
    PS.save(fig, "fig6_critical_tail")
    plt.close(fig)


def campaign_statistics(D):
    """Quantities quoted in the captions and the text (printed, and written to bulk_campaign_statistics.json)."""
    N = D["N"]
    out = {}
    for g in ("0.5", "1", "1.5"):
        r = D["g"][g]
        z = (r["Vm"] - r["Ve"]) / r["Vse"]
        ratio = r["Vm"] / r["Ve"]
        Vh = r["Vh"]
        b = r["b"]
        rng = np.random.default_rng(0)
        S = Vh.shape[0]
        slopes, expo = ([], [])
        for t in range(1000):
            idx = rng.integers(0, S, S)
            m = np.nanmean(Vh[idx], 0)
            T = b[None, :] + m
            Rs = np.argmin(T, 1)
            Es = T.min(1)
            slopes.append(np.polyfit(np.log(N), Rs, 1)[0])
            expo.append(np.polyfit(np.log(N), np.log(Es), 1)[0])
        T0 = b[None, :] + r["Vm"]
        R0 = np.argmin(T0, 1)
        E0 = T0.min(1)
        out[g] = {
            "grid_points": int(z.size),
            "within_2se": float(np.mean(np.abs(z) <= 2)),
            "within_3se": float(np.mean(np.abs(z) <= 3)),
            "max_abs_z": float(np.nanmax(np.abs(z))),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "R_star_measured": R0.tolist(),
            "R_star_exact": r["R_ex"].tolist(),
            "E_star_measured": E0.tolist(),
            "slope_Rstar_vs_lnN": float(np.polyfit(np.log(N), R0, 1)[0]),
            "slope_se_boot": float(np.std(slopes)),
            "exponent_Estar_full": float(np.polyfit(np.log(N), np.log(E0), 1)[0]),
            "exponent_se_boot": float(np.std(expo)),
            "exponent_Estar_last5": float(np.polyfit(np.log(N[-5:]), np.log(E0[-5:]), 1)[0]),
            "E_ratio_1e2_over_1e8": float(E0[0] / E0[-1]),
            "frac_visited_R8": (r["docc"][:, 8] / 2**8).tolist(),
            "frac_visited_R12": (r["docc"][:, 12] / 2**12).tolist(),
            "b_floor_share_at_R11": float(1 - b[10] * (b[10] / b[9]) / b[11]) if g == "0.5" else None,
        }
        print(
            f"g={g}: {z.size} points, within 2 SE {out[g]['within_2se']:.3f}, within 3 SE {out[g]['within_3se']:.3f}, max|z| {out[g]['max_abs_z']:.2f}, ratio {ratio.min():.2f}-{ratio.max():.2f}; R*: {R0.tolist()} (exact {r['R_ex'].tolist()}); slope {out[g]['slope_Rstar_vs_lnN']:.3f}({out[g]['slope_se_boot']:.3f}), exponent {out[g]['exponent_Estar_full']:.3f}({out[g]['exponent_se_boot']:.3f}), last-5 exponent {out[g]['exponent_Estar_last5']:.3f}, E*(1e2)/E*(1e8) = {out[g]['E_ratio_1e2_over_1e8']:.0f}"
        )
    allz = np.concatenate([((D["g"][g]["Vm"] - D["g"][g]["Ve"]) / D["g"][g]["Vse"]).ravel() for g in D["g"]])
    out["all"] = {
        "grid_points": int(allz.size),
        "within_2se": float(np.mean(np.abs(allz) <= 2)),
        "within_3se": float(np.mean(np.abs(allz) <= 3)),
    }
    print("all:", out["all"])
    json.dump(out, open(os.path.join(RESULTS, "bulk_campaign_statistics.json"), "w"), indent=1)
    return out
