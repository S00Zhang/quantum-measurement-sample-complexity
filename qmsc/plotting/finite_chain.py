"""Figures and statistics of the finite-chain study (Appendix C of the paper): the complete Born distribution of
the 50-site open chain learned by the histogram learner.  Everything is read from the stored result files
finite_chain_campaign_n50_g{0.5,1,1.5}.json, nothing is sampled here.

  fig_finite_chain_validation   Fig. 7, measured against exact estimation excess, and the decomposition at N = 1e4
  fig_finite_chain_scaling      Fig. 8, optimal error and optimal radius against N
  campaign_statistics           optimum counts, slopes, exponents and fits (finite_chain_n50_statistics.json)
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
from qmsc.plotting.bulk import G_COLORS, MUTED, TEXT, XN, R_CMAP, style  # noqa: E402


def load(prefix):
    D = {"g": {}}
    for g in ("0.5", "1", "1.5"):
        path = os.path.join(RESULTS, f"{prefix}_g{g}.json")
        d = json.load(open(path))
        Vh = np.array(d["V_hat"])
        vis = np.array(d["cells_visited"])
        D["n"] = d["n"]
        D["N"] = np.array(d["N_grid"], float)
        D["R"] = np.array(d["R_grid"])
        D["seeds"] = d["seeds"]
        D["g"][g] = {
            "B": np.array(d["B_ref"]),
            "B_se": d.get("B_se", 0.0),
            "Vh": Vh,
            "Vm": np.nanmean(Vh, 0),
            "Vse": np.nanstd(Vh, 0, ddof=1) / np.sqrt(Vh.shape[0]),
            "vis": np.nanmean(vis, 0),
            "Ve": np.array(d["V_exact"]),
            "docc": np.array(d["d_occ"]),
            "dR": np.array(d["d_R"], float),
            "vis_seed": vis,
            "clip": np.array(d.get("clip_would_bind", np.zeros_like(Vh))),
            "R_meas": np.array(d["R_star_measured"]),
            "R_ex": np.array(d["R_star_exact"]),
            "R_reg": np.array(d["R_star_regular"]),
            "F_meas": np.array(d["F_star_measured"]),
            "F_ex": np.array(d["F_star_exact"]),
            "T": np.array(d.get("T_tail", [])),
        }
    return D


def rcol(R, Rmax):
    return R_CMAP(0.08 + 0.85 * R / Rmax)


def bias_fit(B, n, lo=3, hi=12):
    """Exponential fit of the site-averaged bias b(R) = B(R)/n over R in [lo, hi]: (A, xi), b ~ A e^{-R/xi}."""
    hi = min(hi, len(B) - 1)
    R = np.arange(lo, hi + 1)
    y = np.log(B[lo : hi + 1] / n)
    slope, inter = np.polyfit(R, y, 1)
    return (float(np.exp(inter)), float(-1.0 / slope))


def _load():
    """The n = 50 campaign data and the exponential bias fits over the radii the optimizer visits."""
    D = load("finite_chain_campaign_n50")
    fits = {
        g: bias_fit(D["g"][g]["B"], D["n"], int(D["g"][g]["R_meas"].min()), int(D["g"][g]["R_meas"].max()))
        for g in ("0.5", "1.5")
    }
    return (D, fits)


def panel_ratio(ax, D, legend=True):
    N = D["N"]
    for g in ("0.5", "1", "1.5"):
        r = D["g"][g]
        ratio = (r["Vm"] / r["Ve"]).ravel()
        NN = np.repeat(N, r["Vm"].shape[1])
        ax.plot(NN, ratio, "o", ms=PS.ms(2.6), color=G_COLORS[g], alpha=0.55, label=f"$g={g}$")
    ax.axhline(1.0, color=TEXT, lw=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0.7, 1.6)
    ax.set_xticks([100.0, 10000.0, 1000000.0])
    style(ax, XN, "measured $\\hat V$ / predicted $V$", ("(a) all (g, R, N) against the identity", "(a)"))
    if legend:
        hs = [
            Line2D([], [], ls="none", marker="o", ms=PS.ms(4), color=G_COLORS[g], label=f"$g={g}$")
            for g in ("0.5", "1", "1.5")
        ]
        ax.legend(
            handles=hs,
            fontsize=PS.LEGEND_PT,
            frameon=False,
            loc="upper right",
            ncol=3,
            handlelength=1.0,
            columnspacing=0.9,
            borderaxespad=0.2,
        )


def panel_decomposition(ax, D, g_show="0.5", N_show=10000.0):
    N = D["N"]
    Rg = D["R"]
    iN = int(np.argmin(np.abs(N - N_show)))
    r = D["g"][g_show]
    B = r["B"]
    Ve = r["Ve"][iN]
    Fm = B[None, :] + r["Vh"][:, iN, :]
    Fmean = Fm.mean(0)
    Fse = Fm.std(0, ddof=1) / np.sqrt(Fm.shape[0])
    col = G_COLORS[g_show]
    ax.plot(Rg, B, "-", lw=2.0, color="black", zorder=3, label="$B_g(R)$")
    ax.plot(Rg, Ve, "--", lw=1.4, color=MUTED, zorder=3, label="$V_g(R,N)$, exact")
    ax.plot(Rg, B + Ve, "-", lw=1.6, color=col, zorder=4, label="$B_g+V_g$")
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
        label="measured $\\hat F_g(R,N)$",
    )
    kp = int(np.argmin(B + Ve))
    k = int(np.argmin(Fmean))
    ax.plot([Rg[kp]], [(B + Ve)[kp]], "o", ms=PS.ms(8), color=col, mfc=col, zorder=6)
    ax.set_yscale("log")
    ax.set_ylim(0.001, 50)
    ax.set_xticks(range(0, 13, 2))
    style(
        ax,
        "context radius $R$",
        "population error [nats]",
        (f"(b) g = {g_show}, N = {N_show:.0e}", f"(b) $g={g_show}$, $N=10^{{{int(np.log10(N_show))}}}$"),
    )
    ax.legend(fontsize=PS.LEGEND_PT, frameon=False, loc="upper right", handlelength=2.0, labelspacing=0.3)
    return (kp, k)


def panel_optimal_error(ax, D, legend=True):
    N = D["N"]
    for g in ("0.5", "1", "1.5"):
        r = D["g"][g]
        ax.plot(N, r["F_ex"], "-", lw=1.8, color=G_COLORS[g], zorder=3)
        ax.plot(N, r["F_meas"], "o", ms=PS.ms(5.5), color=G_COLORS[g], mfc="white", mew=PS.mew(1.4), zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([100.0, 10000.0, 1000000.0])
    ax.set_ylim(2e-05, 3.0)
    style(ax, XN, "$\\mathcal E^\\star(N)$ [nats]", ("(a) optimal error: points measured, lines exact", "(a)"))
    if legend:
        hs = [Line2D([], [], lw=1.8, color=G_COLORS[g], label=f"$g={g}$") for g in ("0.5", "1", "1.5")]
        hs += [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=PS.ms(5.5),
                color=MUTED,
                mfc="white",
                mew=PS.mew(1.4),
                label="measured",
            ),
            Line2D([], [], lw=1.8, color=MUTED, label="exact prediction"),
            Line2D([], [], lw=1.1, ls="--", color=MUTED, label="logarithmic law"),
        ]
        ax.legend(
            handles=hs,
            fontsize=PS.LEGEND_PT,
            frameon=False,
            loc="lower left",
            handlelength=1.8,
            labelspacing=0.3,
            ncol=2,
            columnspacing=1.0,
            borderaxespad=0.3,
        )


def panel_optimal_radius(ax, D, fits):
    N = D["N"]
    for g in ("0.5", "1", "1.5"):
        rr = D["g"][g]
        if g in fits:
            A, xi = fits[g]
            a = 1.0 / (1.0 / xi + np.log(2.0))
            b = a * np.log(2 * A / (xi * np.log(2)))
            Nl = np.logspace(2, 6, 200)
            ax.plot(Nl, a * np.log(Nl) + b, "--", lw=1.1, color=G_COLORS[g], alpha=0.9, zorder=2)
        ax.plot(N, rr["R_ex"], "-", lw=1.8, color=G_COLORS[g], drawstyle="steps-mid", zorder=3)
        ax.plot(N, rr["R_meas"], "o", ms=PS.ms(5.5), color=G_COLORS[g], mfc="white", mew=PS.mew(1.4), zorder=5)
    ax.set_xscale("log")
    ax.set_ylim(0, 13.5)
    ax.set_xticks([100.0, 10000.0, 1000000.0])
    style(ax, XN, "$R^\\star(N)$", ("(b) optimal radius: points measured, steps exact, dashed log law", "(b)"))


def fig_finite_chain_validation(D):
    """Fig. 7: (a) measured over exact estimation excess against N for every coupling and radius, (b) the
    decomposition of the error at g = 0.5 and N = 1e4."""
    fig, axes = plt.subplots(1, 2, figsize=PS.figsize(10.4, 4.0))
    panel_ratio(axes[0], D, legend=True)
    kp, k = panel_decomposition(axes[1], D)
    fig.tight_layout(rect=(0, 0, 1, 1))
    PS.save(fig, "fig7_finite_chain_validation")
    print(f"  decomposition n = 50, g = 0.5, N = 1e4: predicted R* = {kp}, measured minimum {k}")
    return fig


def fig_finite_chain_scaling(D, fits):
    """Fig. 8: optimal error and optimal radius of the 50-site chain against N, with the exact prediction."""
    fig, axes = plt.subplots(1, 2, figsize=PS.figsize(10.4, 4.0))
    panel_optimal_error(axes[0], D, legend=True)
    panel_optimal_radius(axes[1], D, fits)
    fig.tight_layout(rect=(0, 0, 1, 1))
    PS.save(fig, "fig8_finite_chain_scaling")
    return fig


def campaign_statistics(D, fits, tag="n50"):
    """Quantities quoted in Appendix C 2 (printed, and written to data/finite_chain_{tag}_statistics.json)."""
    N = D["N"]
    n = D["n"]
    out = {"n": n, "seeds": D["seeds"]}
    Rh = min(12, int(D["R"].max()))
    n_ex = n_reg = 0
    miss_reg = {}
    for g in ("0.5", "1", "1.5"):
        r = D["g"][g]
        z = (r["Vm"] - r["Ve"]) / r["Vse"]
        rng = np.random.default_rng(0)
        S = r["Vh"].shape[0]
        slopes, expo = ([], [])
        for t in range(1000):
            idx = rng.integers(0, S, S)
            T = r["B"][None, :] + np.nanmean(r["Vh"][idx], 0)
            Rs = np.argmin(T, 1)
            Es = T.min(1)
            slopes.append(np.polyfit(np.log(N), Rs, 1)[0])
            expo.append(np.polyfit(np.log(N[-5:]), np.log(Es[-5:]), 1)[0])
        n_ex += int((r["R_meas"] == r["R_ex"]).sum())
        n_reg += int((r["R_meas"] == r["R_reg"]).sum())
        miss_reg[g] = int((r["R_meas"] != r["R_reg"]).sum())
        A, xi = fits.get(g, (np.nan, np.nan))
        out[g] = {
            "within_2se": float(np.mean(np.abs(z) <= 2)),
            "within_3se": float(np.mean(np.abs(z) <= 3)),
            "max_abs_z": float(np.nanmax(np.abs(z))),
            "grid_points": int(z.size),
            "R_star_measured": r["R_meas"].tolist(),
            "R_star_exact": r["R_ex"].tolist(),
            "R_star_regular": r["R_reg"].tolist(),
            "F_star_measured": r["F_meas"].tolist(),
            "slope_Rstar_vs_lnN": float(np.polyfit(np.log(N), r["R_meas"], 1)[0]),
            "slope_se_boot": float(np.std(slopes)),
            "exponent_last5": float(np.polyfit(np.log(N[-5:]), np.log(r["F_meas"][-5:]), 1)[0]),
            "exponent_se_boot": float(np.std(expo)),
            "fit_A": A,
            "fit_xi": xi,
            "fit_window": [int(r["R_meas"].min()), int(r["R_meas"].max())] if g != "1" else None,
            "pred_slope": float(1 / (1 / xi + np.log(2))) if g != "1" else None,
            "pred_exponent": float(-1 / (1 + xi * np.log(2))) if g != "1" else None,
            "B_se_tail": r["B_se"],
            "clip_cells_total": float(np.nansum(r["clip"])),
            "frac_visited_Rhi": (r["docc"][:, Rh] / r["dR"][Rh]).tolist(),
            "frac_visited_R8": (r["docc"][:, 8] / r["dR"][8]).tolist(),
            "Rhi": Rh,
        }
        print(
            f"g={g}: {z.size} points, 2SE {out[g]['within_2se']:.3f}, 3SE {out[g]['within_3se']:.3f}, max|z| {out[g]['max_abs_z']:.2f}; R* meas {r['R_meas'].tolist()} exact {r['R_ex'].tolist()} regular {r['R_reg'].tolist()}; slope {out[g]['slope_Rstar_vs_lnN']:.3f}({out[g]['slope_se_boot']:.3f}) pred {out[g]['pred_slope']}; exponent(last 5) {out[g]['exponent_last5']:.3f}({out[g]['exponent_se_boot']:.3f}) pred {out[g]['pred_exponent']}; clip cells {out[g]['clip_cells_total']:.0f}"
        )
    out["kvis_ratio_Nmax"] = {}
    for g in ("0.5", "1", "1.5"):
        r = D["g"][g]
        ratio = np.nanmean(2 * N[None, :, None] * r["Vh"] / np.array(r["vis_seed"]), 0)[-1]
        out["kvis_ratio_Nmax"][g] = ratio.tolist()
    out["agree_exact"] = n_ex
    out["agree_regular"] = n_reg
    out["regular_misses_by_g"] = miss_reg
    print(f"optima: exact {n_ex}/27, regular {n_reg}/27, regular misses by g {miss_reg}")
    json.dump(out, open(os.path.join(RESULTS, f"finite_chain_{tag}_statistics.json"), "w"), indent=1)
    return out
