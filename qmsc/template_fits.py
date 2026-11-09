"""
Fits of the full Table I templates to the bulk-site campaign (Sec. VI B, Table II, Appendix D 2).

Templates (per site, integer R >= 1, R <= 79):
  row I   (g = 0.5, 1.5): E_T(N) = min_R [ A e^{-alpha R} + e^{beta R}/(2N) ],  parameters (alpha, A, beta)
  row III (g = 1):        E_T(N) = min_R [ kappa R^{-mu}  + e^{beta R}/(2N) ],  parameters (mu, kappa, beta)
The parameter-count cost e^{beta R}/(2N) has beta = ln 2.  beta is nevertheless left free so that the fitted
leading slope of R*, 1/beta, is a measured quantity.

Fit: least squares on the 13 measured optimal errors (ln E*) and optimal radii (R*) of
data/bulk_site_campaign.json, residuals [ln E_T - ln E*, w (R_T - R*)] with w = 0.05 per radius step (the
result is reported for w = 0.02, 0.05, 0.1, 0.2 as a sensitivity check).  Errors from a bootstrap over the
20 training realizations (300 resamples).  Leading forms reported:
  row I:   E* ~ N^{-alpha/(alpha+beta)},  R* ~ ln N/(alpha+beta)
  row III: E* ~ (ln N)^{-mu},             R* ~ ln N/beta
together with the effective exponents of the fitted template over the data window (direct least-squares
slopes of ln E_T against ln N or ln ln N and of R_T against ln N), which are what a direct fit of the data
gives and which the subleading terms of row III push away from the leading forms.  The sensitivity of the
critical fit to a common offset of the bias (the uncertainty of h_inf, 1e-4 nats) is recorded as well.

No sampling.  Usage: python scripts/fit_templates.py
Output: data/template_fits.json
"""
import json
import os
import numpy as np
from scipy.optimize import least_squares

from qmsc.paths import DATA_DIR

RG = np.arange(1, 80, dtype=float)
LN2 = np.log(2.0)


def template(N, row, p):
    if row == "I":
        alpha, lnA, beta = p
        tot = np.exp(lnA - alpha * RG)[None, :] + np.exp(beta * RG)[None, :] / (2.0 * N[:, None])
    else:
        mu, lnk, beta = p
        tot = np.exp(lnk) * RG[None, :] ** (-mu) + np.exp(beta * RG)[None, :] / (2.0 * N[:, None])
    k = tot.argmin(1)
    return tot[np.arange(len(N)), k], RG[k]


def resid(p, N, E, Rm, row, w):
    e, R = template(N, row, p)
    return np.concatenate([np.log(e) - np.log(E), w * (R - Rm)])


def fit(N, E, Rm, row, w, p0):
    f = least_squares(resid, p0, args=(N, E, Rm, row, w))
    return f.x


def leading(row, p):
    if row == "I":
        alpha, lnA, beta = p
        return {"E_exponent": -alpha / (alpha + beta), "R_slope": 1.0 / (alpha + beta), "alpha": alpha, "A": float(np.exp(lnA)), "beta": beta}
    mu, lnk, beta = p
    return {"E_exponent_lnN": -mu, "R_slope": 1.0 / beta, "mu": mu, "kappa": float(np.exp(lnk)), "beta": beta}


def main():
    c = json.load(open(os.path.join(DATA_DIR, "bulk_site_campaign.json")))
    N = np.array(c["N_grid"], float)
    out = {"w_default": 0.05, "N_grid": c["N_grid"], "g": {}}
    for g, row, p0 in (("0.5", "I", [0.9, np.log(0.1), LN2]), ("1.5", "I", [1.5, np.log(0.002), LN2]), ("1", "III", [1.0, np.log(0.0589), LN2])):
        rec = c["g"][g]
        E = np.array(rec["E_star_measured"]); Rm = np.array(rec["R_star_measured"], float)
        Vh = np.array(rec["V_hat"]); b = np.array(rec["b"])
        res = {"row": row, "fits_by_w": {}}
        for w in (0.02, 0.05, 0.1, 0.2):
            p = fit(N, E, Rm, row, w, p0)
            e, R = template(N, row, p)
            res["fits_by_w"][str(w)] = {"params": p.tolist(), "leading": leading(row, p), "R_matches": int((R == Rm).sum()),
                                        "rms_lnE": float(np.sqrt(np.mean((np.log(e) - np.log(E)) ** 2)))}
        p = fit(N, E, Rm, row, 0.05, p0)
        e, R = template(N, row, p)
        lead = leading(row, p)
        # effective exponents over the window: of the data and of the fitted template
        eff = {"data_E_vs_lnN": float(np.polyfit(np.log(N), np.log(E), 1)[0]), "template_E_vs_lnN": float(np.polyfit(np.log(N), np.log(e), 1)[0]),
               "data_E_vs_lnlnN": float(np.polyfit(np.log(np.log(N)), np.log(E), 1)[0]), "template_E_vs_lnlnN": float(np.polyfit(np.log(np.log(N)), np.log(e), 1)[0]),
               "data_R_slope": float(np.polyfit(np.log(N), Rm, 1)[0]), "template_R_slope": float(np.polyfit(np.log(N), R, 1)[0])}
        # bootstrap over seeds
        rng = np.random.default_rng(0); boots = []
        for t in range(300):
            idx = rng.integers(0, Vh.shape[0], Vh.shape[0])
            T = b[None, :] + np.nanmean(Vh[idx], 0); Es = T.min(1); Rs = T.argmin(1).astype(float)
            boots.append(leading(row, fit(N, Es, Rs, row, 0.05, p)))
        keys = list(lead.keys())
        bse = {k: float(np.std([bb[k] for bb in boots])) for k in keys}
        res.update({"params": p.tolist(), "leading": lead, "leading_se": bse, "effective": eff, "R_matches": int((R == Rm).sum()),
                    "rms_lnE": float(np.sqrt(np.mean((np.log(e) - np.log(E)) ** 2))), "template_E": e.tolist(), "template_R": R.tolist()})
        out["g"][g] = res
        print(f"g={g} (row {row}): " + ", ".join(f"{k}={v:.4f}({bse[k]:.4f})" for k, v in lead.items())
              + f"; R* matched {res['R_matches']}/13, rms ln E resid {res['rms_lnE']:.3f}")
        print("   effective over the window:", {k: round(v, 3) for k, v in eff.items()})
        print("   sensitivity to w:", {w: {k: round(v, 3) for k, v in r["leading"].items() if k in ("E_exponent", "E_exponent_lnN", "R_slope")} for w, r in res["fits_by_w"].items()})
    # sensitivity of the critical fit to a common additive offset of the bias (the h_inf uncertainty, 1e-4 nats)
    rec = c["g"]["1"]; E = np.array(rec["E_star_measured"]); Rm = np.array(rec["R_star_measured"], float)
    p0 = [1.0, np.log(0.0589), LN2]; base = fit(N, E, Rm, "III", 0.05, p0)
    out["g"]["1"]["offset_sensitivity"] = {str(d): leading("III", fit(N, E + d, Rm, "III", 0.05, p0)) for d in (-1e-4, 1e-4)}
    out["g"]["1"]["offset_sensitivity"]["0"] = leading("III", base)
    print("critical fit under a common bias offset:", {k: round(v["mu"], 4) for k, v in out["g"]["1"]["offset_sensitivity"].items()})
    json.dump(out, open(os.path.join(DATA_DIR, "template_fits.json"), "w"), indent=1)
    print("wrote template_fits.json")


if __name__ == "__main__":
    main()
