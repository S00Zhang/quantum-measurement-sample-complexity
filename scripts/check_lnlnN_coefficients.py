"""
Check of the ln ln N coefficients of rows II and III of Table I, (nu - 1)/alpha and (mu + 1)/beta
(Sec. II A and Appendix D 1).

Analytic.  With G = 1/C the marginal balance is -B'(s) = G'(s)/N.
  Row II  (B = e^{-alpha s}, G = s^nu):     alpha e^{-alpha s} = nu s^{nu-1}/N
          => alpha s = ln N - (nu - 1) ln s + ln(alpha/nu)   => s* = [ln N - (nu - 1) ln ln N]/alpha + O(1).
  Row III (B = s^{-mu},   G = e^{beta s}):  mu s^{-mu-1} = beta e^{beta s}/N
          => beta s = ln N - (mu + 1) ln s + ln(mu/beta)     => s* = [ln N - (mu + 1) ln ln N]/beta + O(1).
The asymmetry is the derivative: d/ds of s^nu lowers the power by one (nu - 1), d/ds of s^{-mu} raises
the inverse power by one (mu + 1).

Numerical.  Solve the two balance equations for s*(N) at ln N from 30 to 600 and fit
s* = a + b ln N + c ln ln N by least squares: c must approach -(nu - 1)/alpha and -(mu + 1)/beta, b must
approach 1/alpha and 1/beta.  The O(1) remainder and the o(1) corrections of order ln ln N / ln N limit
the agreement at finite N, so the fit is repeated over two windows.  The script also checks that the
ln ln N term of s* enters E* only as a relative correction.  No sampling, no data.

Usage: python scripts/check_lnlnN_coefficients.py
Output: data/lnlnN_coefficients_check.json
"""
import json
import os
import numpy as np
from scipy.optimize import brentq

from qmsc.paths import DATA_DIR


def solve_row2(alpha, nu, lnN):
    f = lambda s: alpha * s + (nu - 1) * np.log(s) - (lnN + np.log(alpha / nu))
    return brentq(f, 1e-6, 10 * lnN / alpha + 100)


def solve_row3(beta, mu, lnN):
    f = lambda s: beta * s + (mu + 1) * np.log(s) - (lnN + np.log(mu / beta))
    return brentq(f, 1e-6, 10 * lnN / beta + 100)


def fit(lnN, s):
    X = np.column_stack([np.ones_like(lnN), lnN, np.log(lnN)])
    coef, *_ = np.linalg.lstsq(X, s, rcond=None)
    return coef  # a, b, c


def main():
    out = {"row_II": [], "row_III": []}
    print("row II:  s* = [ln N - (nu-1) ln ln N]/alpha + O(1)")
    for alpha, nu in ((1.0, 2.0), (0.7, 1.5), (1.5, 3.0), (1.0, 1.0)):
        for lo, hi in ((30, 200), (200, 600)):
            lnN = np.linspace(lo, hi, 400); s = np.array([solve_row2(alpha, nu, x) for x in lnN])
            a, b, c = fit(lnN, s)
            out["row_II"].append({"alpha": alpha, "nu": nu, "window": [lo, hi], "b": b, "b_expected": 1 / alpha, "c": c, "c_expected": -(nu - 1) / alpha})
            print(f"  alpha={alpha} nu={nu} lnN in [{lo},{hi}]: b={b:.4f} (1/alpha={1/alpha:.4f})  c={c:.4f} (-(nu-1)/alpha={-(nu-1)/alpha:.4f})")
    print("row III: s* = [ln N - (mu+1) ln ln N]/beta + O(1)")
    for beta, mu in ((np.log(2), 1.0), (1.0, 2.0), (0.5, 1.0), (np.log(2), 0.5)):
        for lo, hi in ((30, 200), (200, 600)):
            lnN = np.linspace(lo, hi, 400); s = np.array([solve_row3(beta, mu, x) for x in lnN])
            a, b, c = fit(lnN, s)
            out["row_III"].append({"beta": beta, "mu": mu, "window": [lo, hi], "b": b, "b_expected": 1 / beta, "c": c, "c_expected": -(mu + 1) / beta})
            print(f"  beta={beta:.4f} mu={mu} lnN in [{lo},{hi}]: b={b:.4f} (1/beta={1/beta:.4f})  c={c:.4f} (-(mu+1)/beta={-(mu+1)/beta:.4f})")
    # sharper test: with the right coefficient the remainder r(N) = alpha s* - ln N + (nu-1) ln ln N converges to the
    # constant ln(alpha/nu) + (nu-1) ln alpha, while the wrong coefficient (nu+1) makes it drift by -2 ln ln N
    print("remainders r(N) at ln N = 30, 100, 300, 600 (should converge to the constant in brackets):")
    out["remainders"] = []
    for alpha, nu in ((1.0, 2.0), (1.5, 3.0)):
        r = [alpha * solve_row2(alpha, nu, x) - x + (nu - 1) * np.log(x) for x in (30, 100, 300, 600)]
        const = np.log(alpha / nu) + (nu - 1) * np.log(alpha)
        out["remainders"].append({"row": "II", "alpha": alpha, "nu": nu, "r": r, "limit": const})
        print(f"  row II  alpha={alpha} nu={nu}: " + " ".join(f"{v:.4f}" for v in r) + f"  [{const:.4f}]")
    for beta, mu in ((np.log(2), 1.0), (1.0, 2.0)):
        r = [beta * solve_row3(beta, mu, x) - x + (mu + 1) * np.log(x) for x in (30, 100, 300, 600)]
        const = np.log(mu / beta) + (mu + 1) * np.log(beta)
        out["remainders"].append({"row": "III", "beta": beta, "mu": mu, "r": r, "limit": const})
        print(f"  row III beta={beta:.4f} mu={mu}: " + " ".join(f"{v:.4f}" for v in r) + f"  [{const:.4f}]")
    # worst relative deviation of c on the large window
    worst = max(abs(r["c"] / r["c_expected"] - 1) for r in out["row_II"] + out["row_III"] if r["window"][0] == 200 and r["c_expected"] != 0)
    out["worst_rel_dev_c_large_window"] = worst
    print(f"worst relative deviation of the ln ln N coefficient on ln N in [200, 600]: {worst:.3f}")
    # the ln ln N term of s* reaches E* only as a relative correction:
    #   row II : E* = (s*^nu/N)[1 + nu/(alpha s*)] = alpha^-nu (ln N)^nu / N [1 - nu(nu-1) lnlnN/lnN + O(1/lnN)]
    #   row III: E* = s*^-mu [1 + mu/(beta s*)]    = beta^mu (ln N)^-mu    [1 + mu(mu+1) lnlnN/lnN + O(1/lnN)]
    # Compare the exact ratio E*/leading form (s* from the balance equation) with the bracketed factor.
    out["Estar_correction"] = []
    print("E*/leading form against the ln ln N correction factor (row II alpha=1 nu=2, row III beta=ln2 mu=1):")
    for lnN in (20.0, 50.0, 100.0, 300.0):
        N = np.exp(lnN)
        alpha, nu = 1.0, 2.0
        s = solve_row2(alpha, nu, lnN); E = np.exp(-alpha * s) + s ** nu / N
        r2 = E / (lnN ** nu / (alpha ** nu * N)); f2 = 1 - nu * (nu - 1) * np.log(lnN) / lnN
        beta, mu = np.log(2), 1.0
        s3 = solve_row3(beta, mu, lnN); E3 = s3 ** (-mu) + np.exp(beta * s3 - lnN)
        r3 = E3 / ((beta / lnN) ** mu); f3 = 1 + mu * (mu + 1) * np.log(lnN) / lnN
        out["Estar_correction"].append({"lnN": lnN, "rowII_ratio": r2, "rowII_factor": f2, "rowIII_ratio": r3, "rowIII_factor": f3})
        print(f"  lnN={lnN:5.0f}: row II ratio {r2:.4f} factor {f2:.4f} | row III ratio {r3:.4f} factor {f3:.4f}")
    json.dump(out, open(os.path.join(DATA_DIR, "lnlnN_coefficients_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
