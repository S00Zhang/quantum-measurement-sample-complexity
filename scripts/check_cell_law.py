"""
Exact check of the per-cell estimation law of the add-1/2 estimator (Lemma 2 of Appendix A).

Claim: for fixed theta in (0, 1), the add-1/2 (Krichevsky-Trofimov) estimate theta_hat = (k + 1/2)/(m + 1),
k ~ Bin(m, theta), satisfies

    E D_KL( Bern(theta) || Bern(theta_hat) ) = 1/(2m) + [ 1/(24 theta(1-theta)) - 5/12 ] / m^2 + O(m^-3),

a fixed-theta asymptotic that is not uniform as theta -> 0 or 1 (the 1/(24 theta(1-theta)) term diverges,
which is the origin of the rare-context regime).  Derivation sketch: D_KL(theta || theta + Delta) expanded to
Delta^4 with d2 = 1/S, d3 = 2(2 theta - 1)/S^2, d4 = 6(1 - 3S)/S^3, S = theta(1 - theta), and the moments of
Delta for the add-1/2 estimate, bias = u/(m+1), E[Delta^2] = [mS + u^2]/(m+1)^2,
E[Delta^3] = [5umS + u^3]/(m+1)^3, E[Delta^4] = [3m^2 S^2 + mS(1-6S) + 14u^2 mS + u^4]/(m+1)^4 with
u = 1/2 - theta.  Collecting the m^-2 terms gives 1/(24S) - 5/12.

The law is verified exactly by binomial sums (no sampling) for the unclipped and clipped estimate and for the
clipped maximum-likelihood estimate.  The clip (eps = 1e-4) leaves the add-1/2 risk unchanged to machine
precision over the tabulated occupancies.

Usage: python scripts/check_cell_law.py
Output: data/cell_law_check.json
"""
import json
import os

import numpy as np
from scipy.stats import binom

from qmsc.paths import DATA_DIR

P_GRID = (0.5, 0.3, 0.1, 0.03, 0.01)


def exact_risk(m: int, p: float, estimator: str = "kt", eps: float = 1e-4) -> float:
    """E_k D_KL(Bern(p) || Bern(q_hat(k))) exactly, k ~ Bin(m, p).  estimator in {kt, kt_clip, mle_clip}."""
    k = np.arange(m + 1)
    w = binom.pmf(k, m, p)
    if estimator == "kt":
        q = (k + 0.5) / (m + 1)
    elif estimator == "kt_clip":
        q = np.clip((k + 0.5) / (m + 1), eps, 1 - eps)
    elif estimator == "mle_clip":
        q = np.clip(k / max(m, 1), eps, 1 - eps)
    else:
        raise ValueError(estimator)
    d = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return float(np.sum(w * d))


def c2_theory(p: float) -> float:
    return 1.0 / (24.0 * p * (1.0 - p)) - 5.0 / 12.0


def main():
    m_grid = np.unique(np.round(np.logspace(0, 3.6, 60)).astype(int))
    out = {"m_grid": m_grid.tolist(), "p_grid": list(P_GRID), "eps": 1e-4,
           "risk_kt": {}, "risk_kt_clip": {}, "risk_mle_clip": {}, "c2_theory": {}}
    for p in P_GRID:
        out["risk_kt"][str(p)] = [exact_risk(m, p, "kt") for m in m_grid]
        out["risk_kt_clip"][str(p)] = [exact_risk(m, p, "kt_clip") for m in m_grid]
        out["risk_mle_clip"][str(p)] = [exact_risk(m, p, "mle_clip") for m in m_grid]
        out["c2_theory"][str(p)] = c2_theory(p)
    os.makedirs(DATA_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(DATA_DIR, "cell_law_check.json"), "w"))

    for p in P_GRID:
        r = np.array(out["risk_kt"][str(p)])
        big = m_grid >= 1000
        c2_num = np.mean((r[big] - 1 / (2 * m_grid[big])) * m_grid[big] ** 2)
        print(f"theta={p:5g}: 2m E at m={m_grid[-1]} -> {2*m_grid[-1]*r[-1]:.5f} | c2 numeric ~ {c2_num:9.3f} vs theory {c2_theory(p):9.3f}")
    p = 0.3
    ktc = np.array(out["risk_kt_clip"][str(p)]); kt = np.array(out["risk_kt"][str(p)])
    print("max |clipped - unclipped| (theta=0.3):", np.abs(ktc - kt).max())
    print("wrote data/cell_law_check.json")


if __name__ == "__main__":
    main()
