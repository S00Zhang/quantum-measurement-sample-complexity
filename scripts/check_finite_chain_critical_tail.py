"""
How much of the critical kappa/R tail the 50-site open chain retains (Appendix B 3).

From the exact window conditional entropies H(X_i | C_{i,R}) stored by the n = 50 campaign at g = 1
(data/finite_chain_campaign_n50_g1.json, key H_window, R = 0..12) form the doubling difference of site i,
D_{i,R} = H(X_i | C_{i,R}) - H(X_i | C_{i,2R}), exact for 2R <= 12, and compare 2R D_{i,R} at the chain centre
with the exact infinite-chain value 2R D_R from data/bulk_bias.json (kappa = 0.0589 nats under the kappa/R
law).  No sampling.

Usage: python scripts/check_finite_chain_critical_tail.py
Output: data/critical_tail_n50.json
"""
import json
import os

import numpy as np

from qmsc.paths import DATA_DIR


def main():
    d = json.load(open(os.path.join(DATA_DIR, "finite_chain_campaign_n50_g1.json")))
    H = np.array(d["H_window"])            # (n, W+1): H(X_i | C_{i,R})
    n = H.shape[0]
    W = H.shape[1] - 1
    bulk = json.load(open(os.path.join(DATA_DIR, "bulk_bias.json")))["g"]["1"]["D_exact"]
    centre = n // 2                        # site 26 (1-based) of 50
    out = {"n": n, "W": W, "centre_site": centre + 1, "R": [], "centre_2RD": [], "max_2RD": [], "max_site": [],
           "bulk_2RD": [], "ratio_centre_to_bulk": []}
    print(f"n = {n}: 2R D_R at the centre site {centre + 1} against the infinite chain")
    for R in range(1, W // 2 + 1):
        v = 2 * R * (H[:, R] - H[:, 2 * R])
        c = float(v[centre])
        m = float(v.max())
        b = 2 * R * bulk[str(R)]
        out["R"].append(R)
        out["centre_2RD"].append(c)
        out["max_2RD"].append(m)
        out["max_site"].append(int(v.argmax() + 1))
        out["bulk_2RD"].append(b)
        out["ratio_centre_to_bulk"].append(c / b)
        print(f"  R={R}: centre {c:.4f} (max {m:.4f} at site {v.argmax() + 1}), infinite chain {b:.4f}, ratio {c / b:.2f}")
    json.dump(out, open(os.path.join(DATA_DIR, "critical_tail_n50.json"), "w"), indent=1)
    print("wrote data/critical_tail_n50.json")


if __name__ == "__main__":
    main()
