"""Per-site representation bias b_g(R) = h_R - h_inf of the infinite transverse-field Ising chain in the sigma^z
basis for g = 0.5, 1, 1.5 (Sec. IV and Appendix B 2), assembled from the exact bulk block entropies and the
Monte Carlo doubling differences.

  gapped (g = 0.5, 1.5): h_R = H_{R+1} - H_R from the exact bulk block entropies H_l (Pfaffian / Walsh-Hadamard
      route of :mod:`qmsc.bulk_entropy`, l <= LMAX), with h_inf replaced by h_{LMAX-1}.  The replacement changes
      b(R) by b(LMAX-1).  Values are reported only for R <= R_ok, where b(R) exceeds 100 times the last one-step
      decrement (the rounding floor of the block entropies, about 7e-8 at g = 0.5 for l >= 17 and below 1e-11 at
      g = 1.5) and R <= LMAX - 6.  g = 1.5 is read from data/bulk_block_entropies.json (LMAX = 20), g = 0.5 is
      computed here with the same engine (LMAX = 22).
  critical (g = 1): b(R) = sum_{k >= 0} D_{2^k R} with D_R = h_R - h_{2R}: exact from the block entropies for
      2R <= 19, Monte Carlo (:mod:`qmsc.critical_tail`, exact conditionals averaged over sampled contexts) at
      R = 16, 32, 64, and the tail beyond R = 128 taken as b(128) = 1.5 D_96, the value implied by the 1/R law.
      Reported at R = 1, 2, 4, 8, 16, 32 with the tail fraction of each value.

Usage:  python scripts/compute_bulk_bias.py
Output: data/bulk_bias.json
"""
import json
import os
import time

import numpy as np

from qmsc import bulk_entropy as D8
from qmsc import tfim_chain as T

from qmsc.paths import DATA_DIR as RESULTS
SIGN = -1          # bulk correlator convention fixed by bulk_entropy.validate()
LMAX = int(os.environ.get("BULK_LMAX", "22"))


def bulk_H(g, lmax):
    H = [0.0]
    for l in range(1, lmax + 1):
        t = time.time()
        p = D8.block_probs_from_M(l, D8.bulk_M(l, g, SIGN))
        Hl, pmin, nrm = D8.shannon(p)
        if pmin < -1e-10 or nrm > 1e-10:
            raise SystemExit(f"block law failed at l={l}: pmin={pmin:.1e} norm err={nrm:.1e}")
        H.append(Hl)
        print(f"  g={g} l={l:2d}  H={Hl:.10f}  pmin={pmin:.1e}  [{time.time()-t:.1f}s]")
    return H


def validate_g05():
    """Bulk block law at g = 0.5 against the central block of the ED chain (finite-size accuracy)."""
    print("== bulk block law at g=0.5 vs ED central block (n=16, l=4) ==")
    n, l = 16, 4
    psi = T.ground_state(n, 0.5).psi
    p = T.born_distribution(psi, n)
    i0 = (n - l) // 2
    marg = T.marginal(p, i0 + 1, i0 + l).ravel()
    pb = D8.block_probs_from_M(l, D8.bulk_M(l, 0.5, SIGN))
    # T.marginal is MSB-first over sites; block_probs_from_M indexes bit i = site i (LSB-first)
    pb_msb = np.array([pb[int(f"{x:0{l}b}"[::-1], 2)] for x in range(2 ** l)])
    err = float(np.abs(pb_msb - marg).max())
    print(f"  max |p_bulk - p_ED(n=16)| = {err:.2e} (finite-size level expected)")
    return err


def main():
    d8 = json.load(open(os.path.join(RESULTS, "bulk_block_entropies.json")))
    d9 = json.load(open(os.path.join(RESULTS, "critical_tail_mc.json")))
    out = {"LMAX": LMAX, "sign": SIGN, "g": {}}

    err = validate_g05()
    prev = os.path.join(RESULTS, "bulk_bias.json")
    H05 = None
    if os.path.exists(prev) and os.environ.get("BULK_RECOMPUTE") != "1":
        old = json.load(open(prev))
        if old.get("LMAX") == LMAX and len(old.get("H_g05", [])) == LMAX + 1:
            H05 = old["H_g05"]; print(f"== reusing stored H_l at g=0.5 (l <= {LMAX}; BULK_RECOMPUTE=1 to redo) ==")
    if H05 is None:
        print("== exact bulk block entropies at g=0.5 ==")
        H05 = bulk_H(0.5, LMAX)
    out["H_g05"] = H05
    out["validation_g05_vs_ED16_l4"] = err

    # gapped: b(R) = h_R - h_{LMAX-1}
    for g, H in (("0.5", H05), ("1.5", [0.0] + list(np.cumsum(np.array(d8["g"]["1.5"]["h"]))))):
        H = np.array(H)
        h = np.diff(H)                      # h[R] = H_{R+1} - H_R, R = 0..lmax-1
        lmax = len(h)
        hinf = h[lmax - 1]
        b = h - hinf
        # numerical floor: the one-step decrements D(R) = h_R - h_{R+1} stop decaying where the block
        # entropies reach their rounding floor (g = 0.5: ~7e-8 for l >= 17). Report b(R) only where it
        # exceeds 100x the last decrement and R <= lmax - 6 (truncation of h_inf below 5 %).
        D = h[:-1] - h[1:]
        floor = float(D[-1])
        R_ok = max(R for R in range(lmax) if b[R] >= 100 * floor and R <= lmax - 6)
        out["g"][g] = {"R": list(range(0, lmax)), "b": [float(x) for x in b], "h_inf_proxy": float(hinf),
                       "h": [float(x) for x in h], "lmax": lmax, "R_ok": int(R_ok), "decrement_floor": floor,
                       "note": f"h_inf replaced by h_{lmax-1}; values trusted for R <= R_ok"}
        print(f"g={g}: b(R) = ", " ".join(f"{x:.2e}" for x in b[:12]), f"  R_ok={R_ok} (floor {floor:.1e})")

    # critical: doubling sums
    h1 = np.array(d8["g"]["1.0"]["h"])       # h[R], R = 0..19 exact
    D_exact = {R: float(h1[R] - h1[2 * R]) for R in range(1, 10)}
    D_mc = {r["R"]: (r["D"], r["se"]) for r in d9["Z_g1"]}
    tail128 = 1.5 * D_mc[96][0]              # b(128) under the 1/R law pinned by D_96
    tail128_se = 1.5 * D_mc[96][1]
    rows = []
    for R in (1, 2, 4, 8, 16, 32):
        terms, se2, r = [], 0.0, R
        while r < 128:
            if r in D_exact:
                terms.append(("exact", r, D_exact[r]))
            else:
                terms.append(("mc", r, D_mc[r][0])); se2 += D_mc[r][1] ** 2
            r *= 2
        b = sum(t[2] for t in terms) + tail128
        rows.append({"R": R, "b": b, "se": float(np.sqrt(se2 + tail128_se ** 2)),
                     "tail_fraction": tail128 / b, "terms": terms})
        print(f"g=1 R={R:2d}: b={b:.4e}  tail {100*tail128/b:.1f}%  terms={[(t[0], t[1]) for t in terms]}")
    out["g"]["1"] = {"R": [r["R"] for r in rows], "b": [r["b"] for r in rows], "se": [r["se"] for r in rows],
                     "tail_fraction": [r["tail_fraction"] for r in rows], "tail_b128": tail128,
                     "D_exact": D_exact, "kappa_LG_nats": d8["LG_kappa_nats"]}
    with open(os.path.join(RESULTS, "bulk_bias.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote data/bulk_bias.json")


if __name__ == "__main__":
    main()
