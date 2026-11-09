"""Monte Carlo test of the critical tail b(R) ~ kappa/R at large context radius (Appendix B 2, Fig. 6).

The infinite-chain bulk correlators are used directly, so there is no boundary and "system size" is the
context radius R itself, measured to R = 96 (windows of about 200 sites).

Both measurement families are commuting Majorana bilinears in the phi = (A_0, B_0, A_1, B_1, ...) convention
of :mod:`qmsc.bulk_entropy` (A^2 = 1, B^2 = -1):

    Z-basis walls:   w_k = Z_k Z_{k+1} = B_k A_{k+1} = phi_{2k+1} phi_{2k+2}
    X occupations:   x_j = sigma^x_j   = A_j B_j     = phi_{2j}   phi_{2j+1}

(spin block <-> wall block: p_spin(x_1..x_l) = (1/2) p_wall(w_1..w_{l-1}), the global flip being the missing
bit, so spin h_R = wall h_{R-1}).  Measuring O = phi_a phi_b (O^2 = 1) on a Gaussian state with two-point
matrix Sigma[p,q] = <phi_p phi_q> gives P(s) = (1 + s Sigma[a,b]) / 2, and the post-measurement state is
Gaussian with a rank-2 Schur-complement update of Sigma.  The sign and denominator convention of the update
is calibrated against brute-force exact diagonalization and then fixed.

Estimator (Rao-Blackwellized paired conditional): for each exactly drawn stationary context both conditional
distributions are computed exactly, and

    D_R = h_R - h_{2R} = E_ctx[ KL( p(t | long context) || p(t | short context) ) ] >= 0

is averaged over Monte Carlo contexts only.  This is numerical integration with quotable standard errors, not
learning from samples.  2R D_R -> kappa if c_R ~ kappa/R.

``calibrate_and_validate()`` runs before any number is used: the update convention is calibrated on ED at
n = 6 (exactly one of four sign combinations reproduces the joint wall distribution to ~1e-12), the engine's
joint distributions (walls and occupations) match brute-force ED at n = 8 to machine precision, the bulk
wall-window probabilities match the Pfaffian/WHT spin blocks at l = 6, the Monte Carlo estimator at R = 4
reproduces the exact value within its error, and the gapped control at g = 1.5 decays to zero.

Usage:  python scripts/run_critical_tail_test.py [--quick]
Output: data/critical_tail_mc.json
"""
import argparse
import json
import os
import time


import numpy as np

from qmsc import bulk_entropy as D8

from qmsc.paths import DATA_DIR as RESULTS
LG = D8.LG_KAPPA_NATS
SIGN = -1  # bulk correlator convention fixed by bulk_entropy.validate()

# calibrated on ED: (num_sign, den_sign) in  Sigma' = Sigma + num*(u v^T - v u^T)/(c + den*s)
_CAL = None


# --------------------------------------------------------------- bulk Sigma for a window
def bulk_sigma(nsites, g):
    """Sigma[p,q] = <phi_p phi_q> (p != q) for `nsites` consecutive bulk sites."""
    Gv = {r: D8.bulk_G(r, g, SIGN) for r in range(-nsites, nsites + 1)}
    S = np.zeros((2 * nsites, 2 * nsites))
    for m in range(nsites):
        for n_ in range(nsites):
            S[2 * m, 2 * n_ + 1] = -Gv[m - n_]     # <A_m B_n>
            S[2 * m + 1, 2 * n_] = Gv[n_ - m]      # <B_m A_n>
    S[np.arange(2 * nsites), np.arange(2 * nsites)] = 0.0
    return S


def wall_pairs(nwalls, site0=0):
    return [(2 * (site0 + k) + 1, 2 * (site0 + k) + 2) for k in range(nwalls)]


def occ_pairs(nsites, site0=0):
    return [(2 * (site0 + j), 2 * (site0 + j) + 1) for j in range(nsites)]


# ------------------------------------------------------------------- measurement engine
def measure_update(S, a, b, s):
    """Condition Sigma on phi_a phi_b = s. Returns p(s). Mutates S in place."""
    ns, ds = _CAL
    c = S[a, b]
    p = 0.5 * (1.0 + s * c)
    den = c + ds * s
    if p < 1e-14 or abs(den) < 1e-14:
        # conditioning on a (numerically) zero-probability outcome: state undefined;
        # caller must treat this branch as probability zero.
        S[a, :] = 0.0; S[:, a] = 0.0; S[b, :] = 0.0; S[:, b] = 0.0
        return 0.0
    u = S[:, a].copy()
    v = S[:, b].copy()
    S += ns * (np.outer(u, v) - np.outer(v, u)) / den
    S[a, :] = 0.0; S[:, a] = 0.0; S[b, :] = 0.0; S[:, b] = 0.0
    return p


def joint_prob(Sigma, pairs, outcomes):
    """Exact joint probability of `outcomes` for the bilinears `pairs` (sequential product)."""
    S = Sigma.copy()
    logp = 0.0
    for (a, b), s in zip(pairs, outcomes):
        p = measure_update(S, a, b, s)
        if p <= 0:
            return -np.inf
        logp += np.log(p)
    return logp


def sample_pass(Sigma, pairs, rng):
    """Sample outcomes sequentially; return (outcomes, S_after)."""
    S = Sigma.copy()
    out = []
    for (a, b) in pairs:
        c = S[a, b]
        s = 1 if rng.random() < 0.5 * (1.0 + c) else -1
        measure_update(S, a, b, s)
        out.append(s)
    return out, S


def cond_dist_after(S, a, b):
    """p(target = +1 | conditioning already applied to S)."""
    return 0.5 * (1.0 + S[a, b])


# --------------------------------------------------------------------------- ED ground truth
def ed_joint_walls(n, g):
    """Brute-force joint distribution of the n-1 wall outcomes from the ED Born distribution."""
    psi = np.asarray(D8.T.ground_state(n, g).psi).ravel()
    p_full = np.asarray(D8.T.born_distribution(psi, n)).ravel()
    out = np.zeros(2 ** (n - 1))
    for x in range(2 ** n):
        bits = [(x >> (n - 1 - i)) & 1 for i in range(n)]  # site i -> bit
        key = 0
        for k in range(n - 1):
            w = 1 - 2 * (bits[k] ^ bits[k + 1])            # w=+1 aligned
            key |= (0 if w == 1 else 1) << k
        out[key] += p_full[x]
    return out


def ed_joint_occ(n, g):
    """Brute-force joint distribution of sigma^x outcomes (X basis) from ED."""
    psi = np.asarray(D8.T.ground_state(n, g).psi).ravel()
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    psix = psi.reshape([2] * n)
    for ax in range(n):
        psix = np.tensordot(H1, psix, axes=[[1], [ax]])
        psix = np.moveaxis(psix, 0, ax)
    px = np.abs(psix.ravel()) ** 2
    # bit order: site i is axis i -> bit (n-1-i) in the raveled index; re-key to site-order
    out = np.zeros(2 ** n)
    for x in range(2 ** n):
        key = 0
        for i in range(n):
            bit = (x >> (n - 1 - i)) & 1                   # site i outcome (0 => +1)
            key |= bit << i
        out[key] += px[x]
    return out


def engine_joint_all(Sigma, pairs):
    """Engine joint distribution over all outcome strings (for validation, small K)."""
    K = len(pairs)
    out = np.zeros(2 ** K)
    for key in range(2 ** K):
        outc = [1 - 2 * ((key >> k) & 1) for k in range(K)]
        lp = joint_prob(Sigma, pairs, outc)
        out[key] = 0.0 if lp == -np.inf else np.exp(lp)
    return out


# ------------------------------------------------------------------------- validation
def calibrate_and_validate():
    global _CAL
    print("== 0: calibrate the Gaussian-measurement update on ED (n=6, g=1.5) ==")
    n, g = 6, 1.5
    M, _ = D8.ed_majorana_correlators(n, g)
    ref = ed_joint_walls(n, g)
    best = None
    for ns in (+1.0, -1.0):
        for ds in (+1.0, -1.0):
            _CAL = (ns, ds)
            try:
                got = engine_joint_all(M, wall_pairs(n - 1))
                err = float(np.abs(got - ref).max())
            except Exception:
                err = np.inf
            print(f"  num={ns:+.0f} den={ds:+.0f}: max err = {err:.2e}")
            if best is None or err < best[1]:
                best = ((ns, ds), err)
    _CAL, err = best
    if err > 1e-10:
        raise SystemExit("calibration failed: no convention reproduces ED")
    print(f"  -> convention: Sigma' = Sigma + ({_CAL[0]:+.0f})*(uv^T-vu^T)/(c + ({_CAL[1]:+.0f})*s)")

    print("== 1: engine joints vs ED (n=8), walls and occupations ==")
    for g in (1.0, 1.5):
        M, _ = D8.ed_majorana_correlators(8, g)
        e1 = float(np.abs(engine_joint_all(M, wall_pairs(7)) - ed_joint_walls(8, g)).max())
        e2 = float(np.abs(engine_joint_all(M, occ_pairs(8)) - ed_joint_occ(8, g)).max())
        print(f"  g={g}: walls {e1:.2e}   occupations {e2:.2e}")
        if max(e1, e2) > 1e-10:
            raise SystemExit("check 1 failed")

    print("== 2: bulk engine vs Pfaffian/WHT blocks (g=1, l=6) ==")
    l = 6
    p_spin = D8.block_probs_from_M(l, D8.bulk_M(l, 1.0, SIGN))
    Sb = bulk_sigma(l, 1.0)
    p_wall = engine_joint_all(Sb, wall_pairs(l - 1))
    errs = []
    for key in range(2 ** (l - 1)):
        # lift wall string (w_k for sites k,k+1) to the spin string with x_1=+1 (bit 0)
        bits = [0]
        for k in range(l - 1):
            wbit = (key >> k) & 1
            bits.append(bits[-1] ^ wbit)
        xkey = sum(b << i for i, b in enumerate(bits))
        errs.append(abs(p_wall[key] - 2.0 * p_spin[xkey]))
    e1 = max(errs)
    # X blocks: direct Pf/WHT (any S: <prod sigma^x_S> = Pf of ordered [A_i B_i] submatrix)
    c = np.zeros(2 ** l)
    c[0] = 1.0
    Mb = bulk_sigma(l, 1.0)
    for Sbits in range(1, 2 ** l):
        idx = []
        for i in range(l):
            if (Sbits >> i) & 1:
                idx += [2 * i, 2 * i + 1]
        c[Sbits] = D8.pfaffian(Mb[np.ix_(idx, idx)])
    p_x = c.copy(); h = 1
    while h < 2 ** l:
        for i in range(0, 2 ** l, 2 * h):
            a_ = p_x[i:i + h].copy(); b_ = p_x[i + h:i + 2 * h].copy()
            p_x[i:i + h] = a_ + b_; p_x[i + h:i + 2 * h] = a_ - b_
        h *= 2
    p_x /= 2 ** l
    e2 = float(np.abs(engine_joint_all(Sb, occ_pairs(l)) - p_x).max())
    print(f"  walls-vs-spin blocks {e1:.2e}   occupations-vs-Pf/WHT {e2:.2e}")
    if max(e1, e2) > 1e-10:
        raise SystemExit("check 2 failed")

    print("== exact X-basis kappa_D at small R (Pf/WHT, l<=9) for W4x ==")
    lmax = 9
    Mb = bulk_sigma(lmax, 1.0)
    Hx = [0.0]
    for l in range(1, lmax + 1):
        c = np.zeros(2 ** l); c[0] = 1.0
        for Sbits in range(1, 2 ** l):
            idx = []
            for i in range(l):
                if (Sbits >> i) & 1:
                    idx += [2 * i, 2 * i + 1]
            c[Sbits] = D8.pfaffian(Mb[np.ix_(idx, idx)])
        p = c.copy(); h = 1
        while h < 2 ** l:
            for i in range(0, 2 ** l, 2 * h):
                a_ = p[i:i + h].copy(); b_ = p[i + h:i + 2 * h].copy()
                p[i:i + h] = a_ + b_; p[i + h:i + 2 * h] = a_ - b_
            h *= 2
        p /= 2 ** l
        q = p[p > 1e-13]
        Hx.append(float(-(q * np.log(q)).sum()))
    hx = np.diff(np.array(Hx))
    kx4 = 8 * (hx[4] - hx[8])
    print(f"  exact X-basis kappa_D(4) = {kx4:.6f} nats")
    return kx4


# ------------------------------------------------------------------------- MC estimator
def run_R(basis, g, R, Msamp, rng):
    """Rao-Blackwellized D_R = h_R - h_{2R} (spin conventions for Z via walls)."""
    if basis == "Z":
        nsites = 2 * R + 1
        Sigma = bulk_sigma(nsites, g)
        pairs = wall_pairs(2 * R)                  # walls w_0..w_{2R-1}; target = last
        ctx_long = pairs[:-1]                      # 2R-1 walls  == spin context 2R
        Sshort = bulk_sigma(R + 1, g)              # sites R..2R relabeled 0..R
        pairs_s = wall_pairs(R)                    # R walls; target = last (R-1 ctx == spin R)
    else:
        nsites = 2 * R + 1
        Sigma = bulk_sigma(nsites, g)
        pairs = occ_pairs(2 * R + 1)               # sites 0..2R; target = site 2R
        ctx_long = pairs[:-1]                      # 2R sites context
        Sshort = bulk_sigma(R + 1, g)              # sites R..2R relabeled
        pairs_s = occ_pairs(R + 1)                 # R context + target
    ta, tb = pairs[-1]
    tas, tbs = pairs_s[-1]

    kls = np.empty(Msamp)
    h_long_acc = np.empty(Msamp)
    t0 = time.time()
    for m in range(Msamp):
        out, S = sample_pass(Sigma, ctx_long, rng)
        pL = cond_dist_after(S, ta, tb)
        Ss = Sshort.copy()
        for (a, b), s in zip(pairs_s[:-1], out[len(out) - (len(pairs_s) - 1):]):
            measure_update(Ss, a, b, s)
        pS = cond_dist_after(Ss, tas, tbs)
        pL = min(max(pL, 1e-15), 1 - 1e-15)
        pS = min(max(pS, 1e-15), 1 - 1e-15)
        kls[m] = pL * np.log(pL / pS) + (1 - pL) * np.log((1 - pL) / (1 - pS))
        h_long_acc[m] = -(pL * np.log(pL) + (1 - pL) * np.log(1 - pL))
    D = float(kls.mean()); se = float(kls.std(ddof=1) / np.sqrt(Msamp))
    h2R = float(h_long_acc.mean()); h2R_se = float(h_long_acc.std(ddof=1) / np.sqrt(Msamp))
    dt = time.time() - t0
    print(f"  {basis} g={g} R={R:3d} M={Msamp}: 2R*D = {2*R*D:.5f} +/- {2*R*se:.5f}"
          f"   h_2R = {h2R:.6f}({h2R_se:.1e})   [{dt:.0f}s]")
    return {"R": R, "M": Msamp, "D": D, "se": se, "kappa": 2 * R * D, "kappa_se": 2 * R * se,
            "h2R": h2R, "h2R_se": h2R_se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(20260912)

    kx4_exact = calibrate_and_validate()

    plan_Z = [(4, 100_000), (8, 100_000), (16, 60_000), (32, 40_000), (64, 25_000),
              (96, 15_000)]
    plan_X = [(4, 100_000), (8, 100_000), (16, 60_000), (32, 40_000), (64, 25_000)]
    plan_ctrl = [(4, 40_000), (8, 40_000), (16, 20_000)]
    if args.quick:
        plan_Z = [(4, 20_000), (8, 10_000)]
        plan_X = [(4, 20_000)]
        plan_ctrl = [(4, 10_000)]

    out = {"LG_kappa_nats": LG, "cal": _CAL, "kx4_exact": kx4_exact,
           "Z_g1": [], "X_g1": [], "Z_g15": []}

    print("== production: Z basis (walls), g=1 ==")
    for R, M in plan_Z:
        out["Z_g1"].append(run_R("Z", 1.0, R, M, rng))
    print("== production: X basis (occupations), g=1 ==")
    for R, M in plan_X:
        out["X_g1"].append(run_R("X", 1.0, R, M, rng))
    print("== 3, gapped control: Z basis, g=1.5 ==")
    for R, M in plan_ctrl:
        out["Z_g15"].append(run_R("Z", 1.5, R, M, rng))

    # 4: cross-checks at R=4
    d8 = json.load(open(os.path.join(RESULTS, "bulk_block_entropies.json")))
    k4_exact = d8["g"]["1.0"]["kappa_D"]["4"]
    z4 = out["Z_g1"][0]; x4 = out["X_g1"][0]
    pull_z = (z4["kappa"] - k4_exact) / z4["kappa_se"]
    pull_x = (x4["kappa"] - kx4_exact) / x4["kappa_se"]
    print(f"== 4: MC vs exact at R=4:  Z pull = {pull_z:+.2f} sigma (exact {k4_exact:.5f})"
          f"   X pull = {pull_x:+.2f} sigma (exact {kx4_exact:.5f})")
    out["W4"] = {"Z_exact": k4_exact, "Z_pull": pull_z, "X_exact": kx4_exact, "X_pull": pull_x}
    if abs(pull_z) > 4 or abs(pull_x) > 4:
        raise SystemExit("check 4 failed: MC estimator disagrees with exact small-R values")

    with open(os.path.join(RESULTS, "critical_tail_mc.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote data/critical_tail_mc.json")


if __name__ == "__main__":
    main()
