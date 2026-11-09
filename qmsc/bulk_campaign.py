"""
Learning campaign at a bulk site of the infinite transverse-field Ising chain (Secs. V and VI, Figs. 3 to 5, Appendix C 1).

Target.  For a bulk site, the per-site table of radius R sees the block (R context spins, next spin).  Its
exact statistics are the marginals of the exact bulk block distribution p_l of l = R_MAX + 1 consecutive
sites of the infinite chain (Pfaffian / Walsh-Hadamard route of :mod:`qmsc.bulk_entropy`, deterministic).
The target of the learner is therefore the infinite chain itself: no boundary, no finite n.

Learner.  Add-1/2 (Krichevsky-Trofimov) tables, one per radius, exactly as in the paper
(theta_hat = (k + 1/2)/(m + 1)).  Training data: N i.i.d. blocks drawn by inverse transform from p_l,
nested prefixes for the sample-size grid, S seeds.

Evaluation (exact, no test set).  Realized estimation excess of a fitted table,

    V_hat(R, N) = sum_c pi_c KL(Bern(theta_c) || Bern(theta_hat_c)),

with the exact cells (pi_c, theta_c).  Bias b(R) = h_R - h_inf of the infinite chain: exact h_R from the
block entropies (R <= 19), h_inf as in :mod:`qmsc.bulk_bias` (gapped: h_{lmax-1}; critical: anchored by the
doubling sums of :mod:`qmsc.critical_tail`, h_inf = h_8 - b(8)).  The exact mean excess V(R, N) of
Proposition 1 is evaluated with the same cells (:func:`qmsc.estimation_cost.cell_expectation`), so the
measured optimum argmin_R [b(R) + mean_seeds V_hat] is compared with the prediction
argmin_R [b(R) + V(R, N)], which has no fitted parameter, and with the parameter-count law 2^R/(2N).

Usage:  python scripts/run_bulk_campaign.py [--seeds 20] [--nmax 1000000]
Output: data/bulk_site_campaign.json
"""
import argparse
import json
import os
import time

import numpy as np

from qmsc import estimation_cost as D4
from qmsc import bulk_entropy as D8
from qmsc.paths import DATA_DIR

SIGN = -1
R_MAX = 15
N_GRID_ALL = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, 100000000]
N_GRID = [N for N in N_GRID_ALL if N <= 1000000]
N_DENSE = [int(round(x)) for x in np.logspace(2, 8, 49)]


def bulk_block(g, l):
    """Exact bulk block distribution of l consecutive sites, index bit i (LSB) = site i."""
    t = time.time()
    p = D8.block_probs_from_M(l, D8.bulk_M(l, g, SIGN))
    H, pmin, nrm = D8.shannon(p)
    assert pmin > -1e-12 and nrm < 1e-10, (pmin, nrm)
    p = np.clip(p, 0.0, None); p /= p.sum()
    print(f"  g={g}: exact block l={l}: H={H:.10f} pmin={pmin:.1e} [{time.time()-t:.0f}s]")
    return p, H


def cells(p, l, R):
    """Cells of the radius-R table at the last site of the block: context = sites l−R..l−1 (bits
    l−1−R .. l−2), target = site l (bit l−1). Returns (pi_c, theta_c) indexed by the context code
    whose bit k is the spin k+1 sites before the target (the last spin is bit 0)."""
    x = np.arange(2 ** l)
    target = (x >> (l - 1)) & 1
    ctx = (x >> (l - 1 - R)) & ((1 << R) - 1) if R > 0 else np.zeros_like(x)
    joint = np.zeros((2 ** R, 2))
    np.add.at(joint, (ctx, target), p)
    pi = joint.sum(1)
    theta = np.where(pi > 0, joint[:, 1] / np.maximum(pi, 1e-300), 0.5)
    return pi, theta


def bias_table(g, bulk):
    """b(R) = h_R − h_inf for R = 0..R_MAX from bulk_bias.json conventions."""
    rec = bulk["g"][g]
    if g == "1":
        h = np.array(bulk["h_g1"])                                # exact h_R, R <= 19, from bulk_block_entropies.json
        b8 = bulk["g"]["1"]["b"][bulk["g"]["1"]["R"].index(8)]   # doubling sum at R = 8
        hinf = h[8] - b8
        return h[:R_MAX + 1] - hinf, hinf, "h_inf = h_8 - b_doubling(8)"
    h = np.array(rec["h"])
    hinf = rec["h_inf_proxy"]
    return h[:R_MAX + 1] - hinf, hinf, rec["note"]


def _h_table(H, m, theta):
    """h(m, theta) for integer m >= 1 (array, any shape) and theta (array broadcastable to m)."""
    tf = np.clip(np.minimum(theta, 1 - theta), 1e-7, None)
    lt = np.log(tf)
    it = np.clip(np.searchsorted(H.log_t, lt) - 1, 0, len(H.t_nodes) - 2)
    wt = (lt - H.log_t[it]) / (H.log_t[it + 1] - H.log_t[it])
    lm = np.log(np.maximum(m, 1))
    im = np.clip(np.searchsorted(H.log_m, lm) - 1, 0, len(H.m_nodes) - 2)
    wm = (lm - H.log_m[im]) / (H.log_m[im + 1] - H.log_m[im])
    T = H.table
    it, wt = np.broadcast_to(it, m.shape), np.broadcast_to(wt, m.shape)
    return ((1 - wm) * ((1 - wt) * T[im, it] + wt * T[im, it + 1])
            + wm * ((1 - wt) * T[im + 1, it] + wt * T[im + 1, it + 1]))


def _window_sum(N, pi, theta, m, H):
    """sum_m Bin(N, pi)(m) h(m, theta) over the integer grid m (shape (K, M)), m = 0 handled exactly."""
    from scipy.stats import binom
    valid = (m >= 0) & (m <= N)
    mm = np.where(valid, m, 0)
    w = np.where(valid, binom.pmf(mm, N, pi[:, None]), 0.0)
    h = _h_table(H, np.maximum(mm, 1), theta[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        h0 = np.where(theta > 0, theta * np.log(theta / 0.5), 0.0) + np.where(theta < 1, (1 - theta) * np.log((1 - theta) / 0.5), 0.0)
    h = np.where(mm == 0, h0[:, None], h)
    return np.sum(w * h, axis=1)


def exact_V_vec(N, pi, theta, H):
    """Vectorized Prop. 1 per-site excess sum_c pi_c E_m h(m, theta_c), m ~ Bin(N, pi_c), with the same
    branches as estimation_cost.cell_expectation: an exact window sum for N pi_c < 3000 (the window
    m = 0..256 for N pi_c < 100, otherwise round(N pi_c) +/- 10 sigma + 10), the 1/m expansion for
    N pi_c >= 3000 and theta(1-theta) >= 0.01, and the per-cell quadrature otherwise.
    Returns (V, n_quadrature_cells)."""
    pi = np.clip(np.asarray(pi, float), 0.0, 1.0); theta = np.clip(np.asarray(theta, float), 0.0, 1.0)
    keep = pi > 0
    pi = pi[keep]; theta = theta[keep]
    mu = N * pi
    V = 0.0
    A1 = mu < 100.0
    if A1.any():
        m = np.broadcast_to(np.arange(0, int(min(N, 256)) + 1)[None, :], (int(A1.sum()), int(min(N, 256)) + 1))
        for lo in range(0, int(A1.sum()), 4096):
            sl = slice(lo, lo + 4096)
            V += float(np.sum(pi[A1][sl] * _window_sum(N, pi[A1][sl], theta[A1][sl], m[sl], H)))
    A2 = (~A1) & (mu < 3000.0)
    if A2.any():
        W = int(np.ceil(10 * np.sqrt(3000.0) + 10))
        off = np.arange(-W, W + 1)
        m0 = np.round(mu[A2]).astype(int)
        for lo in range(0, int(A2.sum()), 1024):
            sl = slice(lo, lo + 1024)
            m = m0[sl][:, None] + off[None, :]
            V += float(np.sum(pi[A2][sl] * _window_sum(N, pi[A2][sl], theta[A2][sl], m, H)))
    sB = theta * (1 - theta)
    B = (mu >= 3000.0) & (sB >= D4.S_ASYM)
    if B.any():
        muB = mu[B]; pB = pi[B]
        inv_m = (1.0 / muB) * (1.0 + (1 - pB) / muB)
        inv_m2 = (1.0 / muB ** 2) * (1.0 + 3 * (1 - pB) / muB)
        c2 = 1.0 / (24.0 * sB[B]) - 5.0 / 12.0
        V += float(np.sum(pB * (0.5 * inv_m + c2 * inv_m2)))
    C = (mu >= 3000.0) & (~B)
    nC = int(C.sum())
    for pi_, th_ in zip(pi[C], theta[C]):
        V += pi_ * D4.cell_expectation(N, float(pi_), float(th_), H)[0]
    return V, nC


def kt_excess(ctx_codes, targets, N, R, pi, theta):
    """Exact realized excess of the add-1/2 table fitted on the first N samples."""
    K = 2 ** R
    c = ctx_codes[:N]; t = targets[:N]
    m = np.bincount(c, minlength=K).astype(float)
    k = np.bincount(c, weights=t, minlength=K)
    th = (k + 0.5) / (m + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(theta > 0, theta * np.log(theta / th), 0.0) + \
             np.where(theta < 1, (1 - theta) * np.log((1 - theta) / (1 - th)), 0.0)
    return float(np.sum(pi * kl)), float(np.sum(m > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--nmax", type=int, default=1_000_000)
    ap.add_argument("--gs", default="0.5,1,1.5")
    ap.add_argument("--rmax", type=int, default=15, help="largest context radius (block of rmax + 1 sites)")
    ap.add_argument("--out", default="bulk_site_campaign.json")
    args = ap.parse_args()
    global R_MAX, N_GRID
    R_MAX = args.rmax
    N_GRID = [N for N in N_GRID_ALL if N <= args.nmax]      # nested prefixes of one draw per seed
    l = R_MAX + 1
    bulk = json.load(open(os.path.join(DATA_DIR, "bulk_bias.json")))
    d8 = json.load(open(os.path.join(DATA_DIR, "bulk_block_entropies.json")))
    bulk["h_g1"] = d8["g"]["1.0"]["h"]
    H = D4.HTable()
    out = {"R_MAX": R_MAX, "N_grid": N_GRID, "N_dense": N_DENSE, "seeds": args.seeds, "nmax": args.nmax,
           "learner": "per-site add-1/2 tables at a bulk site of the infinite chain, exact bulk block cells", "g": {}}
    for g in args.gs.split(","):
        gf = float(g)
        p, Hl = bulk_block(gf, l)
        b, hinf, note = bias_table(g, bulk)
        print(f"  g={g}: b(R) = " + " ".join(f"{x:.2e}" for x in b) + f"   ({note})")
        cell = {R: cells(p, l, R) for R in range(R_MAX + 1)}
        # consistency: the l-block entropy must match the stored block entropies
        H_ref = (bulk["H_g05"][l] if g == "0.5" else float(np.sum(d8["g"][g if g != "1" else "1.0"]["h"][:l])))
        print(f"  g={g}: H_{l} block = {Hl:.10f}, stored {H_ref:.10f}, diff {Hl-H_ref:.1e}")
        # sampling: N i.i.d. blocks per seed by inverse transform
        cdf = np.cumsum(p); cdf[-1] = 1.0
        Vhat = np.zeros((args.seeds, len(N_GRID), R_MAX + 1)); visited = np.zeros_like(Vhat)
        t = time.time()
        for s in range(args.seeds):
            rng = np.random.default_rng(20261009 + 1000 * s + int(round(gf * 10)))
            draws = np.empty(args.nmax, dtype=np.int32)
            for a in range(0, args.nmax, 10_000_000):                      # chunked: 1e8 draws in ~1 GB
                u = rng.random(min(10_000_000, args.nmax - a))
                draws[a:a + len(u)] = np.minimum(np.searchsorted(cdf, u, side="right"), 2 ** l - 1)
            targets = ((draws >> (l - 1)) & 1).astype(float)
            for R in range(R_MAX + 1):
                codes = ((draws >> (l - 1 - R)) & ((1 << R) - 1)) if R > 0 else np.zeros_like(draws)
                pi, th = cell[R]
                for iN, N in enumerate(N_GRID):
                    if N > args.nmax:
                        Vhat[s, iN, R] = np.nan; continue
                    Vhat[s, iN, R], visited[s, iN, R] = kt_excess(codes, targets, N, R, pi, th)
            print(f"    g={g}: seed {s + 1}/{args.seeds} done [{time.time()-t:.0f}s]", flush=True)
        print(f"  g={g}: {args.seeds} seeds fitted and evaluated exactly [{time.time()-t:.0f}s]")
        # exact mean excess (Prop. 1) on the campaign grid and a dense grid
        t = time.time()
        Vex = np.zeros((len(N_GRID), R_MAX + 1)); Vex_dense = np.zeros((len(N_DENSE), R_MAX + 1))
        docc = np.zeros((len(N_GRID), R_MAX + 1))
        nquad = 0
        for R in range(R_MAX + 1):
            pi, th = cell[R]
            keep = pi > 0
            for iN, N in enumerate(N_GRID):
                Vex[iN, R], nq = exact_V_vec(N, pi, th, H); nquad += nq
                docc[iN, R] = float(np.sum(1.0 - (1.0 - pi[keep]) ** N))
            for iN, N in enumerate(N_DENSE):
                Vex_dense[iN, R], nq = exact_V_vec(N, pi, th, H); nquad += nq
        # spot check of the vectorized evaluation against the per-cell routine (R = 6, N = 1e4 and 1e6)
        for N in (300, 10000, 1000000):
            pi, th = cell[6]; keep = pi > 0
            ref = sum(pi_ * D4.cell_expectation(N, float(pi_), float(th_), H)[0] for pi_, th_ in zip(pi[keep], th[keep]))
            vec, _ = exact_V_vec(N, pi, th, H)
            print(f"  g={g}: vectorized vs per-cell exact V at R=6, N={N}: {vec:.6e} vs {ref:.6e} (rel {abs(vec-ref)/ref:.1e})")
            assert abs(vec - ref) <= 5e-4 * ref, "vectorized exact V disagrees"
        print(f"  g={g}: quadrature cells {nquad}")
        print(f"  g={g}: exact V on {len(N_GRID)} + {len(N_DENSE)} sample sizes [{time.time()-t:.0f}s]")
        with np.errstate(all="ignore"):
            Vm = np.nanmean(Vhat, axis=0)
        tot_meas = b[None, :] + Vm; tot_ex = b[None, :] + Vex; tot_reg = b[None, :] + (2.0 ** np.arange(R_MAX + 1))[None, :] / (2.0 * np.array(N_GRID, float)[:, None])
        valid = np.array([N <= args.nmax for N in N_GRID])
        R_meas = np.where(valid, np.nanargmin(np.where(np.isnan(tot_meas), np.inf, tot_meas), axis=1), -1)
        R_ex = np.argmin(tot_ex, axis=1); R_reg = np.argmin(tot_reg, axis=1)
        for iN, N in enumerate(N_GRID):
            if not valid[iN]:
                continue
            print(f"    N={N:>7d}: measured R*={R_meas[iN]:2d} E*/site={np.nanmin(tot_meas[iN]):.3e}   exact R*={R_ex[iN]:2d} E*={tot_ex[iN].min():.3e}   parameter-count R*={R_reg[iN]:2d}")
        out["g"][g] = {"block_entropy": Hl, "b": [float(x) for x in b], "h_inf": float(hinf), "bias_note": note,
                       "V_hat": Vhat.tolist(), "visited": visited.tolist(), "V_exact": Vex.tolist(),
                       "V_exact_dense": Vex_dense.tolist(), "d_occ": docc.tolist(),
                       "R_star_measured": R_meas.tolist(), "E_star_measured": [float(np.nanmin(tot_meas[i])) if valid[i] else None for i in range(len(N_GRID))],
                       "R_star_exact": R_ex.tolist(), "E_star_exact": [float(tot_ex[i].min()) for i in range(len(N_GRID))],
                       "R_star_regular": R_reg.tolist(),
                       "cells_pi_theta": {str(R): [cell[R][0].tolist(), cell[R][1].tolist()] for R in range(0, min(R_MAX, 8) + 1)}}
    with open(os.path.join(DATA_DIR, args.out), "w") as f:
        json.dump(out, f)
    print(f"wrote data/{args.out}")


if __name__ == "__main__":
    main()
