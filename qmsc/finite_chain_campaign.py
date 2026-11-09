"""
Learning campaign on the open 50-site transverse-field Ising chain (Sec. III B, Appendix C 2, Figs. 7 and 8).

Setting: open chain, sigma^z basis, g = 0.5, 1, 1.5, radii R = 0..12, sample sizes 1e2..1e6, 20 seeds.

Route (:mod:`qmsc.open_chain`, validated against exact diagonalization at n <= 16):
  target    exact Born distribution of the open-chain ground state through its free-fermion structure
  cells     exact (pi_{i,c}, theta_{i,c}) of every site and radius from Pfaffian / WHT window blocks
  bias      B(R) = sum_i [H(X_i | C_{i,R}) - H(X_i | X_<i)], with H(X_i | X_<i) = H(X_i | C_{i,W}) - T_i,
            W = 12, where the tail T_i = E_ctx KL(p(x_i | x_<i) || p(x_i | C_{i,W})) is a Monte Carlo
            average of exact conditional divergences over a separate sample of contexts (T_i = 0
            exactly for i <= W + 1).  Reported with its standard error.
  samples   N i.i.d. complete configurations per seed by exact sequential conditioning, nested
            prefixes over the sample-size grid, the same samples reused across radii
  learner   site-dependent histogram conditionals with add-1/2 smoothing, unclipped (the paper's rule);
            the number of cells where a floor of eps = 1e-4 would have bound is recorded
  excess    realized V_hat(R, N) = sum_{i,c} pi_{i,c} KL(theta_{i,c} || theta_hat_{i,c}) with the exact
            cells; exact mean V(R, N) of Proposition 1 with the same cells (exact_V_vec)

Usage:
    python scripts/run_finite_chain_campaign.py --validate16        # n = 16, 3 seeds, against exact diagonalization
    python scripts/run_finite_chain_campaign.py --g 0.5 [--seeds 20]  # one coupling, writes data/finite_chain_campaign_n50_g0.5.json
"""
import argparse
import json
import os
import time

import numpy as np

from qmsc import estimation_cost as D4
from qmsc import open_chain as F
from qmsc.bulk_campaign import exact_V_vec
from qmsc.paths import DATA_DIR, ROOT

N_GRID = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
BASE_SEED = 20261013
TAIL_SEED = 777
EPS16 = 1e-4


def hb(t):
    t = np.clip(t, 1e-300, 1 - 1e-16)
    return -(t * np.log(t) + (1 - t) * np.log(1 - t))


def kl_bern(p, q):
    p = np.clip(p, 1e-300, 1 - 1e-16); q = np.clip(q, 1e-300, 1 - 1e-16)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def site_cells(blocks, n, Rmax):
    """cells[R][i-1] = (pi, theta) for every site and R = 0..Rmax."""
    return {R: [F.cells_site(blocks, i, R) for i in range(1, n + 1)] for R in range(Rmax + 1)}


def window_pred(cells, n, W, confs):
    """theta_{i, C_{i,W}}(x) for every configuration and site: the exact window conditional p(x_i = 1 | C_{i,W})."""
    N = len(confs); out = np.empty((N, n))
    for i in range(1, n + 1):
        w = min(W, i - 1)
        c = (confs >> (n - i + 1)) & ((1 << w) - 1)
        out[:, i - 1] = cells[W][i - 1][1][c]
    return out


def realized_excess_grid(confs, n, R, cells_R, N_list):
    """V_hat(R, N), visited cells and would-be-clipped cells for every N in N_list (nested prefixes of
    `confs`), with the unclipped add-1/2 rule and the exact cells. Keys are built once per site."""
    V = np.zeros(len(N_list)); vis = np.zeros(len(N_list)); clipped = np.zeros(len(N_list))
    for i in range(1, n + 1):
        w = min(R, i - 1)
        key = (((confs >> (n - i + 1)) & ((1 << w) - 1)) << 1) | ((confs >> (n - i)) & 1)
        pi, theta = cells_R[i - 1]
        for iN, N in enumerate(N_list):
            cnt = np.bincount(key[:N], minlength=1 << (w + 1)).reshape(-1, 2)
            m = cnt.sum(1); k = cnt[:, 1]
            th = (k + 0.5) / (m + 1.0)
            V[iN] += float(np.sum(pi * kl_bern(theta, th)))
            vis[iN] += int(np.sum(m > 0))
            clipped[iN] += int(np.sum((th < EPS16) | (th > 1 - EPS16)))
    return V, vis, clipped


def run(n, g, seeds, Rmax, W, nmax, tail_samples, out_path, H, validate16=False):
    t0 = time.time()
    ckpt = out_path.replace(".json", "_ckpt.npz")
    F.calibrate()
    M = F.open_chain_M(n, g)
    blocks = F.site_blocks(M, n, W)
    cells = site_cells(blocks, n, max(Rmax, W))      # radii up to Rmax for the learner, plus the W window for the tail
    Hw = F.cond_entropy_windows(blocks, n, W)                       # (n, W+1), w capped at i-1
    print(f"[g={g}] cells and window entropies [{time.time()-t0:.0f}s]", flush=True)
    smp = F.Sampler(M, n)
    Ng = np.array(N_GRID, float)
    base = None; tail_only = None
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        if int(z["n"]) == n and float(z["g"]) == g and int(z["Rmax"]) == Rmax and int(z["W"]) == W:
            if int(z["seeds_done"]) >= 0 and "Vex" in z.files:
                base = z
                print(f"[g={g}] resuming from checkpoint: {int(z['seeds_done'])} seeds done", flush=True)
            else:
                tail_only = z
                print(f"[g={g}] resuming from the tail checkpoint", flush=True)
    if base is None and tail_only is not None:
        T = tail_only["T"]; T_se = tail_only["T_se"]; B = tail_only["B"]; B_se = float(tail_only["B_se"])
    if base is None and tail_only is None:
        # --- tail T_i (exact conditionals, MC over contexts), separate seed, accumulated in batches (memory)
        rng_t = np.random.default_rng(TAIL_SEED + int(round(10 * g)))
        ksum = np.zeros(n); ksq = np.zeros(n); done = 0
        while done < tail_samples:
            B = min(100_000, tail_samples - done)
            xs, cond = smp.sample(B, rng_t, return_cond=True)
            pw = window_pred(cells, n, W, xs)
            x_bits = ((xs[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1)
            p_full = np.where(x_bits == 1, cond, 1.0 - cond)
            kl = kl_bern(p_full, pw)
            ksum += kl.sum(0); ksq += (kl ** 2).sum(0); done += B
            del xs, cond, pw, x_bits, p_full, kl
        T = ksum / tail_samples
        T_se = np.sqrt(np.maximum(ksq / tail_samples - T ** 2, 0.0) / tail_samples)
        T[: W + 1] = 0.0; T_se[: W + 1] = 0.0
        Hfull = Hw[np.arange(n), np.minimum(np.arange(n), W)] - T
        B = np.array([float(np.sum(Hw[:, R] - Hfull)) for R in range(Rmax + 1)])
        B_se = float(np.sqrt(np.sum(T_se ** 2)))
        print(f"[g={g}] tail from {tail_samples} contexts: sum_i T_i = {T.sum():.3e} (SE {B_se:.1e}); B(R) = " + " ".join(f"{b:.3e}" for b in B) + f" [{time.time()-t0:.0f}s]", flush=True)
        np.savez(ckpt, n=n, g=g, Rmax=Rmax, W=W, T=T, T_se=T_se, B=B, B_se=B_se, seeds_done=-1)
    if base is None:
        # --- exact V(R, N), d_occ, d_R
        Vex = np.zeros((len(N_GRID), Rmax + 1)); docc = np.zeros_like(Vex)
        for R in range(Rmax + 1):
            for iN, N in enumerate(N_GRID):
                v = 0.0; d = 0.0
                for i in range(n):
                    pi, th = cells[R][i]
                    vv, _ = exact_V_vec(N, pi, th, H); v += vv
                    keep = pi > 0; d += float(np.sum(1.0 - (1.0 - pi[keep]) ** N))
                Vex[iN, R] = v; docc[iN, R] = d
        print(f"[g={g}] exact V on the grid [{time.time()-t0:.0f}s]", flush=True)
        Vh = np.full((seeds, len(N_GRID), Rmax + 1), np.nan); vis = np.zeros_like(Vh); clip = np.zeros_like(Vh)
        seeds_done = 0
        np.savez(ckpt, n=n, g=g, Rmax=Rmax, W=W, T=T, T_se=T_se, B=B, B_se=B_se, Vex=Vex, docc=docc, Vh=Vh, vis=vis, clip=clip, seeds_done=seeds_done)
    else:
        T = base["T"]; T_se = base["T_se"]; B = base["B"]; B_se = float(base["B_se"]); Vex = base["Vex"]; docc = base["docc"]
        Vh = base["Vh"]; vis = base["vis"]; clip = base["clip"]; seeds_done = int(base["seeds_done"])
        if Vh.shape[0] < seeds:
            pad = np.full((seeds - Vh.shape[0],) + Vh.shape[1:], np.nan)
            Vh = np.concatenate([Vh, pad]); vis = np.concatenate([vis, np.zeros_like(pad)]); clip = np.concatenate([clip, np.zeros_like(pad)])
    dR = np.array([sum(2 ** min(R, i - 1) for i in range(1, n + 1)) for R in range(Rmax + 1)], float)
    # --- campaign, one checkpoint per seed
    for s in range(seeds_done, seeds):
        rng = np.random.default_rng(BASE_SEED + 1000 * s + int(round(10 * g)))
        confs = smp.sample(nmax, rng)
        Nl = [N for N in N_GRID if N <= nmax]
        for R in range(Rmax + 1):
            v, vv, cc = realized_excess_grid(confs, n, R, cells[R], Nl)
            Vh[s, :len(Nl), R] = v; vis[s, :len(Nl), R] = vv; clip[s, :len(Nl), R] = cc
        seeds_done = s + 1
        np.savez(ckpt, n=n, g=g, Rmax=Rmax, W=W, T=T, T_se=T_se, B=B, B_se=B_se, Vex=Vex, docc=docc, Vh=Vh, vis=vis, clip=clip, seeds_done=seeds_done)
        print(f"[g={g}] seed {s+1}/{seeds} done [{time.time()-t0:.0f}s]", flush=True)
    with np.errstate(all="ignore"):
        Vm = np.nanmean(Vh, 0)
    tot_m = B[None, :] + Vm; tot_x = B[None, :] + Vex; tot_r = B[None, :] + dR[None, :] / (2.0 * Ng[:, None])
    R_meas = np.where(np.isnan(tot_m).all(1), -1, np.argmin(np.where(np.isnan(tot_m), np.inf, tot_m), 1))
    R_ex = np.argmin(tot_x, 1); R_reg = np.argmin(tot_r, 1)
    for iN, N in enumerate(N_GRID):
        if N > nmax:
            continue
        print(f"    N={N:>7d}: measured R*={R_meas[iN]:2d} F*={tot_m[iN].min():.4e}   exact R*={R_ex[iN]:2d} F*={tot_x[iN].min():.4e}   parameter-count R*={R_reg[iN]:2d}")
    out = {"n": n, "g": g, "R_grid": list(range(Rmax + 1)), "N_grid": N_GRID, "seeds": seeds, "nmax": nmax, "W": W,
           "tail_samples": tail_samples, "base_seed": BASE_SEED, "tail_seed": TAIL_SEED, "learner": "site-dependent add-1/2 tables, unclipped",
           "H_window": Hw.tolist(), "T_tail": T.tolist(), "T_tail_se": T_se.tolist(), "B_ref": B.tolist(), "B_se": B_se,
           "d_R": dR.tolist(), "V_exact": Vex.tolist(), "d_occ": docc.tolist(),
           "V_hat": Vh.tolist(), "cells_visited": vis.tolist(), "clip_would_bind": clip.tolist(),
           "R_star_measured": R_meas.tolist(), "F_star_measured": [float(np.nanmin(row)) if not np.isnan(row).all() else None for row in tot_m],
           "R_star_exact": R_ex.tolist(), "F_star_exact": tot_x.min(1).tolist(), "R_star_regular": R_reg.tolist(),
           "wall_time_s": time.time() - t0}
    if validate16:
        # exact-diagonalization references computed on the fly: B(R) by the CMI route and the exact
        # estimation cost of Proposition 1 on the ED cells, then the seed mean of V_hat against V
        from qmsc import tfim_chain as TC
        p = TC.born_distribution(TC.ground_state(n, g).psi, n)
        Htab = TC.window_entropy_table(p)
        Bref = np.array([TC.bias_cmi(p, R, Htab)[0] for R in range(Rmax + 1)])
        Lx = np.array([[D4.exact_L(p, R, int(N), float(Bref[R]), H)["L_exact"] for R in range(Rmax + 1)] for N in N_GRID])
        eB = float(np.abs(B - Bref).max())
        eV = float(np.abs(Vex / Lx - 1).max())
        ok = np.array([N <= nmax for N in N_GRID])
        se = np.std(Vh[:, ok], 0, ddof=1) / np.sqrt(seeds)
        z = np.abs(Vm[ok] - Vex[ok]) / se
        print(f"[validate16 g={g}] B vs ED B(R): max |dB| = {eB:.1e} against the tail MC error B_se = {B_se:.1e} (sum_i T_i = {T.sum():.2e}, sites > W+1 = {W+1} carry a tail)")
        print(f"[validate16 g={g}] exact V on free-fermion cells vs on ED cells: max rel err = {eV:.1e}")
        print(f"[validate16 g={g}] seed-mean V_hat vs exact V over {z.size} grid points: {100 * np.mean(z <= 3):.0f} % within 3 SE, "
              f"max |z| = {z.max():.2f} (the SE itself is estimated from {seeds} seeds)")
        out["validate16"] = {"max_dB": eB, "max_rel_dVexact": eV, "frac_within_3se": float(np.mean(z <= 3)), "max_z_Vhat": float(z.max())}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[g={g}] wrote {os.path.basename(out_path)} [{time.time()-t0:.0f}s]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--Rmax", type=int, default=12)
    ap.add_argument("--W", type=int, default=12)
    ap.add_argument("--nmax", type=int, default=1_000_000)
    ap.add_argument("--tail", type=int, default=2_000_000)
    ap.add_argument("--validate16", action="store_true")
    a = ap.parse_args()
    H = D4.HTable()
    if a.validate16:
        # n = 16 against exact diagonalization, 3 seeds; the output goes to validation/, not data/
        vdir = os.path.join(ROOT, "validation")
        os.makedirs(vdir, exist_ok=True)
        for g in ([a.g] if a.g is not None else [0.5, 1.0, 1.5]):
            run(16, g, min(a.seeds, 3), 10, 12, a.nmax, min(a.tail, 200_000),
                os.path.join(vdir, f"finite_chain_campaign_validate16_g{g:g}.json"), H, validate16=True)
        return
    gs = [a.g] if a.g is not None else [0.5, 1.0, 1.5]
    for g in gs:
        run(a.n, g, a.seeds, a.Rmax, a.W, a.nmax, a.tail, os.path.join(DATA_DIR, f"finite_chain_campaign_n{a.n}_g{g:g}.json"), H)


if __name__ == "__main__":
    main()
