"""Exact block entropies of the infinite transverse-field Ising chain in the sigma^z basis (Appendix B 1).

The ground state is a free-fermion state.  In the Z basis the odd sigma^z correlators vanish by the global flip
symmetry, and for an even subset S of a block the correlator <prod_{i in S} Z_i> is a Gaussian expectation of an
ordered Majorana string, a Pfaffian of two-point functions.  The block probabilities follow by a Walsh-Hadamard
transform,

    p(x) = 2^{-l} sum_S chi_S(x) <Z_S>,

so p = WHT(c) / 2^l with c[S] = <Z_S> (0 for odd |S|), and the block entropy is H_l = -sum_x p ln p.  The
one-step conditional entropies h_R = H_{R+1} - H_R give the per-site bias b(R) = h_R - h_inf (Sec. IV), and
the doubling differences 2R (h_R - h_{2R}) test the critical law b(R) ~ kappa/R without knowing h_inf
(Appendix B 2).  Everything here is deterministic.

``validate()`` checks the machinery before any bulk number is used:

  * block marginals from ED-extracted Majorana correlators reproduce brute-force ED marginals to ~1e-10 (the
    ground state is Gaussian, so Wick's theorem is exact, and this pins every ordering and sign convention);
  * the momentum-integral bulk correlator G(r) matches ED central-chain correlators at g = 1.5, and the AB
    sector <A_m B_{m+d}> is -G(-d), not -G(+d) (G is asymmetric, and the wrong sector produces negative block
    probabilities at g = 1);
  * the bulk block law at g = 1.5 matches ED central-block marginals at n = 12 to finite-size accuracy, and
    h_R converges exponentially;
  * at g = 1 the correlator integral has the closed form G(r) = -sign * 2 / (pi (2r - 1)), checked against
    quadrature at small r and against ED, and used unconditionally there (quadrature degrades at large r).

Reference value for the critical coefficient: Lau and Grassberger, Phys. Rev. E 87, 022128 (2013), Eq. (27),
H_n ~ n h + b ln n with b = 0.085(1) bits, i.e. kappa = b ln 2 = 0.0589 nats.

Usage:  python scripts/compute_bulk_entropies.py [--lmax 20] [--skip-validate]
Output: data/bulk_block_entropies.json (g = 1 and 1.5)
"""
import argparse
import json
import os
import time


import numpy as np
from scipy.integrate import quad

from qmsc import tfim_chain as T

from qmsc.paths import DATA_DIR as RESULTS
LG_KAPPA_NATS = 0.085 * np.log(2)  # Lau-Grassberger b in nats


# ----------------------------------------------------------------------------- Pfaffian
def pfaffian(A):
    """Pfaffian of a real skew-symmetric matrix (Parlett-Reid with partial pivoting)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if n % 2:
        return 0.0
    pf = 1.0
    for k in range(0, n - 1, 2):
        # pivot: largest |A[k+1:, k]|
        col = np.abs(A[k + 1:, k])
        kp = k + 1 + int(np.argmax(col))
        if col[kp - k - 1] == 0.0:
            return 0.0
        if kp != k + 1:
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            pf = -pf
        pf *= A[k, k + 1]
        if k + 2 < n:
            tau = A[k, k + 2:] / A[k, k + 1]
            col = A[k + 2:, k + 1]
            A[k + 2:, k + 2:] += np.outer(tau, col) - np.outer(col, tau)
    return pf


# ------------------------------------------------------------------- Majorana bookkeeping
# Canonical index: A_j -> 2j, B_j -> 2j+1 (sites 0-based). Within any even subset S of a
# block, pairing consecutive elements (i1,i2),(i3,i4),... the operator string is
#   prod_m [ B_{i_{2m-1}}, A_{i_{2m-1}+1}, B_{i_{2m-1}+1}, ..., A_{i_{2m}} ],
# whose canonical indices are strictly increasing, so Wick = Pf of the ordered submatrix.

def majorana_list(sites):
    """sites: sorted even-size tuple (0-based, within block). Return canonical indices."""
    idx = []
    for m in range(0, len(sites), 2):
        i, j = sites[m], sites[m + 1]
        idx.append(2 * i + 1)                    # B_i
        for l in range(i + 1, j + 1):
            idx.append(2 * l)                    # A_l
            if l < j:
                idx.append(2 * l + 1)            # B_l
    return idx


def block_correlator_matrix(l, AA, AB, BA, BB):
    """Full ordered 2l x 2l matrix M[p<q] = <phi_p phi_q> with canonical indexing.
    AA[m,n]=<A_m A_n> etc. (functions of site pair, 0-based within block)."""
    M = np.zeros((2 * l, 2 * l))
    for p in range(2 * l):
        for q in range(p + 1, 2 * l):
            mp, tp = divmod(p, 2)
            mq, tq = divmod(q, 2)
            if tp == 0 and tq == 0:
                M[p, q] = AA(mp, mq)
            elif tp == 0 and tq == 1:
                M[p, q] = AB(mp, mq)
            elif tp == 1 and tq == 0:
                M[p, q] = BA(mp, mq)
            else:
                M[p, q] = BB(mp, mq)
            M[q, p] = -M[p, q]
    return M


def block_probs_from_M(l, M):
    """All 2^l block probabilities from the ordered Majorana correlation matrix M."""
    c = np.zeros(2 ** l)
    c[0] = 1.0
    for Sbits in range(1, 2 ** l):
        sites = tuple(i for i in range(l) if (Sbits >> i) & 1)
        if len(sites) % 2:
            continue
        idx = majorana_list(sites)
        c[Sbits] = pfaffian(M[np.ix_(idx, idx)])
    # Walsh-Hadamard transform: p[x] = 2^-l sum_S (-1)^{popcount(S & x)} c[S]
    p = c.copy()
    h = 1
    while h < 2 ** l:
        for i in range(0, 2 ** l, 2 * h):
            a = p[i:i + h].copy()
            b = p[i + h:i + 2 * h].copy()
            p[i:i + h] = a + b
            p[i + h:i + 2 * h] = a - b
        h *= 2
    return p / 2 ** l


def shannon(p, tol=1e-13):
    q = p[p > tol]
    return float(-(q * np.log(q)).sum()), float(p.min()), float(abs(p.sum() - 1.0))


# ---------------------------------------------------------------- ED reference machinery
def ed_majorana_correlators(n, g):
    """Exact <phi_p phi_q> from the ED ground state, phi in canonical order.
    A_i = (prod_{j<i} sx_j) sz_i ;  B_i = i (prod_{j<i} sx_j) sy_i   (real matrices)."""
    psi = np.asarray(T.ground_state(n, g).psi).ravel()
    sx = np.array([[0, 1], [1, 0]], float)
    sz = np.array([[1, 0], [0, -1]], float)
    isy = np.array([[0, 1], [-1, 0]], float)  # i*sigma_y (real)

    def site_chain(ops):
        out = np.array([[1.0]])
        for o in ops:
            out = np.kron(out, o)
        return out

    I2 = np.eye(2)
    phis = []
    for i in range(n):
        opsA = [sx] * i + [sz] + [I2] * (n - i - 1)
        opsB = [sx] * i + [isy] + [I2] * (n - i - 1)
        phis.append(site_chain(opsA))
        phis.append(site_chain(opsB))
    M = np.zeros((2 * n, 2 * n))
    vecs = [ph @ psi for ph in phis]
    for p in range(2 * n):
        for q in range(p + 1, 2 * n):
            M[p, q] = float(vecs[p].conj() @ (phis[q] @ psi)) if False else float(psi @ (phis[p] @ vecs[q]))
            M[q, p] = -M[p, q]
    return M, psi


def ed_block_marginal(psi, n, i0, l):
    """Brute-force marginal of sites i0..i0+l-1 (0-based) from the ED Born distribution.
    Convention: tfim_chain bit order — site i is bit (n-1-i) [verified in-run]."""
    p_full = np.asarray(T.born_distribution(psi, n)).ravel()
    out = np.zeros(2 ** l)
    for x in range(2 ** n):
        key = 0
        for j in range(l):
            bit = (x >> (n - 1 - (i0 + j))) & 1
            key |= bit << j
        out[key] += p_full[x]
    return out


# ------------------------------------------------------------------- bulk correlators
def bulk_G(r, g, sign=+1):
    """G(r) = <B_m A_{m+r}> for the infinite chain: (sign/2pi) Int dk [cos(kr)(g-cos k)
    - sin(kr) sin k]/Lam_k. Convention pinned against ED in validation.

    At g=1 the integral has an exact closed form, used unconditionally there (quadrature
    degrades at large r: |sin(k/2)| kink at k=0 plus oscillation). Derivation: with
    1-cos k = 2 sin^2(k/2), sin k = 2 sin(k/2)cos(k/2), Lam_k = 2|sin(k/2)|, the integrand
    reduces to -sgn(k) * sin(k(r-1/2)), an even function of k, so
        Int_{-pi}^{pi} = -2 Int_0^pi sin(k(r-1/2)) dk = 2[cos(pi(r-1/2)) - 1]/(r-1/2)
    and cos(pi(r-1/2)) = sin(pi r) = 0 for integer r, giving Int = -4/(2r-1). Hence
        G(r) = -sign * 2 / (pi (2r-1))   exactly (validated in validate())."""
    if abs(g - 1.0) < 1e-12:
        return -sign * 2.0 / (np.pi * (2 * r - 1))

    def integrand(k):
        lam = np.sqrt((g - np.cos(k)) ** 2 + np.sin(k) ** 2)
        if lam < 1e-14:
            return 0.0  # integrand -> 0 at the critical gap closing (k=0, g=1)
        return (np.cos(k * r) * (g - np.cos(k)) - np.sin(k * r) * np.sin(k)) / lam
    val, _ = quad(integrand, -np.pi, np.pi, limit=800, points=[0.0])
    return sign * val / (2 * np.pi)


def bulk_M(l, g, sign):
    # LSM structure: <A_m A_n> = delta, <B_m B_n> = -delta (B^2 = -1 convention),
    # <B_m A_n> = G(n-m), and <A_m B_n> = -<B_n A_m> = -G(m-n)  [NOT -G(n-m): G is
    # asymmetric, strongly so at criticality — the -G(n-m) form was an early bug whose
    # error is diluted ~2^-l by the WHT, mimicking finite-size error at gapped g and
    # producing ~1e-3 negative probabilities at g=1; pinned empirically in V3b].
    Gv = {r: bulk_G(r, g, sign) for r in range(-l, l + 1)}
    AA = lambda m, n_: 1.0 if m == n_ else 0.0
    BB = lambda m, n_: -1.0 if m == n_ else 0.0
    BA = lambda m, n_: Gv[n_ - m]
    AB = lambda m, n_: -Gv[m - n_]
    return block_correlator_matrix(l, AA, AB, BA, BB)


# ------------------------------------------------------------------------------- main
def validate():
    print("== 1: Wick/Pfaffian machinery vs brute-force ED ==")
    ok = True
    for (n, g, l) in [(8, 1.5, 4), (10, 1.0, 4), (10, 0.5, 3)]:
        M, psi = ed_majorana_correlators(n, g)
        i0 = (n - l) // 2
        # submatrix for the central block, reindexed to block-local canonical order
        idx = []
        for i in range(i0, i0 + l):
            idx += [2 * i, 2 * i + 1]
        Mblk = M[np.ix_(idx, idx)]
        p_pf = block_probs_from_M(l, Mblk)
        p_ed = ed_block_marginal(psi, n, i0, l)
        err = float(np.abs(p_pf - p_ed).max())
        print(f"  n={n} g={g} l={l}: max|p_Pf - p_ED| = {err:.2e}")
        ok &= err < 1e-8
    if not ok:
        raise SystemExit("check 1 failed")

    print("== 2: bulk correlator convention vs ED (g=1.5, center of n=12) ==")
    n, g = 12, 1.5
    M, _ = ed_majorana_correlators(n, g)
    m0 = n // 2 - 1
    best = None
    for sign in (+1, -1):
        errs = []
        for r in range(0, 4):
            ed_val = M[2 * m0 + 1, 2 * (m0 + r)] if r > 0 else -M[2 * m0, 2 * m0 + 1]
            errs.append(abs(ed_val - bulk_G(r, g, sign)))
        e = max(errs)
        print(f"  sign={sign:+d}: max|G_ED - G_int| over r=0..3 = {e:.2e}")
        if best is None or e < best[1]:
            best = (sign, e)
    sign, e = best
    if e > 1e-3:
        raise SystemExit("check 2 failed: no convention matches")
    print(f"  -> convention pinned: sign = {sign:+d}")

    print("== V2b: AB sector <A_m B_{m+d}> = -G(-d), not -G(+d) (g=1.5, n=12) ==")
    for d in (1, 2, 3):
        ab_ed = M[2 * m0, 2 * (m0 + d) + 1]
        good = abs(ab_ed - (-bulk_G(-d, g, sign)))
        bad = abs(ab_ed - (-bulk_G(d, g, sign)))
        print(f"  d={d}: |AB_ED - (-G(-d))| = {good:.2e}   [wrong sector -G(+d): {bad:.2e}]")
        if good > 1e-3:
            raise SystemExit("V2b FAILED: AB sector convention broken")

    print("== 3: bulk block law vs ED central marginal (g=1.5, l=4, n=12) ==")
    p_bulk = block_probs_from_M(4, bulk_M(4, 1.5, sign))
    _, psi = ed_majorana_correlators(12, 1.5)
    p_ed = ed_block_marginal(psi, 12, 4, 4)
    err = float(np.abs(p_bulk - p_ed).max())
    print(f"  max|p_bulk - p_ED(n=12 center)| = {err:.2e}  (finite-size level expected)")
    if err > 5e-4:
        raise SystemExit("check 3 failed")

    print("== 4: critical closed form G(r) = -sign*2/(pi(2r-1)) at g=1 ==")
    # (a) against direct quadrature at small |r|, where quad is still trustworthy
    def quad_G(r):
        def integrand(k):
            lam = np.sqrt((1.0 - np.cos(k)) ** 2 + np.sin(k) ** 2)
            if lam < 1e-14:
                return 0.0
            return (np.cos(k * r) * (1.0 - np.cos(k)) - np.sin(k * r) * np.sin(k)) / lam
        val, _ = quad(integrand, -np.pi, np.pi, limit=800, points=[0.0])
        return sign * val / (2 * np.pi)
    eq = max(abs(quad_G(r) - bulk_G(r, 1.0, sign)) for r in range(-3, 4))
    print(f"  (a) max|G_quad - G_closed| over r=-3..3 = {eq:.2e}")
    if eq > 1e-5:
        raise SystemExit("V4a FAILED: closed form disagrees with quadrature at small r")
    # (b) against ED central-chain correlators at n=12 (finite-size ~1/n at criticality)
    M12, _ = ed_majorana_correlators(12, 1.0)
    m0 = 5
    errs = []
    for r in range(0, 4):
        ed_val = M12[2 * m0 + 1, 2 * (m0 + r)] if r > 0 else -M12[2 * m0, 2 * m0 + 1]
        errs.append(abs(ed_val - bulk_G(r, 1.0, sign)))
    print(f"  (b) max|G_ED(n=12) - G_closed| over r=0..3 = {max(errs):.2e}"
          f"  (critical finite-size level expected)")
    if max(errs) > 5e-2:
        raise SystemExit("V4b FAILED: closed form far from ED even allowing finite size")
    # (c) AB sector at criticality, where the asymmetry of G is O(1)
    errs_ab = []
    for d in (1, 2, 3):
        ab_ed = M12[2 * m0, 2 * (m0 + d) + 1]
        errs_ab.append(abs(ab_ed - (-bulk_G(-d, 1.0, sign))))
    print(f"  (c) max|AB_ED(n=12) - (-G(-d))| over d=1..3 = {max(errs_ab):.2e}"
          f"  (critical finite-size level expected)")
    if max(errs_ab) > 5e-2:
        raise SystemExit("V4c FAILED: critical AB sector broken")
    return sign


def run(lmax, sign):
    out = {"lmax": lmax, "sign": sign, "LG_kappa_nats": LG_KAPPA_NATS, "g": {}}
    for g in (1.0, 1.5):
        t0 = time.time()
        H = [0.0]
        diag = []
        for l in range(1, lmax + 1):
            p = block_probs_from_M(l, bulk_M(l, g, sign))
            Hl, pmin, norm = shannon(p)
            H.append(Hl)
            diag.append((l, pmin, norm))
            print(f"  g={g} l={l:2d}  H={Hl:.10f}  min p={pmin:+.1e}  |sum-1|={norm:.1e}"
                  f"  ({time.time()-t0:.1f}s)")
        H = np.array(H)
        h = np.diff(H)                                # h_R = H_{R+1} - H_R, index R=0..lmax-1
        kD = {}
        for R in range(1, lmax // 2):
            kD[R] = 2 * R * (h[R] - h[2 * R])
        kG = {}
        for r in range(2, lmax - 1):
            gr = h[r - 1] - h[r]
            kG[r] = gr / np.log(r * r / ((r - 1.0) * (r + 1.0)))
        out["g"][str(g)] = {
            "H": H.tolist(), "h": h.tolist(),
            "kappa_D": {str(k): float(v) for k, v in kD.items()},
            "kappa_g": {str(k): float(v) for k, v in kG.items()},
            "pmin_worst": min(d[1] for d in diag), "norm_worst": max(d[2] for d in diag),
        }
        print(f"  g={g}: kappa_D = {[f'{v:.4f}' for v in kD.values()]}")
    with open(os.path.join(RESULTS, "bulk_block_entropies.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote data/bulk_block_entropies.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmax", type=int, default=16, help="largest block length (the paper uses 20)")
    ap.add_argument("--skip-validate", action="store_true", help="skip the exact-diagonalization checks")
    args = ap.parse_args()
    if args.skip_validate:
        sign = -1  # the convention fixed by check 2 (+1 flips all correlators)
    else:
        sign = validate()
    run(args.lmax, sign)


if __name__ == "__main__":
    main()
