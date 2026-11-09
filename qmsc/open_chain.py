"""
Free-fermion machinery for the open transverse-field Ising chain (Appendices B 1 and C 2 of the paper).

Exact diagonalization is limited to about 20 sites.  The finite-chain campaign at n = 50 instead uses the
free-fermion structure of the open chain, from which every quantity the campaign needs follows exactly:

  * open_chain_M(n, g)      Majorana correlation matrix M[p, q] = <phi_p phi_q> in the conventions of
                            bulk_entropy.ed_majorana_correlators (A_j -> 2j, B_j -> 2j+1, sites 0-based),
                            from the Bogoliubov solution of H = -sum sz sz - g sum sx.  Validated against
                            the exact-diagonalization matrix at small n.
  * window_block(M, lo, l)  exact probabilities of the l-site block starting at site lo (0-based), bit i
                            of the block index <-> site lo + i (LSB = first site, bit 1 = spin down), by the
                            Pfaffian / Walsh-Hadamard route of :mod:`qmsc.bulk_entropy` (odd Z-correlators
                            vanish by the global flip symmetry, which the open-chain ground state has).
  * cells_from_block(...)   (pi_c, theta_c) of every cell of site i for radius R, in the bit convention
                            of histogram_learner.cell_counts / estimation_cost.cells_for (the context packs
                            sites lo..i-1 with site lo as the most significant bit).
  * cond_entropy_windows    H(X_i | C_{i,w}) for w = 0..wmax from one block per site.
  * Sampler                 exact sequential sampling of complete sigma^z configurations through the
                            Gaussian conditioning engine of :mod:`qmsc.critical_tail` (wall outcomes,
                            batched rank-2 updates), returning the configurations as int64 (site 1 = MSB,
                            the convention of the campaign data) and the exact conditional probabilities
                            p(x_i | x_<i) of every site of every sample.

Nothing here is sampled unless Sampler is called.  Running the module

    python -m qmsc.open_chain

executes the validation ladder against exact diagonalization at n = 6, 8, 12 and 16.
"""
import time

import numpy as np

from qmsc import bulk_entropy as D8
from qmsc import tfim_chain as T

# conditioning-engine convention shared with critical_tail.calibrate_and_validate:
#     Sigma' = Sigma + NS (u v^T - v u^T) / (c + DS s)
NS, DS = -1.0, +1.0     # confirmed against the exact Born probabilities at n = 6 in validate()
_CAL_DONE = False


# ------------------------------------------------------------------------------ covariance
def open_chain_M(n, g, sign=None):
    """M[p, q] = <phi_p phi_q> for the open n-site chain, canonical index A_j -> 2j, B_j -> 2j+1.

    Hermitian Majoranas c_{2j} = A_j, c_{2j+1} = -i B_j give sx_j = i c_{2j} c_{2j+1} and
    sz_k sz_{k+1} = i c_{2k+1} c_{2k+2}, so H = (i/4) c^T h c with h_{2j,2j+1} = -2g, h_{2k+1,2k+2} = -2
    (antisymmetric). The ground state fills the negative modes of the Hermitian matrix i h, and
    <c_p c_q> = delta + i Gamma with Gamma = i (I - 2 P_<) real antisymmetric. Then M_AB = -Gamma_AB,
    M_BA = -Gamma_BA, M_AA = M_BB = 0 (checked against ED). `sign` flips Gamma (calibrated once)."""
    if sign is None:
        sign = _SIGN
    h = np.zeros((2 * n, 2 * n))
    for j in range(n):
        h[2 * j, 2 * j + 1] = -2.0 * g; h[2 * j + 1, 2 * j] = 2.0 * g
    for k in range(n - 1):
        h[2 * k + 1, 2 * k + 2] = -2.0; h[2 * k + 2, 2 * k + 1] = 2.0
    A = 1j * h
    w, U = np.linalg.eigh(A)
    Pneg = U[:, w < 0] @ U[:, w < 0].conj().T
    Gamma = 1j * (np.eye(2 * n) - 2.0 * Pneg)
    assert np.abs(Gamma.imag).max() < 1e-10, "Gamma not real"
    Gamma = sign * Gamma.real
    M = -Gamma.copy()
    # AA and BB sectors vanish for the TFIM ground state (checked in validate); enforce exact zeros
    M[0::2, 0::2] = 0.0; M[1::2, 1::2] = 0.0
    return M


_SIGN = +1.0


def calibrate(n=6, g=1.5):
    """Pin the sign of Gamma against the ED correlators (one global flip is the only freedom)."""
    global _SIGN, _CAL_DONE
    Med, _ = D8.ed_majorana_correlators(n, g)
    best = None
    for s in (+1.0, -1.0):
        err = float(np.abs(open_chain_M(n, g, sign=s) - Med).max())
        if best is None or err < best[1]:
            best = (s, err)
    _SIGN, err = best
    if err > 1e-10:
        raise RuntimeError(f"open_chain_M does not reproduce ED at n={n}, g={g}: err {err:.2e}")
    _CAL_DONE = True
    return _SIGN, err


# ------------------------------------------------------------------------------ blocks and cells
def window_block(M, lo, l):
    """Exact probabilities of the block of sites lo..lo+l-1 (0-based), index bit i <-> site lo+i (LSB first)."""
    idx = np.arange(2 * lo, 2 * (lo + l))
    return D8.block_probs_from_M(l, M[np.ix_(idx, idx)])


def block_to_campaign_order(p_block, l):
    """Re-key a block from the bulk_entropy convention (bit i <-> site i of the block, LSB first) to the
    campaign convention (first site = most significant bit): returns an array of shape (2,)*l with axis
    j <-> site j."""
    # bulk_entropy index: x = sum_j bit_j << j  ->  reshape with order='F' gives axis j <-> bit j
    return p_block.reshape((2,) * l, order="F")


def cells_from_block(pb, l):
    """Cells of the LAST site of a block given as (2,)*l array (axis j <-> site j, 0 = up).
    Context = all preceding sites of the block, packed with the first site as the MSB (campaign order).
    Returns (pi, theta) with theta = P(X_last = 1 | context) and pi = P(context)."""
    joint = pb.reshape(2 ** (l - 1), 2)            # C order: first axes are the context sites, MSB first
    pi = joint.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(pi > 0, joint[:, 1] / np.maximum(pi, 1e-300), 0.5)
    return pi, theta


def site_blocks(M, n, wmax):
    """For every site i = 1..n (1-based), the (2,)*l array of its window block with l = min(i, wmax + 1)
    sites ending at site i (axis j <-> site i - l + 1 + j)."""
    out = {}
    for i in range(1, n + 1):
        l = min(i, wmax + 1)
        pb = window_block(M, i - l, l)
        out[i] = block_to_campaign_order(pb, l)
    return out


def cells_site(blocks, i, R):
    """(pi, theta) of site i (1-based) for radius R from the stored window blocks (marginalizing the
    leading sites when R < l - 1)."""
    pb = blocks[i]
    l = pb.ndim
    w = min(R, i - 1)
    if w < l - 1:
        pb = pb.sum(axis=tuple(range(l - 1 - w)))
    return cells_from_block(pb, w + 1)


def hb(t):
    t = np.clip(t, 1e-300, 1 - 1e-16)
    return -(t * np.log(t) + (1 - t) * np.log(1 - t))


def cond_entropy_windows(blocks, n, wmax):
    """H(X_i | C_{i,w}) for i = 1..n and w = 0..wmax (w capped at i-1): array (n, wmax+1)."""
    H = np.zeros((n, wmax + 1))
    for i in range(1, n + 1):
        for w in range(0, wmax + 1):
            pi, th = cells_site(blocks, i, w)
            H[i - 1, w] = float(np.sum(pi * hb(th)))
    return H


# ------------------------------------------------------------------------------ sampler
class SamplerFull:
    """Reference implementation on the full 2n x 2n matrix (slow, used only to cross-check Sampler)."""

    def __init__(self, M, n):
        self.M = np.asarray(M, float); self.n = n
        self.pairs = [(2 * k + 1, 2 * k + 2) for k in range(n - 1)]

    def log_prob(self, confs):
        confs = np.asarray(confs, dtype=np.int64); N = len(confs); n = self.n
        S = np.broadcast_to(self.M, (N, 2 * n, 2 * n)).copy()
        bits = ((confs[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
        logp = np.full(N, np.log(0.5))
        for k, (a, b) in enumerate(self.pairs):
            s = np.where(bits[:, k] == bits[:, k + 1], 1.0, -1.0)
            c = S[:, a, b]
            logp += np.log(np.maximum(0.5 * (1.0 + s * c), 1e-300))
            u = S[:, :, a].copy(); v = S[:, :, b].copy()
            den = c + DS * s
            S += NS * (u[:, :, None] * v[:, None, :] - v[:, :, None] * u[:, None, :]) / den[:, None, None]
            S[:, a, :] = 0.0; S[:, :, a] = 0.0; S[:, b, :] = 0.0; S[:, :, b] = 0.0
        return logp


class Sampler:
    """Exact sequential sampler of complete sigma^z configurations of the open chain.

    Site 1 is up or down with probability 1/2 (global flip symmetry). Walls w_k = sz_k sz_{k+1}
    (k = 1..n-1) are sampled sequentially from the Gaussian state conditioned on the previous walls.
    Because the AA and BB sectors of the correlation matrix vanish, the rank-2 update of the 2n x 2n
    matrix used in :mod:`qmsc.critical_tail` reduces to a rank-1 update of the n x n matrix
    G[j, l] = <A_j B_l>:
        c_k = <B_k A_{k+1}> = -G[k+1, k],   p(w_k = s | past) = (1 + s c_k)/2,
        G'[j, l] = G[j, l] - NS G[j, k] G[k+1, l] / (c_k + DS s)   for the still-unmeasured j >= k+2, l >= k+1,
    and the measured row k+1 and column k drop out. A_1 (index 0) never enters a wall probability.
    p(x_{k+1} | x_<=k) = p(w_k | w_<k), so the exact conditional probability of every site given its
    full past is returned alongside the configurations. Configurations are int64 with site 1 = MSB."""

    def __init__(self, M, n):
        M = np.asarray(M, float); self.n = n
        self.G = M[0::2, 1::2].copy()                               # G[j, l] = <A_j B_l>

    def _run(self, B, rng=None, bits_in=None, want_cond=False):
        n = self.n
        G = np.broadcast_to(self.G, (B, n, n)).copy()
        if bits_in is None:
            x = np.empty((B, n), dtype=np.int8)
            x[:, 0] = (rng.random(B) < 0.5).astype(np.int8)
        else:
            x = bits_in
        cond = np.empty((B, n)) if want_cond else None
        if want_cond:
            cond[:, 0] = 0.5
        for k in range(n - 1):
            c = -G[:, k + 1, k]
            if bits_in is None:
                s = np.where(rng.random(B) < 0.5 * (1.0 + c), 1.0, -1.0)
                x[:, k + 1] = x[:, k] ^ (s < 0).astype(np.int8)
            else:
                s = np.where(x[:, k] == x[:, k + 1], 1.0, -1.0)
            if want_cond:
                cond[:, k + 1] = 0.5 * (1.0 + s * c)
            if k + 2 < n:
                coef = NS / (c + DS * s)
                G[:, k + 2:, k + 1:] -= coef[:, None, None] * G[:, k + 2:, k][:, :, None] * G[:, k + 1, k + 1:][:, None, :]
        return x, cond

    def sample(self, N, rng, batch=8192, return_cond=False):
        n = self.n
        confs = np.empty(N, dtype=np.int64)
        cond = np.empty((N, n)) if return_cond else None
        for lo in range(0, N, batch):
            B = min(batch, N - lo)
            x, cd = self._run(B, rng=rng, want_cond=return_cond)
            packed = np.zeros(B, dtype=np.int64)
            for i in range(n):
                packed |= x[:, i].astype(np.int64) << (n - 1 - i)
            confs[lo:lo + B] = packed
            if return_cond:
                cond[lo:lo + B] = cd
        return (confs, cond) if return_cond else confs

    def log_prob(self, confs, batch=8192):
        """Exact log p(x) of given configurations (int64, site 1 = MSB)."""
        confs = np.asarray(confs, dtype=np.int64); N = len(confs); n = self.n
        out = np.empty(N)
        for lo in range(0, N, batch):
            cf = confs[lo:lo + batch]
            bits = ((cf[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
            _, cond = self._run(len(cf), bits_in=bits, want_cond=True)
            out[lo:lo + batch] = np.log(np.maximum(cond, 1e-300)).sum(1)
        return out


# ------------------------------------------------------------------------------ validation
def validate(verbose=True):
    """Validation ladder against exact diagonalization.

    0. sign of the covariance matrix at n = 6;
    1. open_chain_M against the exact Majorana correlators at n = 8 and 12;
    2. cells and the bias B(R) at n = 16 against the exact-diagonalization routes of
       :mod:`qmsc.estimation_cost` and :mod:`qmsc.tfim_chain`;
    3. the sampler: conditioning-engine signs, exact log-probabilities, and sample pair correlations.
    """
    from qmsc import estimation_cost as D4

    rep = {}
    s, err = calibrate()
    if verbose:
        print(f"check 0  covariance sign calibrated at n=6, g=1.5: sign {s:+.0f}, max |M - M_ED| = {err:.1e}")
    for n, g in ((8, 1.0), (12, 0.5), (12, 1.5), (12, 1.0)):   # the ED Majorana builder is dense, n <= 12
        Med, psi = D8.ed_majorana_correlators(n, g)
        e = float(np.abs(open_chain_M(n, g) - Med).max())
        rep[f"M n={n} g={g}"] = e
        if verbose:
            print(f"check 1  open_chain_M vs ED: n={n} g={g}: max err {e:.1e}")
        assert e < 1e-9, "covariance mismatch"
    # cells and the bias at n = 16 against exact diagonalization
    R_grid = (0, 1, 2, 4, 6, 8, 10)
    for g in (0.5, 1.0, 1.5):
        n = 16
        gs = T.ground_state(n, g); p = T.born_distribution(gs.psi, n)
        M = open_chain_M(n, g)
        blocks = site_blocks(M, n, 15)
        worst = 0.0
        for R in R_grid:
            ed = list(D4.cells_for(p, R))                     # (pi, theta) per site, campaign convention
            k = 0
            for i in range(1, n + 1):
                pi_ff, th_ff = cells_site(blocks, i, R)
                pi_ed, th_ed = ed[k]; k += 1
                worst = max(worst, float(np.abs(pi_ff - pi_ed).max()), float(np.abs((th_ff - th_ed) * (pi_ed > 1e-12)).max()))
        rep[f"cells n=16 g={g}"] = worst
        # bias from the window entropies vs the CMI route of tfim_chain
        Hw = cond_entropy_windows(blocks, n, 15)
        Hfull = Hw[np.arange(n), np.minimum(np.arange(n), 15)]        # w = i - 1 = full past
        B_ff = np.array([float(np.sum(Hw[:, min(R, 15)] - Hfull)) for R in R_grid])
        Htab = T.window_entropy_table(p)
        B_ed = np.array([T.bias_cmi(p, R, Htab)[0] for R in R_grid])
        eB = float(np.abs(B_ff - B_ed).max())
        rep[f"B n=16 g={g}"] = eB
        if verbose:
            print(f"check 2  n=16 g={g}: cells vs ED max err {worst:.1e}; B(R) vs ED max err {eB:.1e}")
        assert worst < 1e-8 and eB < 1e-8    # cells: Pfaffian/WHT roundoff over 2^15 subsets ~1e-9; B to 1e-13
    # engine update signs: the combination that reproduces the exact Born probabilities at n = 6
    global NS, DS
    n, g = 6, 1.5
    gs = T.ground_state(n, g); pfull = T.born_distribution(gs.psi, n).ravel(); M = open_chain_M(n, g)
    best = None
    for ns in (1.0, -1.0):
        for ds in (1.0, -1.0):
            NS, DS = ns, ds
            e = float(np.abs(np.exp(Sampler(M, n).log_prob(np.arange(2 ** n, dtype=np.int64))) - pfull).max())
            if best is None or e < best[1]:
                best = ((ns, ds), e)
    (NS, DS), e = best
    rep["engine signs"] = (NS, DS, e)
    if verbose:
        print(f"check 3a engine update signs (NS, DS) = ({NS:+.0f}, {DS:+.0f}) reproduce ED p(x) at n=6 to {e:.1e}")
    assert e < 1e-12
    # sampler: exact log-probabilities vs ED, and sample marginals / pair correlations vs ED within MC error
    rng = np.random.default_rng(1)
    for n, g in ((8, 1.0), (16, 0.5)):
        gs = T.ground_state(n, g); pfull = T.born_distribution(gs.psi, n).ravel()
        M = open_chain_M(n, g); smp = Sampler(M, n)
        confs = np.arange(2 ** n, dtype=np.int64) if n <= 10 else rng.integers(0, 2 ** n, 3000, dtype=np.int64)
        lp = smp.log_prob(confs)
        e = float(np.abs(np.exp(lp) - pfull[confs]).max())
        rep[f"log_prob n={n} g={g}"] = e
        if n == 16:
            e3 = float(np.abs(SamplerFull(M, n).log_prob(confs[:300]) - lp[:300]).max())
            rep[f"full-vs-reduced n={n}"] = e3
            if verbose:
                print(f"check 3b reduced (n x n) sampler vs full (2n x 2n) engine, n=16: max |dlogp| {e3:.1e}")
            assert e3 < 1e-10
        xs, cond = smp.sample(200_000, rng, return_cond=True)
        # exactness of the returned conditionals: product must equal p(x)
        e2 = float(np.abs(np.exp(np.log(cond).sum(1)) - pfull[xs]).max())
        # pair correlations <Z_i Z_j> (single-site marginals are 1/2 by symmetry, a weak test)
        bits = ((xs[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1)
        zz = 1 - 2 * bits
        C_ff = (zz.T @ zz) / len(xs)
        pr = pfull.reshape((2,) * n)
        C_ed = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                m = pr.sum(axis=tuple(k for k in range(n) if k not in (i, j)))
                C_ed[i, j] = 1.0 if i == j else float(m[0, 0] + m[1, 1] - m[0, 1] - m[1, 0])
        z = float(np.abs(C_ff - C_ed).max() / np.sqrt(1.0 / 200_000))
        rep[f"sampler n={n} g={g}"] = (e, e2, z)
        if verbose:
            print(f"check 3  sampler n={n} g={g}: exact p(x) err {e:.1e}, product of conditionals err {e2:.1e}, pair correlations within {z:.1f} sigma (2e5 samples)")
        assert e < 1e-10 and e2 < 1e-10 and z < 4.5
    return rep


if __name__ == "__main__":
    t = time.time()
    validate()
    print(f"validation done [{time.time()-t:.0f}s]")
    # timing at n = 50
    M = open_chain_M(50, 1.0); smp = Sampler(M, 50)
    t = time.time(); smp.sample(20_000, np.random.default_rng(0), batch=4096); dt = time.time() - t
    print(f"n=50 sampling: {dt:.1f}s per 2e4 configurations -> {dt*50/60:.1f} min per 1e6")
    t = time.time(); blocks = site_blocks(M, 50, 12); dt = time.time() - t
    print(f"n=50 window blocks (l <= 13) for all 50 sites: {dt:.0f}s")
