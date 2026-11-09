"""Exact Born distribution of the finite transverse-field Ising chain and the representation bias of
local autoregressive models (Sec. III A and Appendix B 3 of the paper).

Physics
-------
The 1D transverse-field Ising model with open boundary conditions,

    H(g) = -J sum_{i=1}^{n-1} Z_i Z_{i+1} - g sum_{i=1}^{n} X_i,     J = 1.

The ground state is obtained by matrix-free Lanczos (scipy ``eigsh`` on a ``LinearOperator``).  In the Z basis H
is stoquastic and the flip graph is connected, so for g > 0 the ground state is unique with strictly positive
amplitudes (Perron-Frobenius) and is Z2-even.  The Born distribution p_g(x) = psi_g(x)^2 is therefore strictly
positive and is available exactly for every configuration x, so every population quantity below is evaluated
without sampling noise.

Representation bias
-------------------
For the radius-R local autoregressive class the KL-optimal model is

    q*_R(x) = prod_i p(x_i | C_{i,R}),      C_{i,R} = x_{max(1,i-R) : i-1},

and its representation bias is a sum of conditional mutual informations (Lemma 1),

    B(R) = D_KL(p || q*_R) = sum_i I(X_i : D_{i,R} | C_{i,R}),     D_{i,R} = x_{< i-R}.

Two independent routes are implemented (direct KL on the 2^n grid, and the four-entropy identity) plus a
brute-force reference for n <= 8.  Agreement of the two routes to 1e-10 is the acceptance test.

Conventions
-----------
* Sites are 1-based in the documentation (i = 1..n); array axis i-1 holds site i.
* x_i in {0,1} with Z_i = +1 <-> x_i = 0 and Z_i = -1 <-> x_i = 1, i.e. z_i = 1 - 2 x_i.
* Flat basis index b in [0, 2^n): x_i = (b >> (n-i)) & 1, so site 1 is the most significant bit and
  ``p.reshape((2,)*n)[x_1, ..., x_n] = p(x)``.
* Natural logarithms (nats) everywhere; 0 log 0 = 0.  Energies in units of J.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

__all__ = [
    "tfim_diagonal", "tfim_linear_operator", "tfim_dense", "GroundState", "ground_state",
    "born_distribution", "marginal", "entropy", "window_entropy_table", "conditional_table",
    "multi_information", "bias_cmi", "bias_direct", "bias_bruteforce",
    "compute_bias", "save_json",
]


# ----------------------------------------------------------------------------------------------
# Hamiltonian
# ----------------------------------------------------------------------------------------------
def _bits(n: int) -> np.ndarray:
    """Array of shape (2^n, n) with entry [b, i-1] = x_i(b) = (b >> (n-i)) & 1 (int8)."""
    b = np.arange(2 ** n, dtype=np.int64)[:, None]
    shifts = (n - 1 - np.arange(n, dtype=np.int64))[None, :]
    return ((b >> shifts) & 1).astype(np.int8)


def tfim_diagonal(n: int, J: float = 1.0, bc: str = "open") -> np.ndarray:
    """Diagonal of H in the Z basis: -J sum_i z_i z_{i+1} with z_i = 1 - 2 x_i (units of J).

    ``bc='open'`` sums bonds (1,2)...(n-1,n); ``bc='periodic'`` adds the wrap bond (n,1) — kept only as a
    later control, the study uses open chains.
    """
    z = 1 - 2 * _bits(n).astype(np.float64)
    diag = -J * np.sum(z[:, :-1] * z[:, 1:], axis=1)
    if bc == "periodic":
        diag = diag - J * z[:, -1] * z[:, 0]
    elif bc != "open":
        raise ValueError("bc must be 'open' or 'periodic'")
    return diag


def tfim_linear_operator(n: int, g: float, J: float = 1.0, bc: str = "open") -> LinearOperator:
    """Matrix-free H(g) = -J sum Z_i Z_{i+1} - g sum X_i acting on flat vectors of length 2^n.

    X_i flips bit i, which on the (2,)*n tensor is ``np.flip(v, axis=i-1)``; the transverse-field term is
    therefore applied without ever storing a 2^n x 2^n matrix.  Memory: O(2^n).
    """
    dim = 2 ** n
    diag = tfim_diagonal(n, J, bc)
    shape = (2,) * n

    def matvec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(dim)
        out = diag * v
        vt = v.reshape(shape)
        out_t = out.reshape(shape)          # view onto ``out``
        for axis in range(n):
            out_t -= g * np.flip(vt, axis=axis)
        return out

    return LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def tfim_dense(n: int, g: float, J: float = 1.0, bc: str = "open") -> np.ndarray:
    """Dense H built independently from Kronecker products (for tests only, n <= 10).

    Uses the same site/bit convention as ``tfim_linear_operator`` (site 1 = leftmost Kronecker factor =
    most significant bit) but none of its code, so it is an independent check of the operator.
    """
    I2 = np.eye(2)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])

    def site_op(op: np.ndarray, i: int) -> np.ndarray:
        mats = [I2] * n
        mats[i - 1] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = np.zeros((2 ** n, 2 ** n))
    bonds = [(i, i + 1) for i in range(1, n)]
    if bc == "periodic":
        bonds.append((n, 1))
    for i, j in bonds:
        H -= J * site_op(Z, i) @ site_op(Z, j)
    for i in range(1, n + 1):
        H -= g * site_op(X, i)
    return H


# ----------------------------------------------------------------------------------------------
# Ground state
# ----------------------------------------------------------------------------------------------
@dataclass
class GroundState:
    """Ground-state record.  ``psi`` is flat (length 2^n), unit norm, Z2-even, all entries > 0."""
    n: int
    g: float
    J: float
    bc: str
    psi: np.ndarray
    energy: float            # E_0 (units of J), lowest state of the Z2-even sector = global ground state
    gap: float               # E_1 - E_0 over the FULL spectrum = min(gap_odd, gap_even)
    gap_odd: float           # E_0^{odd} - E_0  (~ e^{-n/xi} for g < 1: the quasi-degenerate partner)
    gap_even: float          # E_1^{even} - E_0 (first excitation inside the even sector)
    residual: float          # || H psi - E_0 psi ||_2
    min_amplitude: float     # min_x psi(x)  (must be > 0 for g > 0)
    wall_time_s: float


def _global_flip(v: np.ndarray, n: int) -> np.ndarray:
    """P v with P = prod_i X_i (reverse every bit): flat index b -> (2^n - 1) - b."""
    return v[::-1].copy() if v.shape == (2 ** n,) else np.flip(v, axis=tuple(range(n)))


def _sector_operator(H: LinearOperator, n: int, parity: int, shift: float) -> LinearOperator:
    """P_s H P_s + shift * (1 - P_s): H restricted to the Z2 sector s = +1 (even) or -1 (odd).

    P_s = (1 + s P)/2 with P the global spin flip.  Projecting on every matvec makes the Lanczos iteration
    exactly sector-preserving (no roundoff leakage between sectors); the complementary sector is pushed to
    eigenvalue ``shift`` (chosen far above the ground energy) so it can never be returned by ``which='SA'``.
    """
    dim = 2 ** n

    def proj(v: np.ndarray) -> np.ndarray:
        return 0.5 * (v + parity * _global_flip(v, n))

    def matvec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(dim)
        pv = proj(v)
        return proj(H @ pv) + shift * (v - pv)

    return LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def ground_state(n: int, g: float, J: float = 1.0, bc: str = "open", tol: float = 1e-13,
                 ncv: int | None = None) -> GroundState:
    """Unique finite-n ground state of H(g) for g > 0, by matrix-free Lanczos.

    Steps: (1) ``eigsh(k=2, which='SA')`` on the Z2-even sector operator (sector-projected on every matvec)
    from a uniform positive start vector -> E_0 and the first even excitation; (2) ``eigsh(k=1)`` on the
    odd-sector operator -> the odd partner E_0^{odd}, so the reported gap is over the full spectrum;
    (3) fix the global sign so the amplitudes are positive and assert positivity (Perron-Frobenius);
    (4) record energy, gaps, residual.
    """
    if g <= 0:
        raise ValueError("g must be > 0 (the g = 0 ground state is degenerate)")
    t0 = time.perf_counter()
    dim = 2 ** n
    H = tfim_linear_operator(n, g, J, bc)
    shift = 10.0 * (J * n + g * n) + 10.0            # far above any physical eigenvalue
    if ncv is None:
        ncv = min(dim, 24)
    rng = np.random.default_rng(0)
    # even sector: E_0 and first even excitation
    v0 = np.full(dim, 1.0 / np.sqrt(dim))
    vals_e, vecs_e = eigsh(_sector_operator(H, n, +1, shift), k=2, which="SA", v0=v0, tol=tol,
                           ncv=ncv, maxiter=20000)
    order = np.argsort(vals_e)
    vals_e, vecs_e = vals_e[order], vecs_e[:, order]
    # odd sector: lowest odd state (the quasi-degenerate partner for g < 1)
    r = rng.standard_normal(dim)
    v0_odd = 0.5 * (r - _global_flip(r, n))
    v0_odd /= np.linalg.norm(v0_odd)
    vals_o = eigsh(_sector_operator(H, n, -1, shift), k=1, which="SA", v0=v0_odd, tol=tol,
                   ncv=ncv, maxiter=20000, return_eigenvectors=False)
    psi = vecs_e[:, 0]
    psi = 0.5 * (psi + _global_flip(psi, n))         # exact even projection (no-op up to roundoff)
    psi /= np.linalg.norm(psi)
    if psi[np.argmax(np.abs(psi))] < 0:
        psi = -psi
    Hpsi = H @ psi
    energy = float(psi @ Hpsi)
    residual = float(np.linalg.norm(Hpsi - energy * psi))
    gap_even = float(vals_e[1] - energy)
    gap_odd = float(vals_o[0] - energy)
    min_amp = float(psi.min())
    if min_amp <= 0:
        raise RuntimeError(f"ground state not strictly positive (min psi = {min_amp:.3e}); "
                           "Perron-Frobenius violated -> Lanczos did not converge to the ground state")
    if gap_odd < -1e-9:
        raise RuntimeError(f"odd sector below even sector (gap_odd = {gap_odd:.3e}); "
                           "contradicts Perron-Frobenius -> Lanczos failure")
    return GroundState(n=n, g=g, J=J, bc=bc, psi=psi, energy=energy, gap=min(gap_even, gap_odd),
                       gap_odd=gap_odd, gap_even=gap_even, residual=residual,
                       min_amplitude=min_amp, wall_time_s=time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------------
# Born distribution and marginals
# ----------------------------------------------------------------------------------------------
def born_distribution(psi: np.ndarray, n: int) -> np.ndarray:
    """p(x) = |<x|psi>|^2 as an array of shape (2,)*n with axis i-1 <-> site i."""
    p = (np.asarray(psi, dtype=np.float64) ** 2).reshape((2,) * n)
    return p


def marginal(p: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Marginal of p on the contiguous window of sites lo..hi (1-based, inclusive).

    Returns an array of shape (2,)*(hi-lo+1); an empty window (hi < lo) returns the scalar array 1.0.
    """
    n = p.ndim
    if hi < lo:
        return np.array(1.0)
    keep = set(range(lo - 1, hi))
    axes = tuple(a for a in range(n) if a not in keep)
    return p.sum(axis=axes) if axes else p


def entropy(q: np.ndarray) -> float:
    """Shannon entropy in nats, with 0 log 0 = 0."""
    q = np.asarray(q, dtype=np.float64).ravel()
    nz = q[q > 0]
    return float(-np.sum(nz * np.log(nz)))


def window_entropy_table(p: np.ndarray) -> np.ndarray:
    """H[a, b] = H(X_{a:b}) for 1 <= a <= b <= n (1-based; H[a, a-1] = 0 for the empty window).

    Stored as an (n+2) x (n+2) array indexed directly by (a, b); unused cells are NaN.
    """
    n = p.ndim
    H = np.full((n + 2, n + 2), np.nan)
    for a in range(1, n + 2):
        H[a, a - 1] = 0.0
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            H[a, b] = entropy(marginal(p, a, b))
    return H


def conditional_table(p: np.ndarray, i: int, R: int) -> tuple[int, np.ndarray]:
    """p(x_i | c_{i,R}) for the window lo..i, lo = max(1, i-R).

    Returns (lo, table) with table of shape (2,)*(i-lo+1); axes are sites lo..i in order, the last axis is
    x_i.  Cells whose context has zero probability are set to 0 (they never contribute: 0 log 0 = 0).
    """
    lo = max(1, i - R)
    joint = marginal(p, lo, i)                       # p(x_{lo:i})
    ctx = joint.sum(axis=-1, keepdims=True)          # p(x_{lo:i-1})
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(ctx > 0, joint / ctx, 0.0)
    return lo, cond


def multi_information(p: np.ndarray) -> float:
    """B(0) reference: sum_i H(X_i) - H(X) (total correlation), computed without the CMI machinery."""
    n = p.ndim
    return sum(entropy(marginal(p, i, i)) for i in range(1, n + 1)) - entropy(p)


# ----------------------------------------------------------------------------------------------
# Representation bias, route B (conditional mutual informations from window entropies)
# ----------------------------------------------------------------------------------------------
def bias_cmi(p: np.ndarray, R: int, H: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Route B.  B(R) = sum_i I(X_i : D_{i,R} | C_{i,R}) with

        I_i(R) = H(X_{lo:i}) + H(X_{1:i-1}) - H(X_{lo:i-1}) - H(X_{1:i}),   lo = max(1, i-R).

    Returns (B, profile) where profile[i-1] = I_i(R).  For i <= R+1 the four terms cancel identically
    (empty discarded past), so I_i(R) = 0 exactly.  ``H`` is the window-entropy table (computed if None).
    """
    n = p.ndim
    if H is None:
        H = window_entropy_table(p)
    profile = np.zeros(n)
    for i in range(1, n + 1):
        lo = max(1, i - R)
        if lo == 1:
            continue                                  # D_{i,R} empty -> I_i = 0 exactly
        profile[i - 1] = H[lo, i] + H[1, i - 1] - H[lo, i - 1] - H[1, i]
    return float(profile.sum()), profile


# ----------------------------------------------------------------------------------------------
# Representation bias, route A (direct KL divergence on the configuration grid)
# ----------------------------------------------------------------------------------------------
def log_q_star(p: np.ndarray, R: int) -> np.ndarray:
    """log q*_R(x) = sum_i log p(x_i | c_{i,R}) on the full (2,)*n grid, by broadcasting each window table."""
    n = p.ndim
    log_q = np.zeros_like(p)
    for i in range(1, n + 1):
        lo, cond = conditional_table(p, i, R)
        with np.errstate(divide="ignore"):
            log_cond = np.log(cond)                   # -inf where cond == 0 (only where p == 0)
        shape = [1] * n
        for axis in range(lo - 1, i):
            shape[axis] = 2
        log_q = log_q + log_cond.reshape(shape)
    return log_q


def bias_direct(p: np.ndarray, R: int) -> float:
    """Route A.  B(R) = sum_x p(x) [log p(x) - log q*_R(x)], masking p(x) = 0 cells (0 log 0 = 0)."""
    log_q = log_q_star(p, R)
    mask = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(mask, p * (np.log(np.where(mask, p, 1.0)) - log_q), 0.0)
    return float(terms.sum())


# ----------------------------------------------------------------------------------------------
# Representation bias, route C (brute force, n <= 8)
# ----------------------------------------------------------------------------------------------
def bias_bruteforce(p: np.ndarray, R: int) -> float:
    """Route C.  Literal chain rule with explicit loops; no reuse of ``marginal``/``entropy``.

        B(R) = sum_x p(x) sum_i [ log p(x_i | x_{<i}) - log p(x_i | c_{i,R}) ],

    where every conditional is p(prefix)/p(shorter prefix) with the prefix probabilities obtained by
    explicit summation over all completions.  O(n 4^n) — tests only.
    """
    n = p.ndim
    if n > 8:
        raise ValueError("brute force is for n <= 8")
    configs = list(itertools.product((0, 1), repeat=n))

    def prob_of_assignment(assign: dict[int, int]) -> float:
        """p(X_j = assign[j] for all j in assign), by summing p over all configurations that agree."""
        total = 0.0
        for x in configs:
            if all(x[j - 1] == v for j, v in assign.items()):
                total += float(p[x])
        return total

    B = 0.0
    for x in configs:
        px = float(p[x])
        if px == 0.0:
            continue
        s = 0.0
        for i in range(1, n + 1):
            full_prefix = {j: x[j - 1] for j in range(1, i)}
            ctx = {j: x[j - 1] for j in range(max(1, i - R), i)}
            with_i_full = {**full_prefix, i: x[i - 1]}
            with_i_ctx = {**ctx, i: x[i - 1]}
            log_p_full = np.log(prob_of_assignment(with_i_full)) - np.log(prob_of_assignment(full_prefix))
            log_p_ctx = np.log(prob_of_assignment(with_i_ctx)) - np.log(prob_of_assignment(ctx))
            s += log_p_full - log_p_ctx
        B += px * s
    return float(B)


# ----------------------------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------------------------
def compute_bias(n: int, g: float, J: float = 1.0, bc: str = "open", R_values=None,
               with_direct: bool = True, verbose: bool = True) -> dict:
    """H(g) -> psi -> p -> q*_R -> B(R) for all R; returns a JSON-serializable results dict.

    Includes both routes, the site-resolved profile I_i(R), the multi-information reference for R = 0,
    ground-state diagnostics and timings, with sanity flags (normalization, positivity, monotonicity in R).
    """
    t0 = time.perf_counter()
    gs = ground_state(n, g, J, bc)
    p = born_distribution(gs.psi, n)
    norm = float(p.sum())
    t1 = time.perf_counter()
    Htab = window_entropy_table(p)
    t2 = time.perf_counter()
    if R_values is None:
        R_values = list(range(0, n))
    B_cmi, B_dir, profiles = [], [], []
    for R in R_values:
        b, prof = bias_cmi(p, R, Htab)
        B_cmi.append(b)
        profiles.append(prof.tolist())
        B_dir.append(bias_direct(p, R) if with_direct else None)
    t3 = time.perf_counter()
    mi = multi_information(p)
    B = np.array(B_cmi)
    checks = {
        "normalization_ok": abs(norm - 1.0) < 1e-12,
        "nonnegative_ok": bool(np.all(B >= -1e-12)),
        "monotone_ok": bool(np.all(np.diff(B) <= 1e-12)),
        "zero_at_full_context_ok": bool(abs(B[-1]) < 1e-12) if R_values[-1] == n - 1 else None,
        "B0_equals_multi_information_ok": bool(abs(B[0] - mi) < 1e-10) if R_values[0] == 0 else None,
        "routes_agree_ok": bool(max(abs(a - b) for a, b in zip(B_dir, B_cmi)) < 1e-10) if with_direct else None,
        "max_route_discrepancy": float(max(abs(a - b) for a, b in zip(B_dir, B_cmi))) if with_direct else None,
        "positivity_ok": gs.min_amplitude > 0,
    }
    out = {
        "n": n, "g": g, "J": J, "bc": bc, "R": list(R_values),
        "energy": gs.energy, "energy_per_site": gs.energy / n, "gap": gs.gap, "gap_odd": gs.gap_odd,
        "gap_even": gs.gap_even, "residual": gs.residual, "min_amplitude": gs.min_amplitude,
        "p_norm": norm, "entropy_total": entropy(p), "multi_information": mi,
        "B_cmi": B_cmi, "B_direct": B_dir, "b_per_site_cmi": (B / n).tolist(),
        "I_profile": profiles,               # I_profile[k][i-1] = I_i(R_k)
        "checks": checks,
        "timing_s": {"ground_state": t1 - t0, "entropy_table": t2 - t1, "bias_all_R": t3 - t2,
                     "total": time.perf_counter() - t0},
        "nats": True,
    }
    if verbose:
        print(f"n={n} g={g}: E0/n={gs.energy/n:.6f} gap={gs.gap:.3e} (odd {gs.gap_odd:.2e}, even {gs.gap_even:.2e}) "
              f"resid={gs.residual:.1e} "
              f"min_psi={gs.min_amplitude:.2e} | B(0)={B[0]:.6f} B(1)={B[1]:.3e} "
              f"maxΔ(A,B)={checks['max_route_discrepancy']} | {out['timing_s']['total']:.1f}s")
    return out


def save_json(result: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
