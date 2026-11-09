"""
Exact finite-sample estimation cost of the smoothed histogram learner (Proposition 1 and Appendix A).

For the add-1/2 learner of :mod:`qmsc.histogram_learner` and an exactly known target p, the expected
estimation cost is an exact identity (linearity of expectation, with the outcomes within a cell i.i.d.
given the counts):

    E_{S_N} L(R, N) = sum_{i,c} pi_{i,c} * E_{m ~ Bin(N, pi_{i,c})} h(m, theta_{i,c}),

    h(m, theta) = E_{k ~ Bin(m, theta)} D_KL( Bern(theta) || Bern( (k + 1/2) / (m + 1) ) ),

with pi_{i,c} = P(C_{i,R} = c) and theta_{i,c} = P(X_i = 1 | C_{i,R} = c) exact window marginals of p.
Dependence between sites inside one configuration affects the fluctuations of L, not this mean.

Numerics: h(m, theta) is tabulated by exact binomial sums over k on m-nodes (every m <= 128, log-spaced to
4000) times a folded theta-grid and interpolated in (log m, log theta).  The outer expectation over m uses
the exact binomial window when it fits under m = 4000, otherwise the fixed-theta expansion
1/(2m) + c_2(theta)/m^2 averaged over m (used only where theta(1 - theta) >= 1e-2) or a Gauss-Hermite
quadrature over m with exact inner sums (extreme theta).  Every cell records which branch it used.  The
predictor is clipped to [EPS, 1 - EPS] as in the learner; the clip never binds inside the tabulated range.

Running the module cross-checks the identity against a fresh Monte Carlo of the learner at n = 12:

    python -m qmsc.estimation_cost
"""
import numpy as np
from scipy.stats import binom

from qmsc import histogram_learner as LL
from qmsc import tfim_chain as T

EPS = 1e-4
M_CAP = 4000                      # exact-window cap for the outer expectation
S_ASYM = 1e-2                     # theta(1-theta) floor for the large-m asymptotic branch


# ----------------------------------------------------------------------------------------------
# Inner expectation h(m, theta), exact at nodes, interpolated between
# ----------------------------------------------------------------------------------------------
def h_exact(m: int, thetas: np.ndarray) -> np.ndarray:
    """Exact E_k D_KL(theta || clipped add-1/2 predictor) for one m and an array of thetas."""
    k = np.arange(m + 1)
    q = np.clip((k + 0.5) / (m + 1), EPS, 1 - EPS)           # (m+1,)
    th = thetas[:, None]                                      # (T, 1)
    w = binom.pmf(k[None, :], m, th)                          # (T, m+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(th > 0, th * np.log(th / q[None, :]), 0.0) + \
            np.where(th < 1, (1 - th) * np.log((1 - th) / (1 - q[None, :])), 0.0)
    return np.sum(w * d, axis=1)


class HTable:
    """h(m, theta) on m-nodes x folded theta-grid; bilinear interpolation in (log m, log theta_folded)."""

    def __init__(self, m_cap: int = M_CAP):
        small = np.arange(1, 129)
        big = np.unique(np.round(np.logspace(np.log10(130), np.log10(m_cap), 120)).astype(int))
        self.m_nodes = np.concatenate([small, big])
        self.t_nodes = np.logspace(-7, np.log10(0.5), 360)
        self.table = np.array([h_exact(int(m), self.t_nodes) for m in self.m_nodes])   # (M, T)
        self.log_m = np.log(self.m_nodes)
        self.log_t = np.log(self.t_nodes)

    def __call__(self, m_arr: np.ndarray, theta: float) -> np.ndarray:
        """h at integer m values (array) for one theta; m = 0 handled exactly (predicts 1/2)."""
        tf = min(theta, 1.0 - theta)                          # symmetry k <-> m-k
        tf = max(tf, 1e-7)
        it = np.clip(np.searchsorted(self.log_t, np.log(tf)) - 1, 0, len(self.t_nodes) - 2)
        wt = (np.log(tf) - self.log_t[it]) / (self.log_t[it + 1] - self.log_t[it])
        col = (1 - wt) * self.table[:, it] + wt * self.table[:, it + 1]
        out = np.empty(len(m_arr))
        zero = m_arr == 0
        if zero.any():
            q0 = 0.5                                          # m = 0: the add-1/2 predictor is exactly 1/2
            out[zero] = (theta * np.log(theta / q0) if theta > 0 else 0.0) + \
                        ((1 - theta) * np.log((1 - theta) / (1 - q0)) if theta < 1 else 0.0)
        pos = ~zero
        lm = np.log(np.maximum(m_arr[pos], 1))
        im = np.clip(np.searchsorted(self.log_m, lm) - 1, 0, len(self.m_nodes) - 2)
        wm = (lm - self.log_m[im]) / (self.log_m[im + 1] - self.log_m[im])
        out[pos] = (1 - wm) * col[im] + wm * col[im + 1]
        return out


def c2(theta: float) -> float:
    s = theta * (1 - theta)
    return 1.0 / (24.0 * s) - 5.0 / 12.0


# ----------------------------------------------------------------------------------------------
# Outer expectation over m ~ Bin(N, pi)
# ----------------------------------------------------------------------------------------------
def cell_expectation(N: int, pi: float, theta: float, H: HTable) -> tuple[float, str]:
    """E_m h(m, theta), m ~ Bin(N, pi).  Returns (value, branch)."""
    pi = min(max(pi, 0.0), 1.0)                              # guard float roundoff (pi = 1 + 2e-16 -> NaN pmf)
    theta = min(max(theta, 0.0), 1.0)
    mu = N * pi
    sig = np.sqrt(max(N * pi * (1 - pi), 1e-30))
    if N <= M_CAP or mu + 10 * sig + 10 <= M_CAP:
        lo = int(max(0, np.floor(mu - 10 * sig - 10)))
        hi = int(min(N, np.ceil(mu + 10 * sig + 10)))
        m = np.arange(lo, hi + 1)
        w = binom.pmf(m, N, pi)
        return float(np.sum(w * H(m, theta))), "window"
    s = theta * (1 - theta)
    if s >= S_ASYM:
        inv_m = (1.0 / mu) * (1.0 + (1 - pi) / mu)           # E[1/m] to O(mu^-3)
        inv_m2 = (1.0 / mu ** 2) * (1.0 + 3 * (1 - pi) / mu)
        return float(0.5 * inv_m + c2(theta) * inv_m2), "asymptotic"
    # extreme theta, huge m: normal quadrature over m with exact inner k-sums (k-range is small)
    nodes, weights = np.polynomial.hermite_e.hermegauss(15)
    m_nodes = np.clip(np.round(mu + sig * nodes).astype(int), 1, N)
    vals = []
    tf = min(theta, 1 - theta)
    for m in m_nodes:
        lam = m * tf
        k_hi = int(lam + 12 * np.sqrt(lam) + 25)
        k = np.arange(0, min(k_hi, m) + 1)
        w = binom.pmf(k, m, tf)
        kk = k if theta <= 0.5 else m - k
        q = np.clip((kk + 0.5) / (m + 1), EPS, 1 - EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (theta * np.log(theta / q) if theta > 0 else 0.0) + (1 - theta) * np.log((1 - theta) / (1 - q))
        vals.append(np.sum(w * d))
    return float(np.sum(weights * np.array(vals)) / np.sum(weights)), "quadrature"


# ----------------------------------------------------------------------------------------------
# Cells and the full identity
# ----------------------------------------------------------------------------------------------
def cells_for(p: np.ndarray, R: int):
    """Yield (pi, theta) for every cell (i, c) of the radius-R site-dependent class."""
    n = p.ndim
    out = []
    for i in range(1, n + 1):
        lo, cond = T.conditional_table(p, i, R)              # cond: (2,)*(i-lo+1), last axis = x_i
        joint = T.marginal(p, lo, i).reshape(-1, 2)          # (2^w, 2) context MSB-first
        pis = joint.sum(axis=1)
        thetas = np.where(pis > 0, joint[:, 1] / np.maximum(pis, 1e-300), 0.5)
        out.append((pis, thetas))
    return out


def exact_L(p: np.ndarray, R: int, N: int, B_ref: float, H: HTable) -> dict:
    """E L = sum_cells pi * E_m h(m, theta) (exact identity), with the parameter count d_R and the expected
    number of visited cells."""
    total = 0.0
    branches = {"window": 0, "asymptotic": 0, "quadrature": 0}
    d_R = 0
    d_eff_visited = 0.0
    for pis, thetas in cells_for(p, R):
        d_R += len(pis)
        for pi, th in zip(pis, thetas):
            if pi <= 0:
                continue
            v, br = cell_expectation(N, float(pi), float(th), H)
            total += pi * v
            branches[br] += 1
            d_eff_visited += 1.0 - (1.0 - pi) ** N
    return {"L_exact": total, "d_R": d_R, "d_eff_expected_visited": d_eff_visited,
            "L_asym_dR": d_R / (2.0 * N), "branches": branches, "B_ref": B_ref}


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------
def validate():
    """Monte Carlo cross-check at n = 12, g = 1: 400 fresh datasets through the actual learner per (N, R)."""
    n, g = 12, 1.0
    p = T.born_distribution(T.ground_state(n, g).psi, n)
    Ht = T.window_entropy_table(p)
    B = {R: T.bias_cmi(p, R, Ht)[0] for R in range(n)}
    H = HTable()
    rng = np.random.default_rng(777)
    print("n=12 g=1: exact identity vs 400-dataset Monte Carlo (mean +- se)")
    for N in (100, 1000, 10000):
        for R in (0, 2, 4, 6):
            ex = exact_L(p, R, N, B[R], H)
            mc = np.array([LL.learn_and_evaluate(p, LL.sample_flat(p, N, rng), R, B[R])["L"]
                           for _ in range(400)])
            se = mc.std() / 20.0
            z = (mc.mean() - ex["L_exact"]) / se
            print(f"  N={N:<6} R={R}: exact={ex['L_exact']:.5e}  MC={mc.mean():.5e} ± {se:.1e}  z={z:+.2f} "
                  f"(branches {ex['branches']})")


if __name__ == "__main__":
    validate()
