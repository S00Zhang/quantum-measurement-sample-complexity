"""
Representation bias B(R) of the finite open chain by exact diagonalization (Appendix B 3).

Computes B(R) by both routes of :mod:`qmsc.tfim_chain` together with the site profiles I_i(R) for a grid
of (n, g) and writes one JSON per point to data/finite_chain_bias_n{n}_g{g}.json.

Usage:
    python scripts/compute_finite_chain_bias.py --n 12 --g 0.5 1.0 1.5
    python scripts/compute_finite_chain_bias.py --n 12 16 20 --g 0.5 1.0 1.5 --no-direct-above 16

``--no-direct-above N`` skips the memory-hungry direct route above size N (the entropy route is kept for
every point).  The direct route is always run at the smaller sizes so that the agreement test is exercised.
"""
import argparse
import os

from qmsc import tfim_chain as T
from qmsc.paths import DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[12])
    ap.add_argument("--g", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--bc", default="open")
    ap.add_argument("--no-direct-above", type=int, default=99, help="skip the direct route for n above this")
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for n in args.n:
        for g in args.g:
            res = T.compute_bias(n, g, bc=args.bc, with_direct=(n <= args.no_direct_above))
            path = os.path.join(args.out, f"finite_chain_bias_n{n}_g{g:g}.json")
            T.save_json(res, path)
            print("  ->", os.path.relpath(path), "| checks:",
                  {k: v for k, v in res["checks"].items() if k.endswith("_ok")})


if __name__ == "__main__":
    main()
