"""Recompute every number quoted in arXiv:2610.38028 from the result files in data/.

Each check prints PASS or FAIL with the derived and the quoted value, grouped by the place in the paper where the
number appears, and the script exits with a non-zero status if any check fails.

Usage (repository root, after ``pip install -e .``):  python scripts/audit_numbers.py
"""
import json
import os
import sys

import numpy as np

from qmsc.paths import DATA_DIR

ok = True


def check(name, got, expect, tol):
    """Numerical check |got - expect| <= tol."""
    global ok
    good = abs(got - expect) <= tol
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {name}: derived {got:.6g}  quoted {expect:.6g}")


def claim(name, good):
    """Qualitative check of a statement."""
    global ok
    ok &= bool(good)
    print(f"[{'PASS' if good else 'FAIL'}] {name}")


def section(title):
    print(f"\n== {title}")


def j(path):
    return json.load(open(os.path.join(DATA_DIR, path)))


bb = j("bulk_bias.json")
mc = j("critical_tail_mc.json")
cp = j("bulk_site_campaign.json")
st = j("bulk_campaign_statistics.json")
tf = j("template_fits.json")
ll = j("lnlnN_coefficients_check.json")
n50 = {g: j(f"finite_chain_campaign_n50_g{g}.json") for g in ("0.5", "1", "1.5")}
s50 = j("finite_chain_n50_statistics.json")
ct = j("critical_tail_n50.json")
cl = j("cell_law_check.json")
kappa = mc["LG_kappa_nats"]
N = np.array(cp["N_grid"], float)
LN2 = np.log(2)

# ------------------------------------------------------------------------------------------------------
section("Sec. IV and Fig. 2: bulk representation bias")
check("kappa, the Lau-Grassberger coefficient in nats (quoted 0.0589)", kappa, 0.0589, 5e-5)
check("bulk_bias.json uses the same kappa", bb["g"]["1"]["kappa_LG_nats"], kappa, 1e-9)
for g, lo, hi in (("0.5", 0.38, 0.45), ("1.5", 0.23, 0.36)):
    rec = bb["g"][g]
    b = np.array(rec["b"])
    R_ok = rec["R_ok"]
    rat = b[4 : R_ok + 1] / b[3:R_ok]  # b(R + 1) / b(R) for R = 3 .. R_ok - 1
    claim(
        f"g = {g}: the bias decays approximately exponentially, ratios b(R+1)/b(R) in [{rat.min():.2f}, {rat.max():.2f}] "
        f"over 3 <= R < {R_ok}, drifting slowly",
        lo - 5e-3 <= rat.min() and rat.max() <= hi + 5e-3 and np.all(np.diff(rat[1:]) > -0.03),
    )
c1 = bb["g"]["1"]
dev = np.array([abs(b_ * R - kappa) / kappa for R, b_ in zip(c1["R"], c1["b"])])
claim(f"g = 1: b(R) R / kappa within 15 % for R = 1 .. 32 (max deviation {100 * dev.max():.0f} %)", dev.max() < 0.15)
se = max(c1["se"])
claim(f"critical entropy-rate uncertainty about 1e-4 nats (doubling-sum standard error {se:.1e})", 3e-5 <= se <= 1.5e-4)
for g, (lo, hi), aq in (("0.5", (3, 10), 0.87), ("1.5", (1, 7), 1.48)):
    b = np.array(bb["g"][g]["b"])
    Rm = np.array(cp["g"][g]["R_star_measured"])
    check(f"radii visited by the optimizer at g = {g}, capped at R_ok (quoted {lo}-{hi})",
          Rm.min() * 100 + min(Rm.max(), bb["g"][g]["R_ok"]), lo * 100 + hi, 0)
    check(f"mean decay rate alpha over that window at g = {g} (quoted {aq})",
          float(np.mean(np.log(b[lo:hi] / b[lo + 1 : hi + 1]))), aq, 6e-3)

# ------------------------------------------------------------------------------------------------------
section("Fig. 3: estimation cost against the parameter-count law at g = 0.5")
r5 = cp["g"]["0.5"]
Ve = np.array(r5["V_exact"])
law = 2.0 ** np.arange(cp["R_MAX"] + 1)[None, :] / (2.0 * N[:, None])
ratio = Ve / law
i4, i8 = list(N).index(1e4), list(N).index(1e8)
claim(f"R = 4: the law is reached by N = 1e4 (exact / law = {ratio[i4, 4]:.2f} at 1e4, {ratio[i4 + 1, 4]:.2f} at 3e4)",
      ratio[i4, 4] >= 0.85 and ratio[i4 + 1, 4] >= 0.95)
claim(f"R = 8: the law is reached only at the largest sample sizes (exact / law = {ratio[i8 - 2, 8]:.2f} at 1e7, "
      f"{ratio[i8 - 1, 8]:.2f} at 3e7, {ratio[i8, 8]:.2f} at 1e8)", ratio[i8 - 2, 8] < 0.9 and ratio[i8 - 1, 8] >= 0.95)
claim(f"R = 16: the law substantially overestimates the cost (exact / law = {ratio[i8, 16]:.2f} at 1e8)", ratio[i8, 16] < 0.5)
docc = np.array(r5["d_occ"])
claim(f"R = 16, N = 1e8: contexts remain poorly sampled ({100 * docc[i8, 16] / 2 ** 16:.0f} % visited)", docc[i8, 16] / 2 ** 16 < 0.2)

# ------------------------------------------------------------------------------------------------------
section("Sec. V B and App. C 1: validation of the exact estimation cost at the bulk site")
n_pts = n3 = 0
for g in ("0.5", "1", "1.5"):
    r = cp["g"][g]
    Vh = np.array(r["V_hat"])
    Vm = np.nanmean(Vh, 0)
    Vse = np.nanstd(Vh, 0, ddof=1) / np.sqrt(Vh.shape[0])
    z = np.abs((Vm - np.array(r["V_exact"])) / Vse)
    n_pts += z.size
    n3 += int((z <= 3).sum())
    claim(f"g = {g}: {100 * np.mean(z <= 2):.1f} % of the grid within 2 SE of the exact prediction, max |z| = {z.max():.2f}",
          np.mean(z <= 2) >= 0.95)
check("grid points (3 couplings x 13 sample sizes x 18 radii)", n_pts, 702, 0)
check("grid points within 3 SE (all)", n3, n_pts, 0)

# ------------------------------------------------------------------------------------------------------
section("Sec. VI A and Fig. 4: capacity selection at the bulk site")
ex_ok = 0
miss_law = {}
for g in ("0.5", "1", "1.5"):
    r = cp["g"][g]
    Rm, Re, Rr = (np.array(r[k]) for k in ("R_star_measured", "R_star_exact", "R_star_regular"))
    ex_ok += int((Rm == Re).sum())
    miss_law[g] = int((Rm != Rr).sum())
check("exact prediction gives the measured optimal radius in all but one of the 39 cases", ex_ok, 38, 0)
r1 = cp["g"]["1"]
i3 = cp["N_grid"].index(3000)
tot1 = np.array(r1["b"])[None, None, :] + np.array(r1["V_hat"])
Fm1, Fse1 = tot1.mean(0), tot1.std(0, ddof=1) / np.sqrt(tot1.shape[0])
Rm3, Re3 = r1["R_star_measured"][i3], r1["R_star_exact"][i3]
claim(f"the one miss (g = 1, N = 3e3, measured {Rm3} against predicted {Re3}) is a tie: the two mean errors differ by "
      f"{abs(Fm1[i3, Re3] - Fm1[i3, Rm3]) / Fse1[i3, Rm3]:.2f} SE", abs(Fm1[i3, Re3] - Fm1[i3, Rm3]) / Fse1[i3, Rm3] < 1.0)
claim(f"the parameter-count law alone misses mostly at g = 0.5 ({miss_law['0.5']} of {sum(miss_law.values())} misses)",
      miss_law["0.5"] >= 0.7 * sum(miss_law.values()))
i6 = cp["N_grid"].index(1000000)
b5, Ve5 = np.array(r5["b"]), np.array(r5["V_exact"])[i6]
b1, Ve1 = np.array(r1["b"]), np.array(r1["V_exact"])[i6]
check("N = 1e6, g = 0.5: predicted optimal radius (quoted 8)", int(np.argmin(b5 + Ve5)), 8, 0)
check("N = 1e6, g = 1: predicted optimal radius (quoted 11)", int(np.argmin(b1 + Ve1)), 11, 0)
F5 = (b5[None, :] + np.array(r5["V_hat"])[:, i6, :]).mean(0)
check("N = 1e6, g = 0.5: measured minimum coincides (8)", int(np.argmin(F5)), 8, 0)
check("N = 1e6, g = 1: measured minimum coincides (11)", int(np.argmin(Fm1[i6])), 11, 0)
claim(f"N = 1e6, g = 1: bias remains the larger contribution at the optimum (b / v = {b1[11] / Ve1[11]:.1f})", b1[11] / Ve1[11] > 2)
claim("Fig. 4(a): the dotted bias continuation starts beyond R = 10, the trusted range at g = 0.5", bb["g"]["0.5"]["R_ok"] == 10)

# ------------------------------------------------------------------------------------------------------
section("Sec. VI B, Fig. 5 and Table II: scaling at the critical and gapped couplings")
for g, aq, ex_pred, ex_fit, sl_pred, sl_fit in (("0.5", 0.87, -0.56, -0.55, 0.64, 0.63), ("1.5", 1.48, -0.68, -0.69, 0.46, 0.43)):
    check(f"g = {g}: predicted exponent -alpha/(alpha + ln 2) (quoted {ex_pred})", -aq / (aq + LN2), ex_pred, 6e-3)
    check(f"g = {g}: fitted exponent of the optimal error over 1e2 .. 1e8 (quoted {ex_fit})", st[g]["exponent_Estar_full"], ex_fit, 6e-3)
    check(f"g = {g}: predicted radius slope 1/(alpha + ln 2) (quoted {sl_pred})", 1 / (aq + LN2), sl_pred, 6e-3)
    check(f"g = {g}: fitted radius slope against ln N (quoted {sl_fit})", st[g]["slope_Rstar_vs_lnN"], sl_fit, 6e-3)
check("g = 1: predicted radius slope 1/beta with beta = ln 2 (quoted 1.44)", 1 / LN2, 1.44, 6e-3)
f1 = tf["g"]["1"]
check("g = 1: leading exponent of the fitted row-III template (quoted (ln N)^-1.01)", f1["leading"]["mu"], 1.01, 6e-3)
check("g = 1: fitted radius slope 1/beta (quoted 1.47)", f1["leading"]["R_slope"], 1.47, 6e-3)
check("joint fit: mu (quoted 1.01)", f1["leading"]["mu"], 1.01, 6e-3)
check("joint fit: bootstrap error of mu (quoted 0.03)", f1["leading_se"]["mu"], 0.03, 0.006)
Rq = np.arange(1, 80)
totq = kappa / Rq[None, :] + 2.0 ** Rq[None, :] / (2.0 * N[:, None])
kq = totq.argmin(axis=1)
claim("the row-III template with mu = 1 keeps its optimum inside the tested radii over the sampled range", kq.max() < cp["R_MAX"])

# ------------------------------------------------------------------------------------------------------
section("App. A, Lemma 2: per-cell error of the Krichevsky-Trofimov estimate")
m = np.array(cl["m_grid"])
for p in ("0.5", "0.3", "0.1"):
    theta = float(p)
    c2 = 1 / (24 * theta * (1 - theta)) - 5 / 12
    r = np.array(cl["risk_kt"][p])
    big = m >= 1000
    check(f"second-order coefficient from exact binomial sums at theta = {p}", np.mean((r[big] - 1 / (2 * m[big])) * m[big] ** 2), c2, 6e-3)
    check(f"leading coefficient 1/2 at theta = {p}", np.mean(r[big] * m[big]), 0.5, 2e-3)

# ------------------------------------------------------------------------------------------------------
section("App. B 1: block entropies of the infinite chain")
check("largest block at g = 0.5 (quoted 22)", bb["g"]["0.5"]["lmax"], 22, 0)
check("largest block at g = 1.5 (quoted 20)", bb["g"]["1.5"]["lmax"], 20, 0)
check("largest block at g = 1 (quoted 20): exact doubling differences for 2R <= 19",
      2 * max(int(k) for k in c1["D_exact"]) + 1, 19, 0)
claim(f"check against exact diagonalization of a 16-site chain (block law error {bb['validation_g05_vs_ED16_l4']:.1e})",
      bb["validation_g05_vs_ED16_l4"] < 1e-3)

# ------------------------------------------------------------------------------------------------------
section("App. B 2: critical-tail test")
devs = np.array([abs(r["kappa"] - kappa) / kappa for r in mc["Z_g1"]])
claim(f"2R D_R within 4 % of kappa for R = 4 .. 96 (max deviation {100 * devs.max():.1f} %)", devs.max() <= 0.04)
check("largest Monte Carlo radius (quoted 96)", mc["Z_g1"][-1]["R"], 96, 0)
D96 = [r["D"] for r in mc["Z_g1"] if r["R"] == 96][0]
check("tail rule b(128) = 1.5 D_96", c1["tail_b128"], 1.5 * D96, 1e-12)
check("tail share of the reconstructed bias at R = 32 (quoted 24 %)", 100 * c1["tail_fraction"][c1["R"].index(32)], 24, 0.6)
cb = np.array(cp["g"]["1"]["b"])
dev12 = np.array([abs(cb[R] * R - kappa) / kappa for R in range(1, 13)])
claim(f"anchored critical bias b(R) R / kappa within 15 % for R <= 12 (max deviation {100 * dev12.max():.0f} %)", dev12.max() < 0.15)
for R, q in ((3, 0.057), (4, 0.058), (6, 0.059)):
    check(f"infinite chain, 2R D_R at R = {R}", ct["bulk_2RD"][R - 1], q, 0.0006)
claim(f"the 50-site centre site retains {100 * ct['ratio_centre_to_bulk'][0]:.0f} % of the bulk value at R = 1 and "
      f"{100 * ct['ratio_centre_to_bulk'][5]:.0f} % at R = 6 (truncated tail)", ct["ratio_centre_to_bulk"][5] < ct["ratio_centre_to_bulk"][0])

# ------------------------------------------------------------------------------------------------------
section("App. B 3: finite-chain bias and boundary effects")
for g in ("0.5", "1.5"):
    I20 = np.array(j(f"finite_chain_bias_n20_g{g}.json")["I_profile"])
    prof = I20[2]  # site-resolved conditional mutual information at R = 2, sites 1 .. 20
    mid, edge = prof[6:14], prof[3]
    claim(f"20-site chain, g = {g}, R = 2: approximately uniform conditional information away from the ends "
          f"(spread {100 * (mid.max() - mid.min()) / mid.mean():.0f} % over the central sites, boundary site at "
          f"{100 * edge / mid.mean():.0f} % of the bulk value)", (mid.max() - mid.min()) / mid.mean() < 0.1 and edge < 0.9 * mid.mean())
for g, xi, win in (("0.5", 1.081, [3, 8]), ("1.5", 0.614, [1, 5])):
    check(f"50-site chain, g = {g}: correlation length of the bias over the radii visited by the optimum (quoted {xi})",
          s50[g]["fit_xi"], xi, 1.5e-3)
    check(f"50-site chain, g = {g}: fit window", s50[g]["fit_window"][0] * 100 + s50[g]["fit_window"][1], win[0] * 100 + win[1], 0)

# ------------------------------------------------------------------------------------------------------
section("App. C: learning protocols")
check("bulk study: largest radius (quoted 17)", cp["R_MAX"], 17, 0)
check("bulk study: number of sample sizes (quoted 13)", len(N), 13, 0)
check("bulk study: smallest sample size (quoted 1e2)", N[0], 1e2, 0)
check("bulk study: largest sample size (quoted 1e8)", N[-1], 1e8, 0)
check("bulk study: training realizations (quoted 20)", cp["seeds"], 20, 0)
check("bulk study: block length R_MAX + 1 (quoted 18)", cp["R_MAX"] + 1, 18, 0)
check("50-site study: chain length", s50["n"], 50, 0)
check("50-site study: largest radius (quoted 12)", int(max(n50["1"]["R_grid"])), 12, 0)
check("50-site study: number of sample sizes (quoted nine)", len(n50["1"]["N_grid"]), 9, 0)
check("50-site study: largest sample size (quoted 1e6)", max(n50["1"]["N_grid"]), 1e6, 0)
check("50-site study: training realizations (quoted 20)", s50["seeds"], 20, 0)
check("50-site study: window of preceding spins W (quoted 12)", n50["1"]["W"], 12, 0)
claim(f"50-site study: Monte Carlo tail of the bias enters with a standard error below 1e-5 nats "
      f"(max {max(n50[g]['B_se'] for g in n50):.1e})", max(n50[g]["B_se"] for g in n50) < 1e-5)
n3 = sum(round((1 - s50[g]["within_3se"]) * s50[g]["grid_points"]) for g in n50)
claim(f"50-site study: the exact prediction agrees with the measured mean excess throughout the grid "
      f"(within 2 SE {[round(s50[g]['within_2se'], 3) for g in n50]}, beyond 3 SE: {n3} of {sum(s50[g]['grid_points'] for g in n50)})",
      min(s50[g]["within_2se"] for g in n50) >= 0.95 and n3 <= 1)
claim("50-site study: the exact prediction gives the measured optimum except at g = 1, N = 300",
      s50["agree_exact"] == 26 and s50["1"]["R_star_measured"][1] != s50["1"]["R_star_exact"][1])
B1, Vh1 = np.array(n50["1"]["B_ref"]), np.array(n50["1"]["V_hat"])
F300 = B1[None, :] + Vh1[:, 1, :]
m300, se300 = F300.mean(0), F300.std(0, ddof=1) / np.sqrt(F300.shape[0])
claim(f"g = 1, N = 300: the two competing radii are a statistical tie (difference {abs(m300[3] - m300[2]):.4f} "
      f"< one SE {max(se300[2], se300[3]):.4f})", abs(m300[3] - m300[2]) < max(se300[2], se300[3]))
claim("50-site study: the parameter-count law misses primarily in the ordered regime",
      s50["regular_misses_by_g"]["0.5"] >= 0.7 * sum(s50["regular_misses_by_g"].values()))
d05 = n50["0.5"]
iN4 = d05["N_grid"].index(10000)
B05, V05 = np.array(d05["B_ref"]), np.array(d05["V_exact"])[iN4]
F05 = (B05[None, :] + np.array(d05["V_hat"])[:, iN4, :]).mean(0)
check("Fig. 7(b): predicted optimal radius at g = 0.5, N = 1e4 (quoted 5)", int(np.argmin(B05 + V05)), 5, 0)
check("Fig. 7(b): measured minimum coincides (5)", int(np.argmin(F05)), 5, 0)

# ------------------------------------------------------------------------------------------------------
section("App. D 1: logarithmic corrections of rows II and III")
for r in ll["remainders"]:
    drift = abs(r["r"][-1] - r["limit"])
    claim(f"row {r['row']}: the remainder of the stated capacity converges (within {drift:.3f} of its limit at ln N = 600, "
          f"an error in the ln ln N coefficient would drift by {2 * np.log(600) - 2 * np.log(30):.1f})", drift < 0.12)
claim("the ln ln N factor explains most of the deviation of the optimal error from its leading form at every tabulated ln N",
      all(abs(c["rowII_ratio"] - 1) > abs(c["rowII_ratio"] - c["rowII_factor"]) and abs(c["rowIII_ratio"] - 1) > abs(c["rowIII_ratio"] - c["rowIII_factor"])
          for c in ll["Estar_correction"]))

# ------------------------------------------------------------------------------------------------------
section("App. D 2: fits to the bulk results")
check("leading-form fit at criticality: effective exponent of the error in ln N (quoted -1.8)", f1["effective"]["data_E_vs_lnlnN"], -1.8, 0.05)
check("leading-form fit at criticality: radius slope (quoted 1.0)", f1["effective"]["data_R_slope"], 1.0, 0.05)
check("joint fit of the full row-III template: mu (quoted 1.01 +- 0.03)", f1["leading"]["mu"], 1.01, 6e-3)
check("joint fit: mu bootstrap error (quoted 0.03)", f1["leading_se"]["mu"], 0.03, 0.006)
check("joint fit: kappa (quoted 0.061 +- 0.004)", f1["leading"]["kappa"], 0.061, 6e-4)
check("joint fit: kappa bootstrap error (quoted 0.004)", f1["leading_se"]["kappa"], 0.004, 0.0012)
check("joint fit: beta (quoted 0.680 +- 0.006)", f1["leading"]["beta"], 0.680, 6e-4)
check("joint fit: beta bootstrap error (quoted 0.006)", f1["leading_se"]["beta"], 0.006, 0.0015)
check("number of fitted optima (quoted 13)", len(N), 13, 0)
check("reference value of beta, ln 2", LN2, 0.693, 6e-4)
claim("the fit is insensitive to the relative weight of the radius residuals",
      all(abs(r["leading"]["mu"] - f1["leading"]["mu"]) < 0.01 and abs(r["leading"]["R_slope"] - f1["leading"]["R_slope"]) < 0.02
          for r in f1["fits_by_w"].values()))
osens = f1["offset_sensitivity"]
dm = max(abs(osens["-0.0001"]["mu"] - osens["0"]["mu"]), abs(osens["0.0001"]["mu"] - osens["0"]["mu"]))
claim(f"a 1e-4 nats offset of the critical bias moves mu by {dm:.3f}, within its bootstrap error", dm < f1["leading_se"]["mu"])
check("bootstrap: training realizations resampled (quoted 20)", cp["seeds"], 20, 0)

print("\nAUDIT RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
