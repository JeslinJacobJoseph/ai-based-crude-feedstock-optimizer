"""
SECTION 6 - Optimization: Maximize GRM with Differential Evolution
==================================================================
Objective : maximize predicted GRM/bbl subject to all 8 refinery constraints.
Method    : Differential Evolution -- a derivative-free GLOBAL optimizer.

Why DE, not LP: the objective is a trained GradientBoosting model (non-convex,
non-differentiable). LP needs a linear objective; DE treats the ML model as a
black-box oracle. (This was mislabelled "Bayesian Optimization" in v1 -- it is
Differential Evolution.)

Run:  python src/s06_optimization.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from scipy.optimize import differential_evolution

from config import (CRUDES, CNAMES, DISPLAY, GRM_MODEL, RISK_MODEL,
                    MIN_RATIO, MAX_RATIO, SEED, CONSTRAINTS)
from features import build_feature_frame, analytic_props

GRM = joblib.load(GRM_MODEL)
RISK = joblib.load(RISK_MODEL)


def predict(ratios):
    """Return (predicted GRM, predicted asphaltene-risk, analytic props)."""
    X = build_feature_frame(ratios)
    grm = float(GRM.predict(X)[0])
    risk = float(RISK.predict_proba(X)[0, 1])
    return grm, risk, analytic_props(ratios)


def objective(r3):
    """DE minimises -> negate GRM. 4th ratio derived from first 3.
    Exterior penalty method handles constraint violations."""
    r4 = np.array(list(r3) + [max(0.0, 1.0 - sum(r3))])
    if r4.min() < MIN_RATIO or r4.max() > MAX_RATIO:
        return 1e6
    r4 = r4 / r4.sum()

    grm, risk, p = predict(r4)
    pen = 0.0
    if p["calc_api"] < CONSTRAINTS["api_min"][0]:
        pen += 200 * (CONSTRAINTS["api_min"][0] - p["calc_api"]) ** 2
    if p["calc_api"] > CONSTRAINTS["api_max"][0]:
        pen += 200 * (p["calc_api"] - CONSTRAINTS["api_max"][0]) ** 2
    if p["calc_sulphur"] > CONSTRAINTS["sulphur_max"][0]:
        pen += 500 * (p["calc_sulphur"] - CONSTRAINTS["sulphur_max"][0]) ** 2
    if p["calc_tan"] > CONSTRAINTS["tan_max"][0]:
        pen += 300 * (p["calc_tan"] - CONSTRAINTS["tan_max"][0]) ** 2
    metals = p["calc_ni"] + p["calc_v"]
    if metals > CONSTRAINTS["metals_max"][0]:
        pen += 400 * (metals - CONSTRAINTS["metals_max"][0]) ** 2
    if p["calc_yres"] > CONSTRAINTS["residue_max"][0]:
        pen += 300 * (p["calc_yres"] - CONSTRAINTS["residue_max"][0]) ** 2
    if p["calc_visc_50"] > CONSTRAINTS["visc_max"][0]:
        pen += 200 * (p["calc_visc_50"] - CONSTRAINTS["visc_max"][0]) ** 2
    if risk > CONSTRAINTS["asphaltene_max"][0]:
        pen += 500 * (risk - CONSTRAINTS["asphaltene_max"][0]) ** 2
    return -grm + pen


def main():
    bounds = [(MIN_RATIO, MAX_RATIO)] * 3
    res = differential_evolution(
        objective, bounds, seed=SEED, maxiter=150, popsize=15, tol=1e-6,
        strategy="best1bin", mutation=(0.5, 1.5), recombination=0.9, polish=True)

    r4 = np.array(list(res.x) + [max(0.0, 1.0 - sum(res.x))])
    r4 = r4 / r4.sum()
    grm, risk, p = predict(r4)
    metals = p["calc_ni"] + p["calc_v"]

    print("\n" + "=" * 50)
    print("OPTIMAL BLEND RATIOS (Maximize GRM)")
    print("=" * 50)
    for name, frac in zip(DISPLAY, r4):
        print(f"  {name:<14}: {frac * 100:5.1f} vol%")
    print(f"\n  Predicted GRM   : ${grm:.2f}/bbl")
    print(f"  Feedstock Cost  : ${p['calc_feedstock_cost']:.2f}/bbl")
    print(f"  Blend API       : {p['calc_api']:.1f}")
    print(f"  Blend Sulphur   : {p['calc_sulphur']:.3f} wt%")
    print(f"  Asphaltene Risk : {risk:.3f} ({'HIGH' if risk > 0.4 else 'low'})")
    print(f"  Metals (Ni+V)   : {metals:.1f} mg/kg")
    print(f"  Residue Yield   : {p['calc_yres'] * 100:.1f}%")
    print(f"  Viscosity @50C  : {p['calc_visc_50']:.2f} cSt")
    print(f"  DE converged in : {res.nit} iterations")
    return r4, grm, risk, p


if __name__ == "__main__":
    main()
