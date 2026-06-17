"""
SECTION 8 - Uncertainty & Risk Analysis
=======================================
8A  Monte Carlo on cargo assay variability -> GRM uncertainty band (P5/P95)
8B  Crude price scenario sensitivity

Depends on the optimal blend from s06.

Run:  python src/s08_uncertainty.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np

from config import GRM_MODEL, FIG_DIR, SEED, DISPLAY
from features import build_feature_dict, FEATURE_COLS
import s06_optimization as s06

GRM = joblib.load(GRM_MODEL)


def main():
    rng = np.random.default_rng(SEED)
    r4, opt_grm, opt_risk, p = s06.main()

    base = build_feature_dict(r4)

    # ---- 8A: assay variability Monte Carlo ------------------------------ #
    N_mc = 1000
    samples = []
    for _ in range(N_mc):
        f = dict(base)
        f["calc_api"]     = base["calc_api"] + rng.normal(0, 1.0)
        f["calc_sulphur"] = base["calc_sulphur"] + rng.normal(0, 0.15)
        row = [f[c] for c in FEATURE_COLS]
        samples.append(float(GRM.predict([row])[0]))
    mc = np.array(samples)

    print("\n=== GRM Uncertainty (Monte Carlo, N=%d) ===" % N_mc)
    print(f"  Mean GRM        : ${mc.mean():.2f}/bbl")
    print(f"  Std Dev         : ${mc.std():.2f}/bbl")
    print(f"  5th percentile  : ${np.percentile(mc, 5):.2f}/bbl")
    print(f"  95th percentile : ${np.percentile(mc, 95):.2f}/bbl")
    print(f"  P(GRM > 10 USD) : {(mc > 10).mean() * 100:.1f}%")

    plt.figure(figsize=(8, 4))
    plt.hist(mc, bins=50, color="steelblue", edgecolor="white")
    plt.axvline(mc.mean(), color="red", linestyle="--", label="Mean")
    plt.axvline(np.percentile(mc, 5), color="orange", linestyle=":", label="P5")
    plt.xlabel("Predicted GRM (USD/bbl)")
    plt.title("GRM Uncertainty from Cargo Assay Variability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "grm_uncertainty.png", dpi=150)
    plt.close()

    # ---- 8B: crude price sensitivity ------------------------------------ #
    scenarios = {
        "Base Case":     [84.50, 79.20, 72.80, 65.40],
        "BL Premium +5": [89.50, 79.20, 72.80, 65.40],
        "Iraq Discount": [84.50, 79.20, 68.80, 61.40],
        "All +10 USD":   [94.50, 89.20, 82.80, 75.40],
        "All -10 USD":   [74.50, 69.20, 62.80, 55.40],
    }
    base_cost = p["calc_feedstock_cost"]
    print("\n=== Crude Price Sensitivity ===")
    for name, prices in scenarios.items():
        cost = sum(r4[i] * prices[i] for i in range(4))
        grm = opt_grm + (base_cost - cost)   # product prices held constant
        print(f"  {name:<15}: feedstock ${cost:6.2f}/bbl | GRM ${grm:6.2f}/bbl")

    print(f"\n[s08] Wrote grm_uncertainty.png to {FIG_DIR}")


if __name__ == "__main__":
    main()
