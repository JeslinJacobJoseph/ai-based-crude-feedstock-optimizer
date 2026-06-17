"""
SECTION 1 - Synthetic Data Generation
=====================================
Generates `data/crude_blend_dataset_v2.csv` (N=3000).

Design principle: directly calculable blend properties (API, sulphur, density,
cost) are computed ANALYTICALLY -- never used as ML targets. The ML targets are
genuinely non-linear / stochastic:

  TARGET 1  actual_grm          Gross Refinery Margin / bbl
  TARGET 2  cdu_preheat_duty    CDU preheat duty MJ/bbl (non-linear in viscosity)
  TARGET 3  asphaltene_risk     compatibility risk [0-1] (pairwise solubility)

Run:  python src/s01_data_generation.py
"""

import numpy as np
import pandas as pd

from config import (CRUDES, CNAMES, RATIO_COLS, PRICES, SEED, N, DATASET,
                    MIN_RATIO, MAX_RATIO, PRODUCT_PRICE_BASE, PRODUCT_PRICE_STD,
                    H2_PRICE_USD_KG, H2_KG_PER_BBL_PER_PCTS)
from features import viscosity_blend_astm


def w(prop):
    return np.array([CRUDES[c][prop] for c in CNAMES], dtype=float)


def sample_blend_ratios(n, rng):
    """Dirichlet sampling: the correct simplex sampler. 5-70% bounds enforced
    by REJECTION sampling (not clipping, which would distort the distribution)."""
    alpha = np.array([1.5, 1.5, 1.5, 1.5])
    out = np.empty((0, 4))
    while len(out) < n:
        cand = rng.dirichlet(alpha, size=n)
        ok = (cand >= MIN_RATIO).all(axis=1) & (cand <= MAX_RATIO).all(axis=1)
        out = np.vstack([out, cand[ok]])
    return out[:n]


def main():
    rng = np.random.default_rng(SEED)

    ratios = sample_blend_ratios(N, rng)
    df = pd.DataFrame(ratios, columns=RATIO_COLS)
    R = df[RATIO_COLS].values

    # ---- Step 1A: analytic blend properties (NOT ML targets) ------------ #
    for prop in ["api", "sulphur", "density", "tan", "ni", "v",
                 "yn", "yk", "ygo", "yvgo", "yres"]:
        df["calc_" + prop] = R @ w(prop)
    df["calc_visc_50"] = np.array([viscosity_blend_astm(r) for r in R])
    df["calc_feedstock_cost"] = R @ np.array(PRICES, dtype=float)

    # ---- Step 1B: cargo-level quality variation (uncertainty) ----------- #
    cargo_std = {"api": 1.0, "sulphur": 0.15, "tan": 0.03, "ni": 1.5, "v": 3.0}
    for prop, std in cargo_std.items():
        for i, cn in enumerate(CNAMES):
            df["cargo_" + cn + "_" + prop] = CRUDES[cn][prop] + rng.normal(0, std, N)
    for prop in ["api", "sulphur", "tan"]:
        df["actual_" + prop] = sum(
            df[RATIO_COLS[i]] * df["cargo_" + cn + "_" + prop]
            for i, cn in enumerate(CNAMES))

    # ---- Step 1C: non-linear ML targets --------------------------------- #
    prod_price = {p: PRODUCT_PRICE_BASE[p] + rng.normal(0, PRODUCT_PRICE_STD[p], N)
                  for p in PRODUCT_PRICE_BASE}
    df["theoretical_product_value"] = (
        df["calc_yn"] * prod_price["naphtha"] +
        df["calc_yk"] * prod_price["kerosene"] +
        df["calc_ygo"] * prod_price["gas_oil"] +
        df["calc_yvgo"] * prod_price["vgo"] +
        df["calc_yres"] * prod_price["residue"])

    s = df["calc_sulphur"].values
    metals = (df["calc_ni"] + df["calc_v"]).values
    tan = df["calc_tan"].values
    interaction = 0.008 * s * np.sqrt(metals) + 0.012 * tan * s
    df["yield_correction"] = 1.0 - interaction + rng.normal(0, 0.003, N)

    df["h2_cost"] = df["actual_sulphur"] * H2_KG_PER_BBL_PER_PCTS * H2_PRICE_USD_KG
    df["operating_cost"] = 4.5 + 0.3 * np.log1p(df["calc_visc_50"])
    df["actual_grm"] = (
        df["theoretical_product_value"] * df["yield_correction"]
        - df["calc_feedstock_cost"] - df["h2_cost"] - df["operating_cost"])

    sg = df["calc_density"] / 1000.0
    cp = (0.388 + 0.00045 * 180) / np.sqrt(sg)
    df["cdu_preheat_duty"] = (
        cp * 280 * df["calc_density"] / 159.0
        + 0.15 * np.log1p(df["calc_visc_50"]) + rng.normal(0, 0.8, N))

    sp = w("solubility_param")
    df["blend_solubility_param"] = R @ sp
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    sp_mismatch = sum(R[:, i] * R[:, j] * abs(sp[i] - sp[j]) for i, j in pairs)
    # Calibrate sigmoid to the mismatch distribution so risk spans [0,1] and the
    # c_asphaltene constraint is occasionally (not never/always) binding.
    center = np.quantile(sp_mismatch, 0.70)
    scale = 0.40 * sp_mismatch.std()
    df["asphaltene_risk"] = 1.0 / (1.0 + np.exp(-(sp_mismatch - center) / scale))
    df["asphaltene_risk"] = (df["asphaltene_risk"] + rng.normal(0, 0.03, N)).clip(0, 1)

    # ---- Step 1D: constraint labels (complete set) ---------------------- #
    df["c_api"] = df["actual_api"].between(26, 38)
    df["c_sulphur"] = df["actual_sulphur"] <= 3.0
    df["c_tan"] = df["actual_tan"] <= 0.5
    df["c_metals_fcc"] = (df["calc_ni"] + df["calc_v"]) <= 40
    df["c_residue"] = df["calc_yres"] <= 0.28
    df["c_visc_cdu"] = df["calc_visc_50"] <= 12.0
    df["c_asphaltene"] = df["asphaltene_risk"] <= 0.50
    df["c_h2_demand"] = df["actual_sulphur"] <= 3.5
    df["is_feasible"] = (
        df["c_api"] & df["c_sulphur"] & df["c_tan"] & df["c_metals_fcc"]
        & df["c_residue"] & df["c_visc_cdu"] & df["c_asphaltene"] & df["c_h2_demand"])

    df.to_csv(DATASET, index=False)
    pct_feas = 100.0 * df["is_feasible"].mean()
    n_high = int((df["asphaltene_risk"] > 0.45).sum())
    print("[s01] Wrote " + str(DATASET))
    print("[s01] Rows: {}  Cols: {}".format(len(df), df.shape[1]))
    print("[s01] Feasible blends: {} / {} ({:.1f} pct)".format(
        int(df["is_feasible"].sum()), N, pct_feas))
    print("[s01] High asphaltene risk (>0.45): {}".format(n_high))
    print("[s01] GRM range: {:.2f} .. {:.2f} USD/bbl".format(
        df["actual_grm"].min(), df["actual_grm"].max()))


if __name__ == "__main__":
    main()
