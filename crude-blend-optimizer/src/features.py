"""
features.py
===========
Shared feature engineering used by BOTH training (s03/s04/s05) and the
optimizer / dashboard (s06/s09).

Design principle (the core of this project):
  * Directly calculable blend properties (API, sulphur, density, cost, ...)
    are computed ANALYTICALLY here with linear / ASTM mixing rules.
  * The ML models only ever PREDICT genuinely non-linear targets
    (GRM, asphaltene risk) -- never the analytic properties.

Keeping the feature builder in one place guarantees the optimizer and the
Streamlit app construct feature vectors identically to training.
"""

import numpy as np
import pandas as pd

from config import CRUDES, CNAMES, PRICES

# Final, ordered feature list fed to every model.
FEATURE_COLS = [
    # blend ratios
    "r_BonnyLight", "r_UpperZakhum", "r_BasrahMedium", "r_BasrahHeavy",
    # analytically computed intermediate properties (FEATURES, not targets)
    "calc_api", "calc_sulphur", "calc_density", "calc_tan",
    "calc_ni", "calc_v", "calc_visc_50", "calc_yres",
    "calc_feedstock_cost", "blend_solubility_param",
    # non-linear interaction features
    "sulphur_metals_interaction", "api_sulphur_ratio",
    "light_heavy_contrast", "heavy_sour_fraction", "vgo_to_residue_ratio",
]


def _w(prop):
    """Vector of a property across the 4 crudes, in CNAMES order."""
    return np.array([CRUDES[c][prop] for c in CNAMES], dtype=float)


def viscosity_blend_astm(ratios):
    """ASTM D7152 (Walther) log-log viscosity blending -- NON-linear."""
    visc = _w("visc_50")
    loglog = np.dot(ratios, np.log(np.log(visc + 0.7)))
    return np.exp(np.exp(loglog)) - 0.7


def analytic_props(ratios):
    """All directly calculable blend properties for a single ratio vector."""
    r = np.asarray(ratios, dtype=float)
    p = {}
    for prop in ["api", "sulphur", "density", "tan", "ni", "v",
                 "yn", "yk", "ygo", "yvgo", "yres"]:
        p["calc_" + prop] = float(r @ _w(prop))
    p["calc_visc_50"]          = float(viscosity_blend_astm(r))
    p["calc_feedstock_cost"]   = float(r @ np.array(PRICES, dtype=float))
    p["blend_solubility_param"] = float(r @ _w("solubility_param"))
    return p


def build_feature_dict(ratios):
    """Full ordered feature dict (analytic props + engineered interactions)."""
    r = np.asarray(ratios, dtype=float)
    p = analytic_props(r)
    feat = {
        "r_BonnyLight": r[0], "r_UpperZakhum": r[1],
        "r_BasrahMedium": r[2], "r_BasrahHeavy": r[3],
        "calc_api": p["calc_api"], "calc_sulphur": p["calc_sulphur"],
        "calc_density": p["calc_density"], "calc_tan": p["calc_tan"],
        "calc_ni": p["calc_ni"], "calc_v": p["calc_v"],
        "calc_visc_50": p["calc_visc_50"], "calc_yres": p["calc_yres"],
        "calc_feedstock_cost": p["calc_feedstock_cost"],
        "blend_solubility_param": p["blend_solubility_param"],
    }
    feat["sulphur_metals_interaction"] = p["calc_sulphur"] * np.sqrt(p["calc_ni"] + p["calc_v"])
    feat["api_sulphur_ratio"]          = p["calc_api"] / (p["calc_sulphur"] + 0.01)
    feat["light_heavy_contrast"]       = r[0] / (r[3] + 0.01)
    feat["heavy_sour_fraction"]        = r[2] + r[3]
    feat["vgo_to_residue_ratio"]       = p["calc_yvgo"] / (p["calc_yres"] + 0.001)
    return feat


def build_feature_frame(ratios):
    """Single-row DataFrame with FEATURE_COLS order -- avoids sklearn warnings."""
    return pd.DataFrame([build_feature_dict(ratios)])[FEATURE_COLS]


def add_engineered_columns(df):
    """Vectorized: add the 5 interaction features to a full dataset (s03)."""
    df = df.copy()
    df["sulphur_metals_interaction"] = df["calc_sulphur"] * np.sqrt(df["calc_ni"] + df["calc_v"])
    df["api_sulphur_ratio"]          = df["calc_api"] / (df["calc_sulphur"] + 0.01)
    df["light_heavy_contrast"]       = df["r_BonnyLight"] / (df["r_BasrahHeavy"] + 0.01)
    df["heavy_sour_fraction"]        = df["r_BasrahMedium"] + df["r_BasrahHeavy"]
    df["vgo_to_residue_ratio"]       = df["calc_yvgo"] / (df["calc_yres"] + 0.001)
    return df
