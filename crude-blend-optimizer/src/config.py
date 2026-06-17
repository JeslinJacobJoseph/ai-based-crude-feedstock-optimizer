"""
config.py -- central configuration for the Crude Blend Ratio Optimizer.

Holds the real crude assay anchors, refinery product prices, constraint limits,
and all file paths. Every other script imports from here so the assay numbers
live in ONE place.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (repo-relative, created automatically on import)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
for _d in (DATA_DIR, MODEL_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET = DATA_DIR / "crude_blend_dataset_v2.csv"
FEATURES = DATA_DIR / "crude_blend_features.csv"
GRM_MODEL = MODEL_DIR / "model_grm.pkl"
RISK_MODEL = MODEL_DIR / "model_asphaltene_risk.pkl"

# --------------------------------------------------------------------------- #
# Reproducibility / dataset size
# --------------------------------------------------------------------------- #
SEED = 42
N = 3000

# --------------------------------------------------------------------------- #
# Real crude property dictionary (assay anchors).
#   yn/yk/ygo/yvgo/yres = TBP volume yields (naphtha, kero, gas oil, VGO, residue)
#   solubility_param     = Hildebrand-type proxy for asphaltene compatibility
# --------------------------------------------------------------------------- #
CRUDES = {
    "BonnyLight": {  # Nigeria, assay June 2020
        "api": 34.9, "sulphur": 0.151, "density": 850.04, "tan": 0.26,
        "ni": 3.62, "v": 0.49, "visc_50": 3.0,
        "yn": 0.210, "yk": 0.140, "ygo": 0.220, "yvgo": 0.250, "yres": 0.180,
        "price": 84.50, "solubility_param": 8.2},
    "UpperZakhum": {  # UAE, assay Jan 2012
        "api": 33.9, "sulphur": 1.82, "density": 855.11, "tan": 0.051,
        "ni": 9.78, "v": 11.0, "visc_50": 4.47,
        "yn": 0.180, "yk": 0.130, "ygo": 0.220, "yvgo": 0.260, "yres": 0.210,
        "price": 79.20, "solubility_param": 8.5},
    "BasrahMedium": {  # Iraq, assay Feb 2024
        "api": 28.2, "sulphur": 3.54, "density": 868.50, "tan": 0.10,
        "ni": 15.0, "v": 35.0, "visc_50": 8.5,
        "yn": 0.162, "yk": 0.131, "ygo": 0.215, "yvgo": 0.252, "yres": 0.212,
        "price": 72.80, "solubility_param": 9.1},
    "BasrahHeavy": {  # Iraq, assay Jun 2015
        "api": 23.7, "sulphur": 4.12, "density": 880.00, "tan": 0.15,
        "ni": 22.0, "v": 58.0, "visc_50": 14.0,
        "yn": 0.138, "yk": 0.126, "ygo": 0.206, "yvgo": 0.263, "yres": 0.251,
        "price": 65.40, "solubility_param": 9.8},
}

CNAMES = list(CRUDES.keys())
DISPLAY = ["Bonny Light", "Upper Zakhum", "Basrah Medium", "Basrah Heavy"]
RATIO_COLS = ["r_BonnyLight", "r_UpperZakhum", "r_BasrahMedium", "r_BasrahHeavy"]
PRICES = [CRUDES[c]["price"] for c in CNAMES]

# Blend ratio bounds (logistics / practical blending)
MIN_RATIO = 0.05
MAX_RATIO = 0.70

# --------------------------------------------------------------------------- #
# Refinery product prices (USD/bbl) + market volatility (std) for the GRM target.
# Base prices reflect upgraded product values (naphtha->gasoline, gas oil->diesel)
# so modelled GRM is positive single-digit USD/bbl, as in a real refinery.
# Volatility is modest vs composition-driven margin -> credible, learnable R^2.
# --------------------------------------------------------------------------- #
PRODUCT_PRICE_BASE = {
    "naphtha": 91.0, "kerosene": 116.0, "gas_oil": 113.0, "vgo": 98.0, "residue": 58.0}
PRODUCT_PRICE_STD = {
    "naphtha": 2.0, "kerosene": 2.5, "gas_oil": 2.2, "vgo": 1.5, "residue": 1.2}

H2_PRICE_USD_KG = 2.80
H2_KG_PER_BBL_PER_PCTS = 0.18  # kg H2 / bbl per 1 wt% sulphur (rule of thumb)

# --------------------------------------------------------------------------- #
# Complete refinery constraint set (limit, engineering basis)
# --------------------------------------------------------------------------- #
CONSTRAINTS = {
    "api_min":      (26.0, "CDU design envelope lower"),
    "api_max":      (38.0, "CDU design envelope upper"),
    "sulphur_max":  (3.0,  "HDS unit hydrogen capacity"),
    "tan_max":      (0.5,  "High-temperature corrosion (Nelson curves)"),
    "metals_max":   (40.0, "FCC equilibrium catalyst deactivation (Ni+V)"),
    "residue_max":  (0.28, "Fuel oil market absorption limit"),
    "visc_max":     (12.0, "CDU preheat train viscosity limit @ 50C"),
    "asphaltene_max": (0.50, "Desalter / CDU compatibility"),
    "h2_demand_max": (3.5, "Hydroprocessing hydrogen availability (sulphur proxy)"),
}
