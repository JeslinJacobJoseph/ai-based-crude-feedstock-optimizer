"""
SECTION 3 - Feature Engineering
===============================
Adds the 5 non-linear interaction features to the raw dataset and writes
`data/crude_blend_features.csv`, the table the model scripts consume.

Engineered features:
  sulphur_metals_interaction = sulphur * sqrt(Ni+V)
  api_sulphur_ratio          = API / (sulphur + 0.01)
  light_heavy_contrast       = BonnyLight / (BasrahHeavy + 0.01)
  heavy_sour_fraction        = BasrahMedium + BasrahHeavy
  vgo_to_residue_ratio       = VGO yield / (residue yield + 0.001)

Run:  python src/s03_feature_engineering.py
"""

import pandas as pd

from config import DATASET, FEATURES
from features import add_engineered_columns, FEATURE_COLS


def main():
    df = pd.read_csv(DATASET)
    df = add_engineered_columns(df)
    df.to_csv(FEATURES, index=False)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert not missing, f"missing feature columns: {missing}"

    print(f"[s03] Wrote {FEATURES}")
    print(f"[s03] Feature matrix shape: {df[FEATURE_COLS].shape}")
    print(f"[s03] Features: {FEATURE_COLS}")


if __name__ == "__main__":
    main()
