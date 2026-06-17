"""
SECTION 7 - SHAP Explainability on the GRM Model
================================================
SHAP is applied to the GRM model -- a genuinely LEARNED non-linear
relationship, not a deterministic rule -- revealing which process variables and
crude ratios most influence refinery margin.

Requires: shap  (pip install shap).  Run:  python src/s07_shap_explainability.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

from config import FEATURES, GRM_MODEL, FIG_DIR, SEED
from features import FEATURE_COLS


def main():
    try:
        import shap
    except ImportError:
        print("[s07] shap not installed -> `pip install shap`. Skipping.")
        return

    df = pd.read_csv(FEATURES)
    X = df[FEATURE_COLS].sample(n=min(500, len(df)), random_state=SEED)
    model = joblib.load(GRM_MODEL)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. Global importance (bar)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS,
                      show=False, plot_type="bar")
    plt.title("SHAP Feature Importance -- GRM Prediction Model")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_grm_importance.png", dpi=150)
    plt.close()

    # 2. Beeswarm (direction + magnitude)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)
    plt.title("SHAP Values: Feature Effect on GRM")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_grm_beeswarm.png", dpi=150)
    plt.close()

    # 3. Dependence plot for the top feature
    top = FEATURE_COLS[int(np.abs(shap_values).mean(axis=0).argmax())]
    shap.dependence_plot(top, shap_values, X, feature_names=FEATURE_COLS, show=False)
    plt.title(f"SHAP Dependence: {top}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_dependence.png", dpi=150)
    plt.close()

    print(f"[s07] Top GRM driver: {top}")
    print(f"[s07] Wrote 3 SHAP figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
