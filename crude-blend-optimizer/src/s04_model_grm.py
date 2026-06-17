"""
SECTION 4 - ML Model 1: GRM Prediction
======================================
Target: actual_grm -- a genuinely non-linear / stochastic quantity because it
includes market product-price variability, a non-linear yield correction, and
non-linear H2 / operating cost terms.

Expected: R^2 ~ 0.78-0.88, MAE ~ 1.0-1.5 USD/bbl  (credible, NOT inflated ~1.0).

Run:  python src/s04_model_grm.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from config import FEATURES, GRM_MODEL, SEED
from features import FEATURE_COLS


def main():
    df = pd.read_csv(FEATURES)
    X = df[FEATURE_COLS]
    y = df["actual_grm"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    # Hyperparameters chosen by domain reasoning (see README):
    #   max_depth=4         -> limits complexity, avoids memorising noise
    #   learning_rate=0.05  -> stable convergence with 300 trees
    #   subsample=0.8       -> stochastic gradient boosting, less overfitting
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=SEED)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    cv = cross_val_score(model, X, y,
                         cv=KFold(5, shuffle=True, random_state=SEED),
                         scoring="r2").mean()

    print("=== GRM Prediction (GradientBoostingRegressor) ===")
    print(f"  MAE   : {mae:.3f} USD/bbl")
    print(f"  R^2   : {r2:.3f}")
    print(f"  CV-R^2: {cv:.3f}   (5-fold)")

    joblib.dump(model, GRM_MODEL)
    print(f"[s04] Saved {GRM_MODEL}")


if __name__ == "__main__":
    main()
