"""
SECTION 5 - ML Model 2: Asphaltene Compatibility Risk Classifier
================================================================
Why non-trivial: asphaltene precipitation risk depends on the solubility-
parameter mismatch between crude PAIRS -- a non-linear pairwise interaction
that linear mixing rules cannot reproduce. The classifier learns a non-linear
surface over blend space, not a threshold rule.

Expected: ROC-AUC ~ 0.82-0.90.

Run:  python src/s05_model_asphaltene.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from config import FEATURES, RISK_MODEL, FIG_DIR, SEED
from features import FEATURE_COLS


def main():
    df = pd.read_csv(FEATURES)
    df["high_asphaltene_risk"] = (df["asphaltene_risk"] > 0.45).astype(int)

    X = df[FEATURE_COLS]
    y = df["high_asphaltene_risk"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        min_samples_leaf=8, random_state=SEED)
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)[:, 1]

    print("=== Asphaltene Compatibility Risk Classifier ===")
    print(classification_report(y_te, pred, target_names=["Low Risk", "High Risk"]))
    print(f"  ROC-AUC: {roc_auc_score(y_te, prob):.3f}")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=["Low Risk", "High Risk"], ax=ax, colorbar=False)
    plt.title("Asphaltene Risk Classifier")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "asphaltene_confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(clf, RISK_MODEL)
    print(f"[s05] Saved {RISK_MODEL}")


if __name__ == "__main__":
    main()
