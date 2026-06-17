"""
SECTION 2 - Exploratory Data Analysis
=====================================
Reads the dataset and writes 5 figures to reports/figures/.

Run:  python src/s02_eda.py
"""

import matplotlib
matplotlib.use("Agg")           # headless / CI-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATASET, FIG_DIR


def main():
    df = pd.read_csv(DATASET)

    # 1. GRM distribution + feasibility comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["actual_grm"], bins=60, color="steelblue", edgecolor="white")
    axes[0].set_title("GRM Distribution (USD/bbl)")
    axes[0].set_xlabel("GRM (USD/bbl)")
    df.boxplot(column="actual_grm", by="is_feasible", ax=axes[1])
    axes[1].set_title("GRM by Feasibility")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "grm_distribution.png", dpi=150)
    plt.close()

    # 2. Feedstock cost vs GRM (cheaper crude != better margin)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["calc_feedstock_cost"], df["actual_grm"],
                     c=df["actual_sulphur"], cmap="RdYlGn_r", alpha=0.4, s=8)
    plt.colorbar(sc, label="Blend Sulphur (wt%)")
    plt.xlabel("Feedstock Cost (USD/bbl)")
    plt.ylabel("GRM (USD/bbl)")
    plt.title("Feedstock Cost vs GRM\n(cheaper crude != better margin: H2 + yield effects)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cost_vs_grm.png", dpi=150)
    plt.close()

    # 3. Asphaltene risk vs heavy crude ratio
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(df["r_BasrahHeavy"], df["asphaltene_risk"],
                     c=df["r_BonnyLight"], cmap="RdYlGn", alpha=0.3, s=6)
    plt.colorbar(sc, label="Bonny Light Ratio")
    plt.xlabel("Basrah Heavy Ratio")
    plt.ylabel("Asphaltene Compatibility Risk")
    plt.title("Asphaltene Risk vs Heavy Crude Ratio")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "asphaltene_risk.png", dpi=150)
    plt.close()

    # 4. Constraint binding frequency
    cons = ["c_api", "c_sulphur", "c_tan", "c_metals_fcc",
            "c_residue", "c_visc_cdu", "c_asphaltene", "c_h2_demand"]
    viol = {c: 1 - df[c].mean() for c in cons}
    plt.figure(figsize=(8, 4))
    plt.bar(list(viol.keys()), list(viol.values()), color="tomato")
    plt.ylabel("Violation Rate")
    plt.title("Which Constraints Are Most Binding?")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "constraint_binding.png", dpi=150)
    plt.close()

    # 5. Yield correction vs sulphur-metals interaction (justifies ML)
    plt.figure(figsize=(8, 5))
    plt.scatter(df["actual_sulphur"] * np.sqrt(df["calc_ni"] + df["calc_v"]),
                df["yield_correction"], alpha=0.3, s=6, color="steelblue")
    plt.xlabel("Sulphur x sqrt(Ni+V) Interaction Term")
    plt.ylabel("Yield Correction Factor")
    plt.title("Non-linear Yield Correction -- Justifies ML over Simple Equations")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "yield_correction_nonlinearity.png", dpi=150)
    plt.close()

    print(f"[s02] Wrote 5 EDA figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
