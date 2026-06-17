"""
run_pipeline.py
===============
Runs the full project end-to-end (sections 1 -> 8). The Streamlit app (s09)
is launched separately with `streamlit run src/s09_app_streamlit.py`.

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import s01_data_generation
import s02_eda
import s03_feature_engineering
import s04_model_grm
import s05_model_asphaltene
import s06_optimization
import s07_shap_explainability
import s08_uncertainty


STEPS = [
    ("s01  data generation",   s01_data_generation.main),
    ("s02  EDA figures",       s02_eda.main),
    ("s03  feature engineering", s03_feature_engineering.main),
    ("s04  GRM model",         s04_model_grm.main),
    ("s05  asphaltene model",  s05_model_asphaltene.main),
    ("s06  optimization",      s06_optimization.main),
    ("s07  SHAP",              s07_shap_explainability.main),
    ("s08  uncertainty",       s08_uncertainty.main),
]


def main():
    for name, fn in STEPS:
        print("\n" + "#" * 70)
        print(f"# RUNNING {name}")
        print("#" * 70)
        fn()
    print("\nPipeline complete. Launch the dashboard with:")
    print("    streamlit run src/s09_app_streamlit.py")


if __name__ == "__main__":
    main()
