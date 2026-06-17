# AI-Based Crude Blend Ratio Optimization

Recommend the optimal crude blend ratio (vol%) to **maximize Gross Refinery Margin (GRM)** subject to CDU, FCC, HDS and desalter constraints — using real crude assay properties for **Bonny Light**, **Upper Zakhum**, **Basrah Medium** and **Basrah Heavy**.

> Portfolio project for: Process Systems Engineering · Industrial AI · Refinery Analytics
> Stack: Python · scikit-learn · SciPy (Differential Evolution) · SHAP · Streamlit

---

## The core idea: ML only does what mixing rules can't

Blend API, sulphur, density and feedstock cost are **directly calculable** from the blend ratios with linear mixing rules — so they are computed analytically and used as *features*, never as ML targets. ML is reserved for quantities that are genuinely non-linear or stochastic:

| ML target | Why it needs ML |
|-----------|-----------------|
| `actual_grm` (GRM/bbl) | Market product-price variability + non-linear yield correction + non-linear H₂/operating costs |
| `cdu_preheat_duty` | Non-linear in viscosity, density, flow regime |
| `asphaltene_risk` | Pairwise solubility-parameter mismatch between crudes — not a linear mix |
| `calc_visc_50` | ASTM D7152 (Walther) log-log blending — non-linear |

This is the project's headline talking point: **R² of 0.78–0.88 on a genuinely non-linear target is more credible and more impressive than R² ≈ 0.99 on a trivially calculable one.**

---

## Repository structure

```
crude-blend-optimizer/
├── README.md
├── requirements.txt
├── run_pipeline.py            # runs sections 1–8 end-to-end
├── .gitignore
├── data/                      # generated datasets (git-ignored)
├── models/                    # trained .pkl models (git-ignored)
├── reports/figures/           # generated plots (git-ignored)
└── src/
    ├── config.py              # assay anchors, prices, constraints, paths
    ├── features.py            # shared analytic + engineered feature builder
    ├── s01_data_generation.py # SECTION 1  synthetic data (Dirichlet sampling)
    ├── s02_eda.py             # SECTION 2  EDA figures
    ├── s03_feature_engineering.py # SECTION 3  interaction features
    ├── s04_model_grm.py       # SECTION 4  GRM regressor (GradientBoosting)
    ├── s05_model_asphaltene.py# SECTION 5  asphaltene risk classifier (RandomForest)
    ├── s06_optimization.py    # SECTION 6  Differential Evolution optimizer
    ├── s07_shap_explainability.py # SECTION 7  SHAP on the GRM model
    ├── s08_uncertainty.py     # SECTION 8  Monte Carlo + price sensitivity
    └── s09_app_streamlit.py   # SECTION 9  Streamlit dashboard
```

`config.py` and `features.py` are shared modules so the assay numbers and the feature vector are defined **once** — training, the optimizer and the dashboard all build features identically.

---

## Quick start

```bash
pip install -r requirements.txt        # add --break-system-packages on some Linux setups

# run the whole pipeline (data → EDA → models → optimize → SHAP → uncertainty)
python run_pipeline.py

# or run a single section
python src/s01_data_generation.py
python src/s04_model_grm.py
python src/s06_optimization.py

# launch the dashboard
streamlit run src/s09_app_streamlit.py
```

`run_pipeline.py` writes the dataset to `data/`, models to `models/`, and figures to `reports/figures/`.

---

## What each section does

**1 — Data generation.** 3000 blends sampled with a **Dirichlet** distribution (the correct simplex sampler — no clip/renormalize distortion), 5–70% per-crude bounds via rejection sampling. Adds cargo-level assay variability and market price noise so the targets are non-trivially learnable.

**2 — EDA.** GRM distribution, feedstock-cost-vs-GRM trade-off, asphaltene risk vs heavy-crude ratio, constraint-binding frequency, and the yield-correction non-linearity that justifies ML.

**3 — Feature engineering.** Adds 5 interaction features: `sulphur_metals_interaction`, `api_sulphur_ratio`, `light_heavy_contrast`, `heavy_sour_fraction`, `vgo_to_residue_ratio`.

**4 — GRM model.** `GradientBoostingRegressor` (n=300, depth=4, lr=0.05, subsample=0.8). Expected **R² ≈ 0.78–0.88, MAE ≈ 1.0–1.5 USD/bbl**, with 5-fold CV.

**5 — Asphaltene risk.** `RandomForestClassifier` learning the non-linear compatibility surface. Expected **ROC-AUC ≈ 0.82–0.90**.

**6 — Optimization.** **Differential Evolution** (global, derivative-free) treats the trained GRM model as a black-box oracle, with an exterior penalty method enforcing all 8 constraints. (Correctly labelled DE — not Bayesian Optimization.)

**7 — SHAP.** Applied to the *learned* GRM model (bar, beeswarm, dependence) — not to deterministic rules.

**8 — Uncertainty.** Monte Carlo over cargo assay variability gives a P5/P95 GRM band; a price-scenario sweep shows margin sensitivity.

**9 — Dashboard.** Streamlit app that clearly separates **computed** (analytic) from **ML-predicted** (non-linear) quantities and live-checks constraints.

---

## Refinery constraints

| Constraint | Limit | Engineering basis |
|------------|-------|-------------------|
| Blend API | 26–38 | CDU design envelope |
| Blend sulphur | ≤ 3.0 wt% | HDS hydrogen capacity |
| TAN | ≤ 0.5 mg KOH/g | High-temp corrosion (Nelson curves) |
| Ni + V | ≤ 40 mg/kg | FCC catalyst deactivation |
| Residue yield | ≤ 28 vol% | Fuel oil market absorption |
| Viscosity @ 50 °C | ≤ 12 cSt | CDU preheat train limit |
| Asphaltene risk | ≤ 0.50 | Desalter / CDU compatibility |
| H₂ demand proxy | sulphur ≤ 3.5 wt% | Hydroprocessing H₂ availability |

---

## Models & metrics

| Model | Task | Target | Expected metric |
|-------|------|--------|-----------------|
| GradientBoostingRegressor | GRM prediction | `actual_grm` (non-linear) | R² 0.78–0.88, MAE ~1.0–1.5 USD/bbl |
| RandomForestClassifier | Asphaltene risk | non-linear compatibility | ROC-AUC 0.82–0.90 |
| Differential Evolution | GRM maximization | optimal blend ratios | converges < 500 iters |
| Monte Carlo | GRM uncertainty | P5/P95 band | quantified risk band |

---

## Crude assay anchors

| Crude | Origin | API | Sulphur (wt%) | Density (kg/m³) | Price (USD/bbl) |
|-------|--------|-----|---------------|-----------------|-----------------|
| Bonny Light | Nigeria | 34.9 | 0.151 | 850.04 | 84.50 |
| Upper Zakhum | UAE | 33.9 | 1.82 | 855.11 | 79.20 |
| Basrah Medium | Iraq | 28.2 | 3.54 | 868.50 | 72.80 |
| Basrah Heavy | Iraq | 23.7 | 4.12 | 880.00 | 65.40 |

> The synthetic dataset is calibrated to these real assay values. Replace `CRUDES` in `src/config.py` with your own assay data to retarget the project.
