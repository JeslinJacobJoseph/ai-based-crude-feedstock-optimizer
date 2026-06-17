"""
SECTION 9 - Deployment: Streamlit Dashboard
===========================================
Manual blend tester. Clearly separates:
  * COMPUTED properties (analytic mixing rules)  -- API, sulphur, cost, ...
  * ML-PREDICTED quantities (non-linear)         -- GRM, asphaltene risk

Run:  streamlit run src/s09_app_streamlit.py
"""

import os
import sys

import joblib
import streamlit as st

# allow `import config / features` when launched from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CRUDES, GRM_MODEL, RISK_MODEL  # noqa: E402
from features import build_feature_frame, analytic_props  # noqa: E402

st.set_page_config(page_title="Crude Blend Optimizer", layout="wide")
st.title("Crude Blend Ratio Optimizer -- GRM Maximization")
st.caption("Real assay data: Bonny Light, Upper Zakhum, Basrah Medium, Basrah Heavy")


@st.cache_resource
def load_models():
    return joblib.load(GRM_MODEL), joblib.load(RISK_MODEL)


grm_model, risk_model = load_models()

# ---- sidebar: blend + constraints --------------------------------------- #
st.sidebar.header("Manual Blend Test")
r1 = st.sidebar.slider("Bonny Light (vol%)", 5, 70, 25) / 100
r2 = st.sidebar.slider("Upper Zakhum (vol%)", 5, 70, 25) / 100
r3 = st.sidebar.slider("Basrah Medium (vol%)", 5, 60, 25) / 100
r4 = max(0.0, 1.0 - r1 - r2 - r3)
st.sidebar.write(f"**Basrah Heavy (auto): {r4 * 100:.1f} vol%**")

st.sidebar.header("Constraints")
max_sulphur = st.sidebar.slider("Max Sulphur (wt%)", 0.5, 4.0, 3.0, 0.1)
max_metals = st.sidebar.slider("Max Ni+V (mg/kg)", 10, 60, 40, 5)
max_risk = st.sidebar.slider("Max Asphaltene Risk", 0.1, 0.9, 0.5, 0.05)

ratios = [r1, r2, r3, r4]
p = analytic_props(ratios)                       # analytic (NOT ML)
X = build_feature_frame(ratios)
pred_grm = float(grm_model.predict(X)[0])        # ML
pred_risk = float(risk_model.predict_proba(X)[0, 1])  # ML

# ---- metrics ------------------------------------------------------------ #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Blend API", f"{p['calc_api']:.1f}", help="Computed (mixing rule)")
c2.metric("Sulphur (wt%)", f"{p['calc_sulphur']:.3f}", help="Computed (mixing rule)")
c3.metric("Cost (USD/bbl)", f"${p['calc_feedstock_cost']:.2f}", help="Computed (mixing rule)")
c4.metric("Predicted GRM", f"${pred_grm:.2f}", help="ML predicted (non-linear)")
c5.metric("Asphalt. Risk", f"{pred_risk:.2f}", help="ML predicted (non-linear)")

# ---- constraint check --------------------------------------------------- #
metals = p["calc_ni"] + p["calc_v"]
constraints = {
    "API (26-38)": 26 <= p["calc_api"] <= 38,
    "Sulphur <= max": p["calc_sulphur"] <= max_sulphur,
    "TAN <= 0.5": p["calc_tan"] <= 0.5,
    f"Ni+V <= {max_metals}": metals <= max_metals,
    "Residue <= 28%": p["calc_yres"] <= 0.28,
    "Viscosity <= 12 cSt": p["calc_visc_50"] <= 12.0,
    f"Asphalt Risk <= {max_risk}": pred_risk <= max_risk,
}

if all(constraints.values()):
    st.success("All constraints satisfied")
else:
    violated = [k for k, ok in constraints.items() if not ok]
    st.error("Violated: " + ", ".join(violated))

for label, ok in constraints.items():
    st.write(f"{'[OK]' if ok else '[X] '} {label}")
