# =============================================================================
# Heart Disease Prediction — Streamlit Web Application
# Author: ML Engineering Team
# Run with: streamlit run app.py
# =============================================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* ── Global ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Gradient header ────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 50%, #e91e63 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(231,76,60,0.25);
    }
    .hero-banner h1 { font-size: 2.2rem; margin:0; font-weight:700; }
    .hero-banner p  { font-size: 1rem; margin:0.4rem 0 0; opacity:0.9; }

    /* ── Metric cards ────────────────────────────── */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #e74c3c;
        margin-bottom: 0.8rem;
    }
    .metric-card h4 { margin:0; color:#7f8c8d; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; }
    .metric-card p  { margin:0.3rem 0 0; font-size:1.8rem; font-weight:700; color:#2c3e50; }

    /* ── Section headers ─────────────────────────── */
    .section-header {
        font-size:1.1rem; font-weight:600; color:#2c3e50;
        padding-bottom:0.4rem;
        border-bottom: 2px solid #e74c3c;
        margin-bottom:1rem;
    }

    /* ── Risk banner ─────────────────────────────── */
    .risk-high {
        background:linear-gradient(135deg,#e74c3c,#c0392b);
        color:white; border-radius:14px; padding:1.8rem 2rem;
        text-align:center; box-shadow:0 6px 24px rgba(231,76,60,0.35);
    }
    .risk-low {
        background:linear-gradient(135deg,#27ae60,#1e8449);
        color:white; border-radius:14px; padding:1.8rem 2rem;
        text-align:center; box-shadow:0 6px 24px rgba(39,174,96,0.35);
    }
    .risk-high h2, .risk-low h2 { margin:0; font-size:1.8rem; }
    .risk-high p,  .risk-low p  { margin:0.3rem 0 0; opacity:0.9; }

    /* ── Input label ─────────────────────────────── */
    label { font-weight:500 !important; }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] { background:#1a1a2e; }
    section[data-testid="stSidebar"] * { color: #ecf0f1 !important; }

    /* ── Buttons ─────────────────────────────────── */
    .stButton>button {
        background:linear-gradient(135deg,#e74c3c,#c0392b);
        color:white; border:none; border-radius:10px;
        padding:0.7rem 2.5rem; font-size:1rem; font-weight:600;
        box-shadow:0 4px 15px rgba(231,76,60,0.4);
        transition:all 0.2s;
    }
    .stButton>button:hover {
        transform:translateY(-2px);
        box-shadow:0 6px 20px rgba(231,76,60,0.55);
    }
    /* Hide Streamlit defaults */
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    div[data-testid="stDecoration"] {display:none;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, and metadata from disk."""
    base = os.path.dirname(__file__)
    model_path  = os.path.join(base, "models", "best_model.pkl")
    scaler_path = os.path.join(base, "models", "scaler.pkl")
    meta_path   = os.path.join(base, "models", "model_metadata.json")

    if not os.path.exists(model_path):
        return None, None, {}

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta   = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return model, scaler, meta

model, scaler, meta = load_model_artifacts()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ❤️ Heart Disease AI")
    st.markdown("---")

    if meta:
        st.markdown("### 🤖 Model Info")
        st.info(f"**{meta.get('best_model_name','—')}**")
        cols = st.columns(2)
        cols[0].metric("Accuracy", f"{meta.get('accuracy',0):.1%}")
        cols[1].metric("F1 Score", f"{meta.get('f1_score',0):.1%}")
        cols[0].metric("AUC-ROC",  f"{meta.get('auc',0):.3f}")
        cols[1].metric("CV Score", f"{meta.get('cv_score',0):.1%}")
    else:
        st.warning("⚠️ Model not found.\nRun `train_model.py` first.")

    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown("""
This tool uses Machine Learning to assess heart disease risk based on clinical parameters.

> ⚕️ **Disclaimer:** This is for educational purposes only and does not replace professional medical advice.
""")
    st.markdown("---")
    st.markdown("### 🔗 Navigation")
    page = st.radio("Go to", ["🏠 Predict","📊 Dataset Info","📈 Model Results"])

# =============================================================================
# PAGE: PREDICT
# =============================================================================
if page == "🏠 Predict":
    st.markdown("""
    <div class='hero-banner'>
        <h1>❤️ Heart Disease Risk Predictor</h1>
        <p>Enter patient clinical parameters below to assess cardiovascular risk using AI</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("❌ Trained model not found. Please run `python train_model.py` first.")
        st.stop()

    # ── INPUT FORM ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📝 Patient Information</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age    = st.number_input("Age (years)",      min_value=20, max_value=100, value=45)
            sex    = st.selectbox("Sex",                 ["Male (1)", "Female (0)"])
            fbs    = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes (1)", "No (0)"])
            restecg = st.selectbox("Resting ECG Results",
                                   ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])

        with col2:
            st.markdown("**Vital Signs**")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
            chol     = st.number_input("Serum Cholesterol (mg/dl)",       min_value=100, max_value=600, value=200)
            thalach  = st.number_input("Max Heart Rate Achieved",          min_value=60, max_value=220, value=150)
            oldpeak  = st.number_input("ST Depression (Oldpeak)",          min_value=0.0, max_value=8.0,
                                       value=1.0, step=0.1, format="%.1f")

        with col3:
            st.markdown("**Clinical Tests**")
            cp    = st.selectbox("Chest Pain Type",
                                 ["Typical Angina (0)", "Atypical Angina (1)",
                                  "Non-anginal Pain (2)", "Asymptomatic (3)"])
            exang = st.selectbox("Exercise Induced Angina", ["Yes (1)", "No (0)"])
            slope = st.selectbox("Slope of Peak ST Segment",
                                 ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
            ca    = st.selectbox("Major Vessels Colored (0–3)", ["0","1","2","3"])
            thal  = st.selectbox("Thalassemia",
                                 ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Unknown (3)"])

        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)

    # ── VALIDATION & PREDICTION ───────────────────────────────────────────────
    if submitted:
        # Parse dropdown values (extract the trailing integer)
        def parse_val(s):
            return int(s.split("(")[-1].replace(")",""))

        features = {
            "age":      age,
            "sex":      parse_val(sex),
            "cp":       parse_val(cp),
            "trestbps": trestbps,
            "chol":     chol,
            "fbs":      parse_val(fbs),
            "restecg":  parse_val(restecg),
            "thalach":  thalach,
            "exang":    parse_val(exang),
            "oldpeak":  oldpeak,
            "slope":    parse_val(slope),
            "ca":       int(ca),
            "thal":     parse_val(thal),
        }

        # Validation
        errors = []
        if trestbps < 60 or trestbps > 220:
            errors.append("Blood pressure seems outside normal range (60–220 mm Hg).")
        if chol < 100 or chol > 600:
            errors.append("Cholesterol seems outside normal range (100–600 mg/dl).")
        if thalach < 40 or thalach > 220:
            errors.append("Max heart rate outside valid range (40–220 bpm).")

        if errors:
            for e in errors:
                st.warning(f"⚠️ {e}")

        input_df  = pd.DataFrame([features])
        input_s   = scaler.transform(input_df)
        prediction  = model.predict(input_s)[0]
        probability = model.predict_proba(input_s)[0]
        prob_disease = probability[1]
        prob_healthy = probability[0]

        st.markdown("---")
        st.markdown("### 🏥 Prediction Result")

        r1, r2 = st.columns([2, 1])
        with r1:
            if prediction == 1:
                risk_pct = f"{prob_disease*100:.1f}%"
                if prob_disease >= 0.75:
                    risk_level = "HIGH RISK"
                    emoji = "🔴"
                elif prob_disease >= 0.5:
                    risk_level = "MODERATE RISK"
                    emoji = "🟠"
                else:
                    risk_level = "LOW-MODERATE RISK"
                    emoji = "🟡"
                st.markdown(f"""
                <div class='risk-high'>
                    <h2>{emoji} Heart Disease Detected</h2>
                    <p style='font-size:1.2rem;font-weight:600;margin-top:0.5rem;'>{risk_level}</p>
                    <p>Probability: <strong>{risk_pct}</strong></p>
                    <p style='font-size:0.9rem;margin-top:0.5rem;'>Please consult a cardiologist immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                risk_pct = f"{prob_healthy*100:.1f}%"
                st.markdown(f"""
                <div class='risk-low'>
                    <h2>✅ No Heart Disease Detected</h2>
                    <p style='font-size:1.2rem;font-weight:600;margin-top:0.5rem;'>LOW RISK</p>
                    <p>Healthy Probability: <strong>{risk_pct}</strong></p>
                    <p style='font-size:0.9rem;margin-top:0.5rem;'>Maintain a healthy lifestyle.</p>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_disease * 100,
                number={"suffix": "%", "font": {"size": 32}},
                title={"text": "Disease Probability", "font": {"size": 14}},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#e74c3c" if prediction == 1 else "#27ae60"},
                    "steps": [
                        {"range": [0, 40],   "color": "#d5f5e3"},
                        {"range": [40, 70],  "color": "#fdebd0"},
                        {"range": [70, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "#2c3e50", "width": 3}, "value": 50},
                },
            ))
            fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20),
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#2c3e50")
            st.plotly_chart(fig, use_container_width=True)

        # Probability bar
        st.markdown("#### Prediction Confidence")
        p1, p2 = st.columns(2)
        with p1:
            fig2 = go.Figure(go.Bar(
                x=["No Disease","Heart Disease"],
                y=[prob_healthy*100, prob_disease*100],
                marker_color=["#27ae60","#e74c3c"],
                text=[f"{prob_healthy*100:.1f}%", f"{prob_disease*100:.1f}%"],
                textposition="auto",
            ))
            fig2.update_layout(title="Probability Breakdown", yaxis_title="%",
                               height=300, margin=dict(t=40,b=20,l=20,r=20),
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        with p2:
            st.markdown("#### 🔍 Input Summary")
            summary_df = pd.DataFrame.from_dict(features, orient="index", columns=["Value"])
            st.dataframe(summary_df, use_container_width=True)

        # Recommendations
        st.markdown("---")
        st.markdown("#### 💡 Health Recommendations")
        recs = []
        if chol > 200:   recs.append("🟡 **High Cholesterol** — Consider dietary changes and consult a physician.")
        if trestbps > 130: recs.append("🔴 **Elevated Blood Pressure** — Monitor regularly; reduce sodium intake.")
        if age > 55:     recs.append("🟠 **Age Factor** — Regular cardiac checkups are recommended over 55.")
        if parse_val(sex) == 1 and age > 45: recs.append("ℹ️ **Risk Factor** — Males over 45 have higher cardiac risk.")
        if not recs:     recs.append("✅ All parameters appear within acceptable ranges.")
        for r in recs:
            st.markdown(r)

# =============================================================================
# PAGE: DATASET INFO
# =============================================================================
elif page == "📊 Dataset Info":
    st.markdown("""
    <div class='hero-banner'>
        <h1>📊 Dataset Information</h1>
        <p>Cleveland Heart Disease Dataset — Column Descriptions & Statistics</p>
    </div>
    """, unsafe_allow_html=True)

    col_desc = {
        "age":      ("Age","Continuous","29–77 years","Patient age in years"),
        "sex":      ("Sex","Binary","0=Female, 1=Male","Gender of the patient"),
        "cp":       ("Chest Pain","Categorical","0–3","0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic"),
        "trestbps": ("Blood Pressure","Continuous","94–200 mm Hg","Resting blood pressure on admission"),
        "chol":     ("Cholesterol","Continuous","126–564 mg/dl","Serum cholesterol level"),
        "fbs":      ("Blood Sugar","Binary","0=No, 1=Yes","Fasting blood sugar > 120 mg/dl"),
        "restecg":  ("ECG Result","Categorical","0,1,2","Resting electrocardiogram results"),
        "thalach":  ("Max HR","Continuous","71–202 bpm","Maximum heart rate achieved"),
        "exang":    ("Exercise Angina","Binary","0=No, 1=Yes","Exercise-induced angina"),
        "oldpeak":  ("ST Depression","Continuous","0–6.2","ST depression relative to rest"),
        "slope":    ("ST Slope","Categorical","0–2","0=Up, 1=Flat, 2=Down"),
        "ca":       ("Vessels","Ordinal","0–3","Major vessels colored by fluoroscopy"),
        "thal":     ("Thalassemia","Categorical","0–3","Blood disorder type"),
        "target":   ("Target","Binary","0=No, 1=Yes","⭐ Heart disease presence (target variable)"),
    }

    df_desc = pd.DataFrame(col_desc, index=["Feature","Type","Range","Description"]).T.reset_index()
    df_desc.columns = ["Column","Feature","Type","Range","Description"]
    st.markdown("### 📋 Column Reference")
    st.dataframe(df_desc, use_container_width=True, height=500)

    st.markdown("---")
    st.markdown("### 📈 Quick Statistics")
    ds_path = "dataset/heart.csv"
    if os.path.exists(ds_path):
        df = pd.read_csv(ds_path)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Patients", df.shape[0])
        c2.metric("Features",       df.shape[1]-1)
        c3.metric("Disease Cases",  int(df["target"].sum()))
        c4.metric("Healthy Cases",  int((df["target"]==0).sum()))
        st.markdown("#### Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Run `train_model.py` to generate the dataset.")

# =============================================================================
# PAGE: MODEL RESULTS
# =============================================================================
elif page == "📈 Model Results":
    st.markdown("""
    <div class='hero-banner'>
        <h1>📈 Model Performance</h1>
        <p>Comparing all trained models — metrics and visualizations</p>
    </div>
    """, unsafe_allow_html=True)

    shots = "screenshots"
    images = {
        "Disease Distribution":    "01_disease_distribution.png",
        "Age Distribution":        "02_age_distribution.png",
        "Cholesterol Analysis":    "03_cholesterol_distribution.png",
        "Correlation Heatmap":     "04_correlation_heatmap.png",
        "Feature Importance":      "05_feature_importance.png",
        "Confusion Matrices":      "06_confusion_matrices.png",
        "ROC Curves":              "07_roc_curves.png",
        "Model Comparison":        "08_model_comparison.png",
    }

    for title, fname in images.items():
        fpath = os.path.join(shots, fname)
        if os.path.exists(fpath):
            st.markdown(f"#### {title}")
            st.image(fpath, use_column_width=True)
            st.markdown("---")
        else:
            st.info(f"⚠️ `{fname}` not found — run `train_model.py` to generate plots.")
