import streamlit as st
import numpy as np
import joblib
import os
 
# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .result-box-positive {
        background: linear-gradient(135deg, #ffe0e0, #ffb3b3);
        border-left: 5px solid #e03c3c;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-box-negative {
        background: linear-gradient(135deg, #e0f7e9, #b3f0cb);
        border-left: 5px solid #27ae60;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stSlider > div > div > div { background: #6c63ff !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("model/parkinsons_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feats  = joblib.load("model/feature_names.pkl")
    return model, scaler, feats
 
try:
    model, scaler, FEATURES = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
 
# ── Feature metadata (min, max, default, description) ────────
FEATURE_INFO = {
    "MDVP:Fo(Hz)":   (88.0,  260.0, 154.2, "Average vocal fundamental frequency"),
    "MDVP:Fhi(Hz)":  (102.0, 592.0, 197.1, "Maximum vocal fundamental frequency"),
    "MDVP:Flo(Hz)":  (65.0,  240.0, 116.3, "Minimum vocal fundamental frequency"),
    "MDVP:Jitter(%)": (0.001, 0.033, 0.006, "MDVP jitter (%)"),
    "MDVP:Jitter(Abs)": (0.000007, 0.00026, 0.000044, "MDVP absolute jitter"),
    "MDVP:RAP":      (0.0003, 0.021, 0.003, "MDVP relative amplitude perturbation"),
    "MDVP:PPQ":      (0.0003, 0.020, 0.003, "MDVP pitch period perturbation quotient"),
    "Jitter:DDP":    (0.0009, 0.066, 0.010, "Average absolute difference of jitter"),
    "MDVP:Shimmer":  (0.009,  0.119, 0.029, "MDVP local shimmer"),
    "MDVP:Shimmer(dB)": (0.085, 1.302, 0.282, "MDVP shimmer in dB"),
    "Shimmer:APQ3":  (0.004,  0.056, 0.015, "Three-point amplitude perturbation quotient"),
    "Shimmer:APQ5":  (0.005,  0.080, 0.018, "Five-point amplitude perturbation quotient"),
    "MDVP:APQ":      (0.007,  0.137, 0.024, "MDVP amplitude perturbation quotient"),
    "Shimmer:DDA":   (0.013,  0.170, 0.045, "Average absolute difference of shimmer"),
    "NHR":           (0.0006, 0.315, 0.025, "Noise-to-harmonics ratio"),
    "HNR":           (8.441,  33.047, 21.886, "Harmonics-to-noise ratio"),
    "RPDE":          (0.256,  0.686, 0.498, "Recurrence period density entropy"),
    "DFA":           (0.574,  0.825, 0.718, "Detrended fluctuation analysis"),
    "spread1":       (-7.965, -2.434, -5.684, "Nonlinear measure of freq variation"),
    "spread2":       (0.006,  0.451, 0.226, "Nonlinear measure of freq variation"),
    "D2":            (1.423,  3.671, 2.382, "Correlation dimension"),
    "PPE":           (0.044,  0.527, 0.206, "Pitch period entropy"),
}
 
# ── Sidebar — About ──────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/1024px-NASA_logo.svg.png",
             width=40)  # placeholder, replace with your logo
    st.title("About")
    st.info(
        "This tool uses a Support Vector Machine (SVM) trained on the **UCI Parkinson's "
        "Voice Dataset** (195 recordings, 22 features). Accuracy ~96%.\n\n"
        "**⚠️ For educational use only. Not a medical diagnostic tool.**"
    )
    st.markdown("---")
    st.markdown("**Model Details**")
    st.markdown("- Algorithm: SVM (RBF kernel)")
    st.markdown("- Dataset: UCI ML Repository")
    st.markdown("- Accuracy: ~96%")
    st.markdown("- ROC-AUC: ~98%")
    st.markdown("---")
    st.markdown("Built with Python + Streamlit")
 
# ── Main layout ──────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 Parkinson\'s Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter voice biomarker measurements to predict Parkinson\'s disease risk.</div>', unsafe_allow_html=True)
 
if not model_loaded:
    st.error(
        "⚠️ Model files not found in `model/` folder.\n\n"
        "Run `python 2_train_model.py` first to train and save the model, "
        "then commit the `model/` folder to your repo."
    )
    st.stop()
 
# ── Input tabs ───────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎛️ Enter Values", "📊 Batch Predict (CSV)"])
 
with tab1:
    st.subheader("Voice Biomarker Inputs")
    st.caption("Adjust the sliders to match the patient's voice measurements. Default values are dataset means.")
 
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(FEATURES):
        info = FEATURE_INFO.get(feat, (0.0, 1.0, 0.5, feat))
        mn, mx, default, desc = info
        with cols[i % 3]:
            range_size = float(mx) - float(mn)
            step_size = range_size / 1000
            inputs[feat] = st.slider(
                label=feat,
                min_value=float(mn),
                max_value=float(mx),
                value=float(default),
                step=step_size,
                help=desc,
                format="%.6f" if mx < 0.01 else ("%.5f" if mx < 1 else "%.3f")
            )
 
    st.markdown("---")
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
 
    if predict_btn:
        input_array = np.array([[inputs[f] for f in FEATURES]])
        input_scaled = scaler.transform(input_array)
        proba       = model.predict_proba(input_scaled)[0]
        # Use 0.5 threshold explicitly on Parkinson's probability
        prediction  = 1 if proba[1] >= 0.5 else 0
        confidence  = proba[prediction] * 100
 
        
 
        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
 
        with col_res1:
            if prediction == 1:
                st.error("🔴 Parkinson's Detected")
            else:
                st.success("🟢 Healthy — No Parkinson's")
 
        with col_res2:
            st.metric("Confidence", f"{confidence:.1f}%")
 
        with col_res3:
            st.metric("P(Parkinson's)", f"{proba[1]*100:.1f}%")
 
        # Probability bar
        st.markdown("#### Prediction Probability")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.markdown(f"**Healthy** — {proba[0]*100:.1f}%")
            st.progress(float(proba[0]))
        with prob_col2:
            st.markdown(f"**Parkinson's** — {proba[1]*100:.1f}%")
            st.progress(float(proba[1]))
 
        if prediction == 1:
            st.markdown(
                '<div class="result-box-positive">'
                '<b>⚠️ High Risk Indicated</b><br>'
                'The model suggests Parkinson\'s-like vocal patterns. '
                'Please consult a neurologist for a clinical evaluation.'
                '</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box-negative">'
                '<b>✅ Low Risk</b><br>'
                'The model does not detect Parkinson\'s-like patterns in these measurements.'
                '</div>', unsafe_allow_html=True
            )
 
with tab2:
    st.subheader("Batch Prediction from CSV")
    st.caption("Upload a CSV with the same 22 feature columns as the UCI dataset (without 'name' and 'status').")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
 
    if uploaded:
        import pandas as pd
        df_up = pd.read_csv(uploaded)
 
        # Drop non-feature columns if present
        for col in ["name", "status"]:
            if col in df_up.columns:
                df_up = df_up.drop(columns=[col])
 
        st.write(f"Loaded {len(df_up)} rows, {len(df_up.columns)} columns")
        st.dataframe(df_up.head())
 
        if st.button("Predict All Rows"):
            X_up = df_up[FEATURES] if all(f in df_up.columns for f in FEATURES) else df_up
            X_sc = scaler.transform(X_up)
            preds = model.predict(X_sc)
            probas = model.predict_proba(X_sc)[:, 1]
 
            df_up["Prediction"] = ["Parkinson's" if p == 1 else "Healthy" for p in preds]
            df_up["P(Parkinson's)"] = (probas * 100).round(1).astype(str) + "%"
 
            st.success("Predictions complete!")
            st.dataframe(df_up[["Prediction", "P(Parkinson's)"] + list(df_up.columns[:-2])])
 
            csv_out = df_up.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv_out, "predictions.csv", "text/csv")