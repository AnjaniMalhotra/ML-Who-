import streamlit as st
import pandas as pd

from ML_WHO_statistics import show_statistics
from ML_WHO_visualisations import show_visualisations
from ML_WHO_xai import show_feature_engineering
from ML_WHO_evaluation import evaluate_models

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="ML WHO?",
    page_icon="🧠",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
body {
    background-color: #f7f9fb;
}
.main {
    font-family: 'Segoe UI', sans-serif;
}
.section-title {
    color: #0066cc;
    font-size: 28px;
    margin-bottom: 0.5em;
}
.stButton > button {
    background-color: #0066cc;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
}
.stButton > button:hover {
    background-color: #004c99;
}
.uploaded-file {
    font-weight: bold;
    color: green;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE DEFAULTS
# ==================================================
if "section" not in st.session_state:
    st.session_state.section = "Home"

# ==================================================
# APP TITLE
# ==================================================
st.markdown(
    "<h1 style='text-align:center; color:#0c4a6e;'>🤖 ML WHO? – Intelligent Dataset Explorer</h1>",
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("🔍 Navigation")

st.session_state.section = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Upload Data",
        "Statistics",
        "Visualisations",
        "Feature Engineering with XAI",
        "Model Evaluation"
    ]
)

# ==================================================
# HOME
# ==================================================
if st.session_state.section == "Home":
    st.markdown("""
    <h3 class="section-title">📊 Welcome to ML WHO?</h3>
    <p>An intelligent machine learning assistant to explore, visualize, explain,
    and evaluate datasets with ease.</p>
    <ul>
        <li>📈 Quick statistics & insights</li>
        <li>🧩 Visual trend exploration</li>
        <li>🧠 Explainable AI (SHAP & LIME)</li>
        <li>✅ Model training & evaluation</li>
    </ul>
    <p>👈 Start by uploading a dataset.</p>
    """, unsafe_allow_html=True)

# ==================================================
# UPLOAD DATA
# ==================================================
elif st.session_state.section == "Upload Data":
    st.markdown("<h3 class='section-title'>📁 Upload Your Dataset</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a CSV, Excel, or JSON file",
        type=["csv", "xlsx", "xls", "json"]
    )

    if uploaded_file:
        try:
            file_name = uploaded_file.name

            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif file_name.endswith(".json"):
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format.")
                st.stop()

            st.session_state.df = df.copy()

            st.success(f"✅ {file_name} uploaded successfully!")
            st.markdown(
                f"<p class='uploaded-file'>Preview of {file_name}</p>",
                unsafe_allow_html=True
            )
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")

# ==================================================
# REMAINING SECTIONS (REQUIRE DATA)
# ==================================================
elif "df" in st.session_state:
    df = st.session_state.df

    # ---------------- STATISTICS ----------------
    if st.session_state.section == "Statistics":
        st.markdown("<h3 class='section-title'>📊 Dataset Statistics</h3>", unsafe_allow_html=True)
        show_statistics(df)

    # ---------------- VISUALISATIONS ----------------
    elif st.session_state.section == "Visualisations":
        st.markdown("<h3 class='section-title'>📈 Visual Exploration</h3>", unsafe_allow_html=True)
        show_visualisations(df)

    # ---------------- FEATURE ENGINEERING ----------------
    elif st.session_state.section == "Feature Engineering with XAI":
        st.markdown("<h3 class='section-title'>🧠 Explainable Feature Engineering</h3>", unsafe_allow_html=True)

        # ❗ IMPORTANT: DO NOT ASSIGN RETURN VALUES
        show_feature_engineering(df)

    # ---------------- MODEL EVALUATION ----------------
    elif st.session_state.section == "Model Evaluation":
        st.markdown("<h3 class='section-title'>🧪 Model Training & Evaluation</h3>", unsafe_allow_html=True)

        if "X" in st.session_state and "y" in st.session_state:
            evaluate_models(
                st.session_state.X,
                st.session_state.y
            )
        else:
            st.warning("⚠️ Please complete Feature Engineering with XAI first.")

# ==================================================
# NO DATA FALLBACK
# ==================================================
else:
    st.info("📁 Please upload a dataset first using **Upload Data**.")
