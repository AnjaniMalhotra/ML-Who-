import streamlit as st
import pandas as pd

from ML_WHO_statistics import show_statistics
from ML_WHO_visualisations import show_visualisations
from ML_WHO_xai import show_feature_engineering
from ML_WHO_evaluation import evaluate_models

# ========== Page Setup ==========
st.set_page_config(page_title="ML WHO?", page_icon="🧠", layout="wide")

# ========== Custom CSS Styling ==========
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

# ========== Session State Defaults ==========
if 'section' not in st.session_state:
    st.session_state['section'] = "Home"

# ========== App Title ==========
st.markdown("<h1 style='text-align: center; color: #0c4a6e;'>🤖 ML WHO? - Intelligent Dataset Explorer</h1>", unsafe_allow_html=True)

# ========== Navigation ==========
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to", ["Home", "Upload Data", "Statistics", "Visualisations", "Feature Engineering with XAI", "Model Evaluation"])
st.session_state['section'] = section

# ========== Home ==========
if st.session_state['section'] == "Home":
    st.markdown("""
        <h3 class="section-title">📊 Welcome to ML WHO?</h3>
        <p>This intelligent machine learning assistant helps you explore, visualize, explain, and evaluate models on your datasets with ease.</p>
        <ul>
            <li>📈 Generate quick insights and statistics</li>
            <li>🧩 Visualize relationships and trends</li>
            <li>🤖 Perform feature engineering with explainability (XAI: SHAP & LIME)</li>
            <li>✅ Train and evaluate classification models</li>
        </ul>
        <p>👈 Start by uploading your dataset from the sidebar.</p>
    """, unsafe_allow_html=True)

# ========== File Upload ==========
if st.session_state['section'] == "Upload Data":
    st.markdown("<h3 class='section-title'>📁 Upload Your Dataset</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV, Excel or JSON file", type=["csv", "xlsx", "xls", "json"])

    if uploaded_file:
        file_name = uploaded_file.name
        try:
            if file_name.endswith(".csv"):
                df_original = pd.read_csv(uploaded_file)
            elif file_name.endswith((".xlsx", ".xls")):
                df_original = pd.read_excel(uploaded_file)
            elif file_name.endswith(".json"):
                df_original = pd.read_json(uploaded_file)
            else:
                st.error("❌ Unsupported file format. Please upload CSV, Excel, or JSON.")
                st.stop()

            df = df_original.copy()
            st.session_state['df'] = df
            st.success(f"✅ {file_name} uploaded successfully!")
            st.markdown(f"<p class='uploaded-file'>Preview of {file_name}:</p>", unsafe_allow_html=True)
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Failed to load file. Error: {e}")

# ========== Other Sections ==========
if 'df' in st.session_state:
    df = st.session_state['df']

    if st.session_state['section'] == "Statistics":
        st.markdown("<h3 class='section-title'>📊 Dataset Statistics</h3>", unsafe_allow_html=True)
        show_statistics(df)

    elif st.session_state['section'] == "Visualisations":
        st.markdown("<h3 class='section-title'>📈 Visual Exploration</h3>", unsafe_allow_html=True)
        show_visualisations(df)

    elif st.session_state['section'] == "Feature Engineering with XAI":
        st.markdown("<h3 class='section-title'>🧠 Explainable Feature Engineering</h3>", unsafe_allow_html=True)
        X, y = show_feature_engineering(df)
        if X is not None and y is not None:
            st.session_state['X'] = X
            st.session_state['y'] = y

    elif st.session_state['section'] == "Model Evaluation":
        st.markdown("<h3 class='section-title'>🧪 Model Training & Evaluation</h3>", unsafe_allow_html=True)
        if 'X' in st.session_state and 'y' in st.session_state:
            evaluate_models(st.session_state['X'], st.session_state['y'])
        else:
            st.warning("⚠️ Please complete 'Feature Engineering with XAI' first.")

else:
    if section != "Home" and section != "Upload Data":
        st.info("📁 Please upload a dataset first using the **Upload Data** section.")
