import streamlit as st
import pandas as pd

from ML_WHO_statistics import show_statistics
from ML_WHO_visualisations import show_visualisations
from ML_WHO_xai import show_feature_engineering
from ML_WHO_evaluation import evaluate_models

st.title("ML WHO?")

uploaded_file = st.file_uploader("Upload your dataset (any file type)", type=None)

if uploaded_file is not None:
    file_name = uploaded_file.name
    try:
        if file_name.endswith(".csv"):
            df_original = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df_original = pd.read_excel(uploaded_file)
        elif file_name.endswith(".json"):
            df_original = pd.read_json(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload CSV, Excel, or JSON.")
            st.stop()
        
        df = df_original.copy()
        st.success(f"✅ {file_name} uploaded and loaded successfully!")
    
    except Exception as e:
        st.error(f"❌ Failed to load the file. Error: {e}")

    section = st.sidebar.radio("Choose Section", [
        "Statistics", "Visualisations", "Feature Engineering with XAI", "Model Evaluation"
    ])

    if section == "Statistics":
        show_statistics(df)

    elif section == "Visualisations":
        show_visualisations(df)

    elif section == "Feature Engineering with XAI":
        X, y = show_feature_engineering(df)
        if X is not None and y is not None:
            st.session_state['X'] = X
            st.session_state['y'] = y

    elif section == "Model Evaluation":
        if 'X' in st.session_state and 'y' in st.session_state:
            evaluate_models(st.session_state['X'], st.session_state['y'])
        else:
            st.warning("⚠️ Please first complete the Feature Engineering step to proceed with Model Evaluation.")
