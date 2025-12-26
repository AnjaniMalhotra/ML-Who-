import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def show_feature_engineering(df):
    st.header("🧠 Feature Engineering with XAI")

    # --------------------------------------------------
    # BASIC DATA CHECKS
    # --------------------------------------------------
    st.write("Dataset shape:", df.shape)
    st.write("Total missing values:", df.isnull().sum().sum())
    st.write("Columns:", df.columns.tolist())

    # --------------------------------------------------
    # TARGET SELECTION
    # --------------------------------------------------
    target_col = st.selectbox("Select the target column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --------------------------------------------------
    # TARGET CLEANING (CRITICAL FIX)
    # --------------------------------------------------
    if y.dtype == "object":
        y = y.astype(str).str.lower().fillna("unknown")

    # Ensure numeric coercion if possible
    y = pd.to_numeric(y, errors="ignore")

    # Encode categorical or low-cardinality targets
    if y.dtype == "object" or y.nunique() <= 10:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y))  # FORCE pandas Series

    # Ensure y is ALWAYS a pandas Series
    y = pd.Series(y)

    # Bin high-cardinality numeric targets
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        y = pd.qcut(y, q=3, labels=False)
        y = y.fillna(0).astype(int)
        st.info("Target binned into 3 quantile-based classes (Low / Medium / High).")

    # --------------------------------------------------
    # FEATURE ENCODING
    # --------------------------------------------------
    X = pd.get_dummies(X)

    # Handle NaN / infinite values (VERY IMPORTANT)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    feature_columns = X.columns

    # --------------------------------------------------
    # TRAIN-TEST SPLIT
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # --------------------------------------------------
    # SCALING
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use small subset for XAI (performance safe)
    sample_X = pd.DataFrame(X_train_scaled[:100], columns=feature_columns)
    sample_y = pd.Series(y_train[:100])

    st.success("Feature engineering completed successfully.")

    # ==================================================
    # SHAP EXPLANATION
    # ==================================================
    st.subheader("🔍 SHAP Feature Importance")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(sample_X, sample_y)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(sample_X)

    fig_shap, ax_shap = plt.subplots()

    # Handle binary vs multiclass SHAP output
    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values[0],
            sample_X,
            plot_type="bar",
            show=False
        )
    else:
        shap.summary_plot(
            shap_values,
            sample_X,
            plot_type="bar",
            show=False
        )

    st.pyplot(fig_shap)

    # ==================================================
    # LIME EXPLANATION
    # ==================================================
    st.subheader("🔍 LIME Explanation")
