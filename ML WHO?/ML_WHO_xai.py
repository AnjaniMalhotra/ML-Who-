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

    # -------------------------------
    # Basic dataset checks
    # -------------------------------
    st.write("Dataset Shape:", df.shape)
    st.write("Total Missing Values:", df.isnull().sum().sum())
    st.write("Columns:", df.columns.tolist())

    # -------------------------------
    # Target selection
    # -------------------------------
    target_col = st.selectbox("Select the target column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------------------------------
    # Clean target variable
    # -------------------------------
    if y.dtype == "object":
        y = y.astype(str).str.lower().fillna("unknown")

    if pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors="coerce").fillna(0)

    # Encode categorical or low-unique targets
    if y.dtype == "object" or y.nunique() <= 10:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Bin high-cardinality numeric targets
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        y = pd.qcut(y, q=3, labels=False)
        y = pd.Series(y).fillna(0).astype(int)
        st.info("Target binned into 3 quantile-based classes.")

    # -------------------------------
    # Feature encoding
    # -------------------------------
    X = pd.get_dummies(X)

    # Clean features (VERY IMPORTANT)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    feature_columns = X.columns

    # -------------------------------
    # Train-test split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # -------------------------------
    # Scaling
    # -------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use small sample for XAI (performance safe)
    sample_X = pd.DataFrame(X_train_scaled[:100], columns=feature_columns)
    sample_y = y_train[:100]

    st.success("Feature engineering completed successfully.")

    # =====================================================
    # SHAP EXPLANATION
    # =====================================================
    st.subheader("🔍 SHAP Feature Importance")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(sample_X, sample_y)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(sample_X)

    fig_shap, ax_shap = plt.subplots()

    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], sample_X, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)

    st.pyplot(fig_shap)

    # =====================================================
    # LIME EXPLANATION
    # =====================================================
    st.subheader("🔍 LIME Explanation")

    if sample_X.isnull().values.any() or np.isinf(sample_X.values).any():
        st.error("Invalid values found in features. Cannot run LIME.")
    else:
        class_names = [f"Class {i}" for i in np.unique(sample_y)]

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=sample_X.values,
            feature_names=sample_X.columns.tolist(),
            class_names=class_names,
            discretize_continuous=True
        )

        exp = lime_explainer.explain_instance(
            data_row=sample_X.values[0],
            predict_fn=rf_model.predict_proba,
            num_features=10
        )

        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)

    # =====================================================
    # DECISION TREE FEATURE IMPORTANCE
    # =====================================================
    st.subheader("🌲 Decision Tree Feature Importance")

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(sample_X, sample_y)

    importance_df = pd.DataFrame({
        "Feature": sample_X.columns,
        "Importance": dt_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_dt, ax_dt = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df.head(10),
        ax=ax_dt
    )
    ax_dt.set_title("Top 10 Important Features")
    st.pyplot(fig_dt)

    # =====================================================
    # OPTIONAL FEATURE REMOVAL
    # =====================================================
    st.subheader("🧹 Optional Feature Removal")

    cols_to_remove = st.multiselect(
        "Select features to remove",
        options=list(X.columns)
    )

    if cols_to_remove:
        X = X.drop(columns=cols_to_remove)
        st.warning(f"Removed columns: {cols_to_remove}")

    # =====================================================
    # DOWNLOAD MODIFIED DATASET
    # =====================================================
    final_df = X.copy()
    final_df[target_col] = y

    csv = final_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Modified Dataset",
        data=csv,
        file_name="modified_dataset.csv",
        mime="text/csv"
    )

    return X, y
