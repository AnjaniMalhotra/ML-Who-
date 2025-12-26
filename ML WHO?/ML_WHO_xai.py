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
    # BASIC INFO (ALWAYS RENDER SOMETHING)
    # --------------------------------------------------
    st.write("Dataset shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    # --------------------------------------------------
    # TARGET SELECTION (USER CONTROLLED)
    # --------------------------------------------------
    target_col = st.selectbox(
        "Select the target column",
        ["-- Select target column --"] + list(df.columns)
    )

    if target_col == "-- Select target column --":
        st.info("Please select a target column to proceed.")
        return

    if not st.button("🚀 Run Feature Engineering"):
        st.warning("Click the button to start feature engineering.")
        return

    # --------------------------------------------------
    # SPLIT FEATURES & TARGET
    # --------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --------------------------------------------------
    # TARGET VALIDATION (FINAL FIX)
    # --------------------------------------------------
    if pd.api.types.is_numeric_dtype(y):
        st.error(
            "❌ Numeric targets are NOT supported in XAI mode.\n\n"
            "Please select a categorical target such as:\n"
            "- Payment Status\n"
            "- Card Order Status\n"
            "- Gateway\n"
            "- Company Name\n"
            "- Month"
        )
        st.stop()

    # --------------------------------------------------
    # TARGET CLEANING (CATEGORICAL ONLY)
    # --------------------------------------------------
    y = y.astype(str).str.lower().fillna("unknown")

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))

    # --------------------------------------------------
    # FEATURE ENCODING
    # --------------------------------------------------
    X = pd.get_dummies(X)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    feature_columns = X.columns

    # --------------------------------------------------
    # TRAIN–TEST SPLIT
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

    # Small subset for XAI (CLOUD SAFE)
    sample_X = pd.DataFrame(
        X_train_scaled[:100],
        columns=feature_columns
    )
    sample_y = pd.Series(y_train[:100])

    st.success("Feature engineering completed successfully.")

    # --------------------------------------------------
    # SAVE TO SESSION STATE (IMPORTANT)
    # --------------------------------------------------
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["target"] = target_col

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

    fig_shap, _ = plt.subplots()

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

    st.pyplot(exp.as_pyplot_figure())

    # ==================================================
    # DECISION TREE FEATURE IMPORTANCE
    # ==================================================
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

    # ==================================================
    # DOWNLOAD MODIFIED DATASET
    # ==================================================
    final_df = X.copy()
    final_df[target_col] = y

    st.download_button(
        "📥 Download Modified Dataset",
        final_df.to_csv(index=False).encode("utf-8"),
        "modified_dataset.csv",
        "text/csv"
    )
