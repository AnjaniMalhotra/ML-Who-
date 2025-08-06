import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def show_feature_engineering(df):
    st.header("🧠 Feature Engineering with XAI")

    # Select target column
    target_col = st.selectbox("Select the target column", df.columns)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Process target column
    if y.dtype == 'object':
        y = y.str.lower()
    elif pd.api.types.is_numeric_dtype(y):
        y = y.fillna(0)

    # Label encode categorical target or small unique values
    if y.dtype == 'object' or y.nunique() < 10:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Bin high-cardinality numeric targets into categories
    if pd.api.types.is_numeric_dtype(y):
        y = pd.Series(y)
        if y.nunique() > 20:
            y = pd.cut(y, bins=3, labels=False)
            st.info("Target converted into 3 categories: Low, Medium, High.")

    # Encode categorical features
    X = pd.get_dummies(X)

    # Store column names before scaling
    feature_columns = X.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare sample for explanation
    sample_X = pd.DataFrame(X_train_scaled[:100], columns=feature_columns)
    sample_y = y_train[:100]

    st.write("Training SHAP, LIME, and Decision Tree on a 100-sample subset...")

    # SHAP explanation
    st.subheader("🔍 SHAP Summary Plot")
    model_for_shap = RandomForestClassifier().fit(sample_X, sample_y)
    explainer_shap = shap.TreeExplainer(model_for_shap)
    shap_values = explainer_shap.shap_values(sample_X)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    st.pyplot(fig)

    # LIME explanation
    st.subheader("🔍 LIME Explanation - Random Forest Model")

    # Check for invalid values in sample
    if sample_X.isnull().values.any() or np.isinf(sample_X.values).any():
        st.error("The sample contains invalid values (NaN or Infinite). Please clean your data.")
        return X, y

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=sample_X.values,
        feature_names=sample_X.columns.tolist(),
        class_names=["Class 0", "Class 1"],
        discretize_continuous=True
    )

    # Explain first instance
    i = 0
    exp = explainer_lime.explain_instance(
        data_row=sample_X.values[i],
        predict_fn=model_for_shap.predict_proba,
        num_features=10
    )
    fig_lime = exp.as_pyplot_figure()
    st.pyplot(fig_lime)

    # Decision Tree Feature Importances
    st.subheader("🌲 Decision Tree Feature Importances")
    dt_model = DecisionTreeClassifier().fit(sample_X, sample_y)
    importances = dt_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': sample_X.columns, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig_dt, ax_dt = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax_dt, palette="coolwarm")
    ax_dt.set_title("Top 10 Important Features by Decision Tree")
    st.pyplot(fig_dt)

    # Optional Feature Removal
    st.subheader("🧹 Optional Feature Removal")
    cols_to_remove = st.multiselect("Select columns to remove from training", options=list(X.columns))

    if target_col in cols_to_remove:
        st.warning("You cannot remove the target column.")
        cols_to_remove.remove(target_col)

    if cols_to_remove:
        X = X.drop(columns=cols_to_remove)
        st.warning(f"Dropped columns: {cols_to_remove}")

    # Option to download modified dataset
    modified_df = X.copy()
    modified_df[target_col] = y  # Add back target column
    csv = modified_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📅 Download Modified Dataset as CSV",
        data=csv,
        file_name='modified_dataset.csv',
        mime='text/csv'
    )

    return X, y
