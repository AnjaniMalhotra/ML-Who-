import streamlit as st
import pandas as pd
import io

def classify_features(df):
    categorical = []
    numerical_discrete = []
    numerical_continuous = []

    for col in df.columns:
        if df[col].dtype == 'object':
            categorical.append(col)
        elif df[col].nunique() < 15:
            numerical_discrete.append(col)
        else:
            numerical_continuous.append(col)

    return categorical, numerical_discrete, numerical_continuous

def clean_dataset(df):
    # Drop duplicates
    df_cleaned = df.drop_duplicates()

    # Fill missing values (numerical columns) with median
    for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)

    return df_cleaned

def show_statistics(df):
    st.header("📈 Dataset Overview & Statistics")

    st.subheader("First 5 Rows (Head)")
    st.dataframe(df.head())

    st.subheader("Last 5 Rows (Tail)")
    st.dataframe(df.tail())

    st.subheader("Shape of Dataset")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.subheader("Columns in Dataset")
    st.write(df.columns.tolist())

    st.subheader("Duplicate Rows")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Info (Data Types and Null Counts)")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Summary Statistics (Numerical Columns)")
    st.write(df.describe())

    st.subheader("Unique Values in Each Column")
    nunique = df.nunique()
    st.write(nunique)

    st.subheader("Feature Classification")
    categorical, numerical_discrete, numerical_continuous = classify_features(df)
    st.write("**Categorical Features:**", categorical)
    st.write("**Numerical Discrete Features:**", numerical_discrete)
    st.write("**Numerical Continuous Features:**", numerical_continuous)

    # --- Cleaned Dataset Section ---
    st.markdown("---")
    st.header("Cleaned Dataset (Duplicates Removed & Missing Values Filled)")

    df_cleaned = clean_dataset(df)

    st.subheader("Shape After Cleaning")
    st.write(f"Rows: {df_cleaned.shape[0]} | Columns: {df_cleaned.shape[1]}")

    st.subheader("Missing Values After Cleaning")
    st.write(df_cleaned.isnull().sum())

    st.subheader("Preview of Cleaned Data")
    st.dataframe(df_cleaned.head())

    # Optional: Return cleaned data for further use in app
    return df_cleaned
