import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def show_visualisations(df):
    st.header("📊 Data Visualizations")

    viz_type = st.selectbox("Choose Type", ["Univariate", "Bivariate"])
    
    if viz_type == "Univariate":
        chart_type = st.selectbox("Select Chart Type", ["Histogram", "Box", "Pie", "Bar"])
        
        if chart_type == "Histogram":
            col = st.selectbox("Select numerical column", df.select_dtypes(include=np.number).columns)
            fig, ax = plt.subplots()
            df[col].plot.hist(ax=ax, color='lightblue', edgecolor='black')
            st.pyplot(fig)

        elif chart_type == "Box":
            col = st.selectbox("Select numerical column", df.select_dtypes(include=np.number).columns)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=col, palette="pastel", ax=ax)
            st.pyplot(fig)

        elif chart_type == "Pie":
            col = st.selectbox("Select categorical column", df.select_dtypes(include='object').columns)
            pie_data = df[col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
            st.pyplot(fig)

        elif chart_type == "Bar":
            col = st.selectbox("Select categorical column", df.select_dtypes(include='object').columns)
            fig, ax = plt.subplots()
            df[col].value_counts().plot.bar(ax=ax, color='lightcoral', edgecolor='black')
            st.pyplot(fig)

    elif viz_type == "Bivariate":
        chart_type = st.selectbox("Select Chart Type", ["Scatter", "Violin", "Heatmap", "Cross Table", "Pivot Table"])

        if chart_type == "Scatter":
            x = st.selectbox("X-axis", df.select_dtypes(include=np.number).columns)
            y = st.selectbox("Y-axis", df.select_dtypes(include=np.number).columns)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x, y=y, palette="pastel", ax=ax)
            st.pyplot(fig)

        elif chart_type == "Violin":
            num = st.selectbox("Select numerical column", df.select_dtypes(include=np.number).columns)
            cat = st.selectbox("Select categorical column", df.select_dtypes(include='object').columns)
            fig, ax = plt.subplots()
            sns.violinplot(data=df, x=cat, y=num, palette="pastel", ax=ax)
            st.pyplot(fig)

        elif chart_type == "Heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, cmap="pastel", ax=ax)
            st.pyplot(fig)

        elif chart_type == "Cross Table":
            col1 = st.selectbox("Select first categorical column", df.select_dtypes(include='object').columns)
            col2 = st.selectbox("Select second categorical column", df.select_dtypes(include='object').columns)
            st.write(pd.crosstab(df[col1], df[col2]))

        elif chart_type == "Pivot Table":
            values = st.selectbox("Select value column", df.select_dtypes(include=np.number).columns)
            index = st.selectbox("Select index column", df.columns)
            columns = st.selectbox("Select columns", df.columns)
            st.write(pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc='mean'))
