import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Optional imports for regression
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

try:
    from catboost import CatBoostRegressor
    cat_available = True
except ImportError:
    cat_available = False

def evaluate_models(X, y):
    st.header("📌 Machine Learning Model Evaluation")

    tab1, tab2 = st.tabs(["🧠 Classification", "📈 Regression"])

    with tab1:
        evaluate_classification_models(X, y)

    with tab2:
        evaluate_regression_models(X, y)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_classification_models(X, y):
    st.write("🔍 Problem Type: **Classification**")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Naive Bayes (Bernoulli)": BernoulliNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0),
        "SVM": SVC(probability=True),
        "AdaBoost": AdaBoostClassifier(),
        "Bagging": BaggingClassifier(),
    }

    results = []
    model_instances = {}  # ✅ Define this to store trained models

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        avg_score = np.mean([acc, prec, rec, f1])

        results.append([name, acc, prec, rec, f1, avg_score])
        model_instances[name] = model  # ✅ Save the trained model

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Avg Score"])
    top5 = results_df.sort_values(by="Avg Score", ascending=False).head(5)

    st.subheader("🏆 Top 5 Classification Models")
    st.dataframe(top5)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Avg Score", y="Model", data=top5, ax=ax, palette="cubehelix")
    ax.set_title("Top 5 Models by Average Score")
    st.pyplot(fig)

    # ✅ Add Confusion Matrix for Top Model
    top_model_name = top5.iloc[0]["Model"]
    top_model = model_instances[top_model_name]
    y_pred_top = top_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_top)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    st.subheader(f"📊 Confusion Matrix for Top Model: {top_model_name}")
    fig_cm, ax_cm = plt.subplots()
    disp.plot(ax=ax_cm, cmap="Blues")
    st.pyplot(fig_cm)

    

def evaluate_regression_models(X, y):
    with st.container():
        st.write("🔍 Problem Type: **Regression**")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet Regression": ElasticNet(),
            "LARS Regression": Lars(),
            "LassoLars Regression": LassoLars(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "SVR (RBF Kernel)": SVR(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
        }

        if xgb_available:
            models["XGBoost"] = XGBRegressor(random_state=42)
        if cat_available:
            models["CatBoost"] = CatBoostRegressor(verbose=0, random_state=42)

        # Store results for evaluation
        results = []
        predictions_dict = {}

        # Train & evaluate
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results.append([name, mse, r2])
                predictions_dict[name] = y_pred
            except Exception as e:
                st.warning(f"⚠️ {name} skipped due to error: {e}")

        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results, columns=["Model", "MSE", "R2"]).sort_values(by="R2", ascending=False)
        top5 = results_df.head(5)

        # Show top 5 model results
        st.subheader("🏆 Top 5 Regression Models by R² Score")
        st.dataframe(top5)

        # Dropdown to choose one of the top 5 for plotting
        selected_model = st.selectbox("📊 Select a Top Model for Visualization", top5["Model"].tolist())

        y_pred = predictions_dict[selected_model]
        residuals = y_test - y_pred

        # Plot: Actual vs Predicted
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax1.set_title(f"Actual vs Predicted - {selected_model}")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        st.pyplot(fig1)

        # Plot: Residuals vs Predicted
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax2, alpha=0.6)
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_title(f"Residuals vs Predicted - {selected_model}")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        st.pyplot(fig2)
