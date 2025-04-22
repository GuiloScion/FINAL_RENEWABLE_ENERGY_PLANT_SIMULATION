import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
import time
import datetime
import seaborn as sns
import psutil
import platform
import logging
import joblib
from datetime import datetime
from scipy.stats import shapiro
import shap  # For explainability
import h2o  # For AutoML
from h2o.automl import H2OAutoML
import json
from io import BytesIO

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up Streamlit configuration
st.set_page_config(page_title="Renewable Energy Predictor", layout="wide", initial_sidebar_state="expanded")

# Language Support
languages = {
    "English": {  # English translations
        "title": "🔋 Renewable Energy Production Predictor",
        "resources": "Project Resources",
        "readme": "README",
        "license": "LICENSE",
        "notebook": "MODEL_NOTEBOOK",
        "requirements": "REQUIREMENTS",
        "upload_data": "Upload Data",
        "choose_csv": "Choose a CSV file",
        "raw_data": "Raw Data",
        "data_visualization": "📊 Data Visualization",
        "select_column": "Select a column to visualize",
        "feature_selection": "Feature Selection",
        "select_features": "Select features for prediction",
        "target_selection": "Target Selection",
        "select_targets": "Select target columns",
        "model_training": "Model Training",
        "select_model": "Select Model",
        "number_of_trees": "Number of Trees (for Tree-based Models)",
        "max_depth": "Max Depth (for Tree-based Models)",
        "learning_rate": "Learning Rate (for Gradient Boosting Models)",
        "train_model": "Train Model",
        "cross_validation_scores": "🔄 Cross-Validation Scores",
        "mean_r2": "Mean R² score",
        "model_evaluation": "Model Evaluation",
        "mae": "🧮 MAE",
        "rmse": "📉 RMSE",
        "r2_score": "📈 R² Score",
        "training_time": "⏱️ Training Time",
        "feature_importances": "🔍 Feature Importances",
        "predictions_vs_actual": "📋 Predictions vs Actual",
        "scatter_plot": "📈 Predictions vs Actual Scatter Plot",
        "residual_analysis": "Residual Error Analysis",
        "residual_distribution": "Residuals Distribution",
        "shapiro_test": "Shapiro-Wilk Test",
        "cpu_usage": "CPU Usage",
        "memory_usage": "Memory Usage",
        "platform_info": "System Platform",
        "no_file_uploaded": "Please upload a CSV file to proceed.",
        "error_loading_file": "Error reading the file: ",
        "missing_values_warning": "Data contains missing values. Consider cleaning the data.",
        "processing_error": "Error during preprocessing: ",
        "empty_csv": "Uploaded file is empty or invalid. Please upload a valid CSV.",
        "training_error": "Error during model training: ",
    }
}

# Get the selected language from the user
lang = st.sidebar.selectbox("Change Language", list(languages.keys()))
texts = languages[lang]

# Title
st.title(texts["title"])

# Sidebar: Project Resources
st.sidebar.markdown(f"### {texts['resources']}")
st.sidebar.markdown(f"""
- [{texts['readme']}](https://github.com/GuiloScion/90909/blob/main/README.md)
- [{texts['license']}](https://github.com/GuiloScion/90909/blob/main/LICENSE.txt)
- [{texts['notebook']}](https://github.com/GuiloScion/90909/blob/main/RENEWABLE_ENERGY_ML_MODEL_FAST_v3_EXECUTED_FIXED.ipynb)
- [{texts['requirements']}](https://github.com/GuiloScion/90909/blob/main/requirements.txt)
""")

# Sidebar: File Upload
with st.sidebar.expander(texts["upload_data"], expanded=True):
    uploaded_file = st.file_uploader(texts["choose_csv"], type="csv")

# Function to load data
@st.cache_data
def load_data(file) -> pd.DataFrame:
    try:
        if file is None or not file.name.endswith('.csv'):
            raise ValueError(texts["empty_csv"])
        data = pd.read_csv(file)
        if data.empty:
            raise ValueError(texts["empty_csv"])
        return data
    except Exception as e:
        st.error(f"{texts['error_loading_file']} {e}")
        return pd.DataFrame()

# Function to preprocess data
def preprocess_data(data: pd.DataFrame, features: list, target_cols: list):
    try:
        if data.isnull().any().any():
            st.warning(texts["missing_values_warning"])
            data = data.dropna()

        if 'date' in features:
            features.remove('date')

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(data[features])
        X = pd.DataFrame(scaled_features, columns=features)

        y = data[target_cols] if len(target_cols) > 1 else data[target_cols[0]]
        return X, y, scaler
    except Exception as e:
        st.error(f"{texts['processing_error']} {e}")
        return None, None, None

if uploaded_file is not None:
    logging.info("File uploaded successfully.")
    data = load_data(uploaded_file)
    logging.info("Data loaded successfully.")

    if data.empty:
        st.error(texts["empty_csv"])
        st.stop()

    st.subheader(texts["raw_data"])
    st.dataframe(data)

    # Interactive Visualization
    st.subheader(texts["data_visualization"])
    selected_column = st.selectbox(texts["select_column"], data.columns)
    fig = px.histogram(data, x=selected_column, title=f"{texts['data_visualization']} - {selected_column}")
    st.plotly_chart(fig)

else:
    logging.warning(texts["no_file_uploaded"])
    st.warning(texts["no_file_uploaded"])
    st.stop()

# Sidebar: Feature Selection
features = st.sidebar.multiselect(texts["select_features"], data.columns.tolist(), default=data.columns.tolist()[:-1])

# Check if default target columns exist in the dataset
default_target_cols = ["energy_output", "cost_per_kWh", "energy_consumption"]
available_target_cols = [col for col in default_target_cols if col in data.columns]

# Sidebar: Target Selection
target_cols = st.sidebar.multiselect(
    texts["select_targets"],
    data.columns.tolist(),
    default=available_target_cols  # Dynamically set only available columns as default
)

if not target_cols:
    st.warning("No valid target columns selected. Please choose at least one target column.")
    st.stop()

X, y, scaler = preprocess_data(data, features, target_cols)
if X is None or y is None:
    st.stop()

# Sidebar: Model Training Parameters
model_choice = st.sidebar.selectbox(texts["select_model"], ["Random Forest", "Gradient Boosting", "XGBoost", "AutoML"])

# Train the model if button is clicked
if st.sidebar.button(texts["train_model"]):
    with st.spinner(texts["train_model"]):
        logging.info(f"Model training started using {model_choice}.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            start_time = time.time()

            if model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
            elif model_choice == "XGBoost":
                model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            st.subheader(texts["cross_validation_scores"])
            st.write(f"{texts['mean_r2']}: {np.mean(cv_scores):.3f}")

            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            logging.info("Model training completed.")

            model_filename = f"trained_model_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(model, model_filename)
            st.success(f"Model saved as {model_filename}")

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.subheader(texts["model_evaluation"])
            st.metric(texts["mae"], f"{mae:.3f}")
            st.metric(texts["rmse"], f"{rmse:.3f}")
            st.metric(texts["r2_score"], f"{r2:.3f}")
            st.metric(texts["training_time"], f"{training_time:.2f} seconds")

            if hasattr(model, "feature_importances_"):
                st.subheader(texts["feature_importances"])
                importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by="Importance", ascending=False)
                st.dataframe(importance_df)

        except Exception as e:
            logging.error(f"{texts['training_error']} {e}")
            st.error(f"{texts['training_error']} {e}")
            st.stop()
