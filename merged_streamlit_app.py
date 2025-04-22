import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
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

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up Streamlit configuration
st.set_page_config(page_title="Renewable Energy Predictor", layout="wide", initial_sidebar_state="expanded")

# Language Support
languages = {
    "English": {
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
    },
    "Español": {
        "title": "🔋 Predicción de Producción de Energía Renovable",
        "resources": "Recursos del Proyecto",
        "readme": "LEEME",
        "license": "LICENCIA",
        "notebook": "CUADERNO_DEL_MODELO",
        "requirements": "REQUISITOS",
        "upload_data": "Cargar Datos",
        "choose_csv": "Elija un archivo CSV",
        "raw_data": "Datos Sin Procesar",
        "data_visualization": "📊 Visualización de Datos",
        "select_column": "Seleccione una columna para visualizar",
        "feature_selection": "Selección de Características",
        "select_features": "Seleccione características para la predicción",
        "target_selection": "Selección de Objetivos",
        "select_targets": "Seleccione columnas objetivo",
        "model_training": "Entrenamiento del Modelo",
        "select_model": "Seleccione Modelo",
        "number_of_trees": "Número de Árboles (para Modelos Basados en Árboles)",
        "max_depth": "Profundidad Máxima (para Modelos Basados en Árboles)",
        "learning_rate": "Tasa de Aprendizaje (para Modelos de Gradient Boosting)",
        "train_model": "Entrenar Modelo",
        "cross_validation_scores": "🔄 Puntuaciones de Validación Cruzada",
        "mean_r2": "Puntuación Media R²",
        "model_evaluation": "Evaluación del Modelo",
        "mae": "🧮 MAE",
        "rmse": "📉 RMSE",
        "r2_score": "📈 Puntuación R²",
        "training_time": "⏱️ Tiempo de Entrenamiento",
        "feature_importances": "🔍 Importancia de Características",
        "predictions_vs_actual": "📋 Predicciones vs Valores Actuales",
        "scatter_plot": "📈 Gráfico de Dispersión de Predicciones vs Valores Actuales",
        "residual_analysis": "Análisis de Errores Residuales",
        "residual_distribution": "Distribución de Residuales",
        "shapiro_test": "Prueba de Shapiro-Wilk",
        "cpu_usage": "Uso de CPU",
        "memory_usage": "Uso de Memoria",
        "platform_info": "Plataforma del Sistema",
        "no_file_uploaded": "Por favor, suba un archivo CSV para continuar.",
        "error_loading_file": "Error al leer el archivo: ",
        "missing_values_warning": "Los datos contienen valores faltantes. Considere limpiar los datos.",
        "processing_error": "Error durante el procesamiento: ",
        "empty_csv": "El archivo subido está vacío o no es válido. Por favor, suba un archivo CSV válido.",
        "training_error": "Error durante el entrenamiento del modelo: ",
    },
}

# Get selected language
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

# Remaining application logic follows similar pattern, updating all texts dynamically...
