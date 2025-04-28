from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    },
    "Español": {  # Spanish translations
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
    "Français": {  # French translations
        "title": "🔋 Prédiction de la Production d'Énergie Renouvelable",
        "resources": "Ressources du Projet",
        "readme": "LISEZ-MOI",
        "license": "LICENCE",
        "notebook": "CAHIER_DU_MODÈLE",
        "requirements": "EXIGENCES",
        "upload_data": "Télécharger les Données",
        "choose_csv": "Choisissez un fichier CSV",
        "raw_data": "Données Brutes",
        "data_visualization": "📊 Visualisation des Données",
        "select_column": "Sélectionner une colonne à visualiser",
        "feature_selection": "Sélection des Caractéristiques",
        "select_features": "Sélectionner les caractéristiques pour la prédiction",
        "target_selection": "Sélection des Cibles",
        "select_targets": "Sélectionner les colonnes cibles",
        "model_training": "Entraînement du Modèle",
        "select_model": "Sélectionner le Modèle",
        "number_of_trees": "Nombre d'Arbres (pour les Modèles Basés sur les Arbres)",
        "max_depth": "Profondeur Maximale (pour les Modèles Basés sur les Arbres)",
        "learning_rate": "Taux d'Apprentissage (pour les Modèles Gradient Boosting)",
        "train_model": "Entraîner le Modèle",
        "cross_validation_scores": "🔄 Scores de Validation Croisée",
        "mean_r2": "Score Moyen R²",
        "model_evaluation": "Évaluation du Modèle",
        "mae": "🧮 MAE",
        "rmse": "📉 RMSE",
        "r2_score": "📈 Score R²",
        "training_time": "⏱️ Temps d'Entraînement",
        "feature_importances": "🔍 Importance des Caractéristiques",
        "predictions_vs_actual": "📋 Prédictions vs Réels",
        "scatter_plot": "📈 Graphique de Dispersion Prédictions vs Réels",
        "residual_analysis": "Analyse des Erreurs Résiduelles",
        "residual_distribution": "Distribution des Résidus",
        "shapiro_test": "Test de Shapiro-Wilk",
        "cpu_usage": "Utilisation du CPU",
        "memory_usage": "Utilisation de la Mémoire",
        "platform_info": "Plateforme du Système",
        "no_file_uploaded": "Veuillez télécharger un fichier CSV pour continuer.",
        "error_loading_file": "Erreur lors de la lecture du fichier : ",
        "missing_values_warning": "Les données contiennent des valeurs manquantes. Veuillez nettoyer les données.",
        "processing_error": "Erreur lors du traitement : ",
        "empty_csv": "Le fichier téléchargé est vide ou invalide. Veuillez télécharger un fichier CSV valide.",
        "training_error": "Erreur lors de l'entraînement du modèle : ",
    },
    "Deutsch": {  # German translations
        "title": "🔋 Vorhersage der Produktion Erneuerbarer Energien",
        "resources": "Projektressourcen",
        "readme": "README",
        "license": "LIZENZ",
        "notebook": "MODELL_NOTIZBUCH",
        "requirements": "ANFORDERUNGEN",
        "upload_data": "Daten Hochladen",
        "choose_csv": "Wählen Sie eine CSV-Datei",
        "raw_data": "Rohdaten",
        "data_visualization": "📊 Datenvisualisierung",
        "select_column": "Wählen Sie eine Spalte zur Visualisierung",
        "feature_selection": "Merkmalsauswahl",
        "select_features": "Wählen Sie Merkmale für die Vorhersage",
        "target_selection": "Zielauswahl",
        "select_targets": "Wählen Sie Zielspalten",
        "model_training": "Modelltraining",
        "select_model": "Wählen Sie ein Modell",
        "number_of_trees": "Anzahl der Bäume (für baumbasierte Modelle)",
        "max_depth": "Maximale Tiefe (für baumbasierte Modelle)",
        "learning_rate": "Lernrate (für Gradient-Boosting-Modelle)",
        "train_model": "Modell Trainieren",
        "cross_validation_scores": "🔄 Kreuzvalidierungsergebnisse",
        "mean_r2": "Mittlerer R²-Wert",
        "model_evaluation": "Modellevaluierung",
        "mae": "🧮 MAE",
        "rmse": "📉 RMSE",
        "r2_score": "📈 R²-Wert",
        "training_time": "⏱️ Trainingszeit",
        "feature_importances": "🔍 Merkmalswichtigkeit",
        "predictions_vs_actual": "📋 Vorhersagen vs Tatsächliche Werte",
        "scatter_plot": "📈 Streudiagramm Vorhersagen vs Tatsächliche Werte",
        "residual_analysis": "Analyse der Residualfehler",
        "residual_distribution": "Verteilung der Residualfehler",
        "shapiro_test": "Shapiro-Wilk-Test",
        "cpu_usage": "CPU-Nutzung",
        "memory_usage": "Speichernutzung",
        "platform_info": "Systemplattform",
        "no_file_uploaded": "Bitte laden Sie eine CSV-Datei hoch, um fortzufahren.",
        "error_loading_file": "Fehler beim Lesen der Datei: ",
        "missing_values_warning": "Die Daten enthalten fehlende Werte. Bitte bereinigen Sie die Daten.",
        "processing_error": "Fehler bei der Verarbeitung: ",
        "empty_csv": "Die hochgeladene Datei ist leer oder ungültig. Bitte laden Sie eine gültige CSV-Datei hoch.",
        "training_error": "Fehler beim Training des Modells: ",
    },
    "Nederlands": {  # Dutch translations
        "title": "🔋 Voorspelling van Hernieuwbare Energieproductie",
        "resources": "Projectbronnen",
        "readme": "README",
        "license": "LICENTIE",
        "notebook": "MODEL_NOTITIEBOEK",
        "requirements": "VEREISTEN",
        "upload_data": "Gegevens Uploaden",
        "choose_csv": "Kies een CSV-bestand",
        "raw_data": "Ruwe Gegevens",
        "data_visualization": "📊 Gegevensvisualisatie",
        "select_column": "Selecteer een kolom om te visualiseren",
        "feature_selection": "Kenmerken Selecteren",
        "select_features": "Selecteer kenmerken voor voorspelling",
        "target_selection": "Doel Selecteren",
        "select_targets": "Selecteer doelkolommen",
        "model_training": "Modeltraining",
        "select_model": "Selecteer Model",
        "number_of_trees": "Aantal Bomen (voor boombasede Modellen)",
        "max_depth": "Maximale Diepte (voor boombasede Modellen)",
        "learning_rate": "Leertempo (voor Gradient Boosting Modellen)",
        "train_model": "Model Trainen",
        "cross_validation_scores": "🔄 Kruisvalideringsscores",
        "mean_r2": "Gemiddelde R²-score",
        "model_evaluation": "Model Evaluatie",
        "mae": "🧮 MAE",
        "rmse": "📉 RMSE",
        "r2_score": "📈 R²-score",
        "training_time": "⏱️ Trainingstijd",
        "feature_importances": "🔍 Kenmerkbelangrijkheid",
        "predictions_vs_actual": "📋 Voorspellingen vs Werkelijke Waarden",
        "scatter_plot": "📈 Spreidingsdiagram Voorspellingen vs Werkelijke Waarden",
        "residual_analysis": "Analyse van Residualfouten",
        "residual_distribution": "Verdeling van Residualfouten",
        "shapiro_test": "Shapiro-Wilk Test",
        "cpu_usage": "CPU-gebruik",
        "memory_usage": "Geheugengebruik",
        "platform_info": "Systeemplatform",
        "no_file_uploaded": "Upload een CSV-bestand om verder te gaan.",
        "error_loading_file": "Fout bij het lezen van het bestand: ",
        "missing_values_warning": "De gegevens bevatten ontbrekende waarden. Overweeg om de gegevens op te schonen.",
        "processing_error": "Fout bij de verwerking: ",
        "empty_csv": "Het geüploade bestand is leeg of ongeldig. Upload een geldig CSV-bestand.",
        "training_error": "Fout tijdens het trainen van het model: ",
    },
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

# Add a default demonstration dataset
def load_demo_data() -> pd.DataFrame:
    demo_data = """
    solar_output,inverter_eff,converter_eff,li_batt_charge,flow_batt_charge,geothermal_output,caes_storage,chp_output,biomass_output,htf_temp,molten_salt_storage,flywheel_storage,dac_rate,carbon_util_rate
    37.454011884736246,0.9872626164988103,0.8047143778530101,0.9082658859666537,0.6420316461542878,14.722444603479284,5.16817211686077,30.188175514805263,4.124954753437304,510.8587663709747,139.63234280394903,26.00817505559967,1.6893506307216455,0.43460144043711846
    95.07143064099162,0.9697619541025003,0.895461561689567,0.23956189066697242,0.08413996499504883,19.254886430096263,53.1354631568148,51.389390471299336,36.102116267182666,423.361699894322,107.21927326882408,42.609075015927004,2.7859033903195862,2.685532709092739
    """
    from io import StringIO
    return pd.read_csv(StringIO(demo_data))

# Check if a file is uploaded, otherwise load the demo data
if uploaded_file is not None:
    logging.info("File uploaded successfully.")
    data = load_data(uploaded_file)
else:
    logging.info("No file uploaded. Using demonstration dataset.")
    st.warning(texts["no_file_uploaded"] + " Using a demonstration dataset instead.")
    data = load_demo_data()

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
with st.sidebar.expander(texts["feature_selection"], expanded=True):
    st.sidebar.header(texts["feature_selection"])
    features = st.sidebar.multiselect(texts["select_features"], data.columns.tolist(), default=data.columns.tolist()[:-1])

# Define default target columns
default_target_cols = ["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"]
available_target_cols = [col for col in default_target_cols if col in data.columns]

# Sidebar: Target Selection
with st.sidebar.expander(texts["target_selection"], expanded=True):
    # Dynamically filter default target columns based on the dataset
    default_target_cols = ["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"]
    available_target_cols = [col for col in default_target_cols if col in data.columns]

    target_cols = st.sidebar.multiselect(
        texts["select_targets"],
        data.columns.tolist(),
        default=available_target_cols  # Only include available columns
    )

if not target_cols:
    st.warning("No valid target columns selected. Please choose at least one target column.")
    st.stop()

X, y, scaler = preprocess_data(data, features, target_cols)
if X is None or y is None:
    st.stop()

# Sidebar: Model Training Parameters
with st.sidebar.expander(texts["model_training"], expanded=True):
    st.sidebar.header(texts["model_training"])
    model_choice = st.sidebar.selectbox(texts["select_model"], ["Random Forest", "Gradient Boosting", "XGBoost"])
    n_estimators = st.sidebar.slider(texts["number_of_trees"], 10, 200, 100)
    max_depth = st.sidebar.slider(texts["max_depth"], 1, 20, 10)
    learning_rate = st.sidebar.slider(texts["learning_rate"], 0.01, 0.3, 0.1)

# Train the model if button is clicked
if st.sidebar.button(texts["train_model"]):
    with st.spinner(texts["train_model"]):
        logging.info(f"Model training started using {model_choice}.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        start_time = time.time()

        try:
            if model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
            elif model_choice == "XGBoost":
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            st.subheader(texts["cross_validation_scores"])
            st.write(f"{texts['mean_r2']}: {np.mean(cv_scores):.3f}")

            model.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"{texts['training_error']} {e}")
            st.error(f"{texts['training_error']} {e}")
            st.stop()

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

        st.subheader(texts["predictions_vs_actual"])
        pred_df = pd.DataFrame({"Actual": y_test.values.flatten(), "Predicted": y_pred.flatten()})
        st.dataframe(pred_df)

        st.subheader(texts["scatter_plot"])
        fig, ax = plt.subplots()
        ax.scatter(pred_df["Actual"], pred_df["Predicted"], alpha=0.7, label="Predictions")
        ax.plot([pred_df["Actual"].min(), pred_df["Actual"].max()],
                [pred_df["Actual"].min(), pred_df["Actual"].max()], 'k--', color='red', label="Perfect Fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        st.pyplot(fig)

        st.subheader(texts["residual_analysis"])
        residuals = y_test.values.flatten() - y_pred.flatten()
        fig, ax = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.set_title(texts["residual_distribution"])
        st.pyplot(fig)

        shapiro_stat, shapiro_p = shapiro(residuals)
        st.write(f"{texts['shapiro_test']}: Statistic={shapiro_stat:.3f}, p-value={shapiro_p:.3f}")

        st.subheader(texts["cpu_usage"])
        st.write(f"{texts['cpu_usage']}: {psutil.cpu_percent()}%")
        st.write(f"{texts['memory_usage']}: {psutil.virtual_memory().percent}%")
        st.write(f"{texts['platform_info']}: {platform.system()} {platform.release()}")

   
# Feature 1: Correlation Heatmap
st.subheader("Correlation Heatmap")
if st.checkbox("Show Correlation Heatmap"):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Feature 2: Descriptive Statistics
st.subheader("Descriptive Statistics")
if st.checkbox("Show Descriptive Statistics"):
    st.write(data.describe())

# Feature 3: Missing Value Visualization
st.subheader("Missing Values")
if st.checkbox("Show Missing Values"):
    missing_values = data.isnull().sum()
    st.bar_chart(missing_values)

# Feature 4: Hyperparameter Search
if st.sidebar.checkbox("Enable Hyperparameter Tuning"):
    if 'model' not in locals() or model is None:
        st.error("Please train a model first before performing hyperparameter tuning.")
    else:
        # Define parameter grid based on the selected model
        if model_choice == "Random Forest":
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, 15],
            }
        elif model_choice in ["Gradient Boosting", "XGBoost"]:
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, 15],
                "learning_rate": [0.01, 0.1, 0.2],
            }
        else:
            st.error("Hyperparameter tuning is not supported for the selected model.")
            st.stop()

        try:
            # Run grid search
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2")
            grid_search.fit(X, y)
            st.write(f"Best Parameters: {grid_search.best_params_}")
        except Exception as e:
            st.error(f"Error during hyperparameter tuning: {e}")
