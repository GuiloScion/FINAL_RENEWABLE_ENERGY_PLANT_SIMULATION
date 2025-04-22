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
from scipy.stats import shapiro

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up Streamlit configuration
st.set_page_config(page_title="Renewable Energy Predictor", layout="wide", initial_sidebar_state="expanded")

# Language Support
lang = st.sidebar.selectbox("Change Language", ["English", "Español", "Français", "Deutsch"])
if lang == "Español":
    st.title("🔋 Predicción de Producción de Energía Renovable")
elif lang == "Français":
    st.title("🔋 Prédiction de la Production d'Énergie Renouvelable")
elif lang == "Deutsch":
    st.title("🔋 Vorhersage der Produktion Erneuerbarer Energien")
else:
    st.title("🔋 Renewable Energy Production Predictor")

# Sidebar: Light Mode Toggle
light_mode = st.sidebar.checkbox("Enable Light Mode", value=True)
if light_mode:
    st.markdown("""
        <style>
        body {
            background-color: #FFFFFF;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

# Sidebar: Project Resources
st.sidebar.markdown("### Project Resources")
st.sidebar.markdown("""
- [README](https://github.com/GuiloScion/90909/blob/main/README.md)
- [LICENSE](https://github.com/GuiloScion/90909/blob/main/LICENSE.txt)
- [MODEL_NOTEBOOK](https://github.com/GuiloScion/90909/blob/main/RENEWABLE_ENERGY_ML_MODEL_FAST_v3_EXECUTED_FIXED.ipynb)
- [REQUIREMENTS](https://github.com/GuiloScion/90909/blob/main/requirements.txt)
""")

# Sidebar: File Upload
with st.sidebar.expander("Upload Data", expanded=True):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to load data
@st.cache_data
def load_data(file) -> pd.DataFrame:
    try:
        if file is None or not file.name.endswith('.csv'):
            raise ValueError("Uploaded file is not a valid CSV.")
        data = pd.read_csv(file)
        if data.empty:
            raise ValueError("Uploaded file is empty.")
        return data
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return pd.DataFrame()

# Function to preprocess data
def preprocess_data(data: pd.DataFrame, features: list, target_cols: list):
    try:
        if data.isnull().any().any():
            st.warning("Data contains missing values. Consider cleaning the data.")
            data = data.dropna()

        if 'date' in features:
            features.remove('date')

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(data[features])
        X = pd.DataFrame(scaled_features, columns=features)

        y = data[target_cols] if len(target_cols) > 1 else data[target_cols[0]]
        return X, y, scaler
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, None, None

if uploaded_file is not None:
    logging.info("File uploaded successfully.")
    data = load_data(uploaded_file)
    logging.info("Data loaded successfully.")

    if data.empty:
        st.error("Uploaded file is empty or invalid. Please upload a valid CSV.")
        st.stop()

    st.subheader("Raw Data")
    st.dataframe(data)

    # Interactive Visualization
    st.subheader("📊 Data Visualization")
    selected_column = st.selectbox("Select a column to visualize", data.columns)
    fig = px.histogram(data, x=selected_column, title=f"Distribution of {selected_column}")
    st.plotly_chart(fig)

else:
    logging.warning("No file uploaded.")
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Sidebar: Feature Selection
with st.sidebar.expander("Feature Selection", expanded=True):
    st.sidebar.header("Feature Selection")
    features = st.sidebar.multiselect("Select features for prediction", data.columns.tolist(), default=data.columns.tolist()[:-1])

# Define default target columns
default_target_cols = ["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"]
available_target_cols = [col for col in default_target_cols if col in data.columns]

# Sidebar: Target Selection
with st.sidebar.expander("Target Selection", expanded=True):
    target_cols = st.sidebar.multiselect("Select target columns", data.columns.tolist(), default=available_target_cols)

if not features or not target_cols:
    st.error("Please select at least one feature and one target column.")
    st.stop()

X, y, scaler = preprocess_data(data, features, target_cols)
if X is None or y is None:
    st.stop()

# Sidebar: Model Training Parameters
with st.sidebar.expander("Model Training", expanded=True):
    st.sidebar.header("Model Training")
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "XGBoost"])
    n_estimators = st.sidebar.slider("Number of Trees (for Tree-based Models)", 10, 200, 100)
    max_depth = st.sidebar.slider("Max Depth (for Tree-based Models)", 1, 20, 10)
    learning_rate = st.sidebar.slider("Learning Rate (for Gradient Boosting Models)", 0.01, 0.3, 0.1)

# Train the model if button is clicked
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
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
            st.subheader("🔄 Cross-Validation Scores")
            st.write("Cross-validation R² scores:", cv_scores)
            st.write("Mean R² score:", np.mean(cv_scores))

            model.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            st.error(f"Error during model training: {e}")
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

        st.subheader("Model Evaluation")
        st.metric("🧮 MAE", f"{mae:.3f}")
        st.metric("📉 RMSE", f"{rmse:.3f}")
        st.metric("📈 R² Score", f"{r2:.3f}")
        st.metric("⏱️ Training Time", f"{training_time:.2f} seconds")

        if hasattr(model, "feature_importances_"):
            st.subheader("🔍 Feature Importances")
            importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by="Importance", ascending=False)
            st.dataframe(importance_df)

        st.subheader("📋 Predictions vs Actual")
        pred_df = pd.DataFrame({"Actual": y_test.values.flatten(), "Predicted": y_pred.flatten()})
        st.dataframe(pred_df)

        st.subheader("📈 Predictions vs Actual Scatter Plot")
        fig, ax = plt.subplots()
        ax.scatter(pred_df["Actual"], pred_df["Predicted"], alpha=0.7, label="Predictions")
        ax.plot([pred_df["Actual"].min(), pred_df["Actual"].max()],
                [pred_df["Actual"].min(), pred_df["Actual"].max()], 'k--', color='red', label="Perfect Fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Residual Error Analysis")
        residuals = y_test.values.flatten() - y_pred.flatten()
        fig, ax = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.set_title("Residuals Distribution")
        st.pyplot(fig)

        shapiro_stat, shapiro_p = shapiro(residuals)
        st.write(f"Shapiro-Wilk Test: Statistic={shapiro_stat:.3f}, p-value={shapiro_p:.3f}")

        st.subheader("⚙️ System Resource Usage")
        st.write(f"CPU Usage: {psutil.cpu_percent()}%")
        st.write(f"Memory Usage: {psutil.virtual_memory().percent}%")
        st.write(f"System Platform: {platform.system()} {platform.release()}")
