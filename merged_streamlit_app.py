# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
import time
import seaborn as sns
import psutil
import platform
import logging
import joblib
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up Streamlit configuration
st.set_page_config(page_title="Renewable Energy Predictor", layout="wide")
st.title("🔋 Renewable Energy Production Predictor")

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

@st.cache_data
def load_data(file: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file)
        logging.info(f"Data loaded successfully with shape {data.shape}.")
        return data
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        logging.error(f"Error reading the file: {e}")
        return pd.DataFrame()

# Default dataset for demonstration
demo_data_path = "default_dataset.csv"

def get_demo_data() -> pd.DataFrame:
    """Load a demo dataset or generate a sample dataset if the default file is missing."""
    try:
        data = pd.read_csv(demo_data_path)
        logging.info(f"Default dataset loaded successfully with shape {data.shape}.")
        return data
    except FileNotFoundError:
        logging.warning("Default dataset not found. Generating a sample dataset.")
        st.warning("Default dataset not found. Using a sample dataset for demonstration.")
        
        # Generate a sample dataset with meaningful relationships
        num_samples = 100
        np.random.seed(42)
        sample_data = pd.DataFrame({
            "solar_output": np.random.uniform(20, 100, num_samples),
            "inverter_eff": np.random.uniform(0.9, 0.99, num_samples),
            "converter_eff": np.random.uniform(0.8, 0.95, num_samples),
            "li_batt_charge": np.random.uniform(0.1, 1.0, num_samples),
            "flow_batt_charge": np.random.uniform(0.1, 1.0, num_samples),
            "geothermal_output": np.random.uniform(10, 50, num_samples),
            "caes_storage": np.random.uniform(10, 100, num_samples),
            "chp_output": np.random.uniform(5, 60, num_samples),
            "biomass_output": np.random.uniform(5, 40, num_samples),
            "htf_temp": np.random.uniform(200, 600, num_samples),
            "molten_salt_storage": np.random.uniform(50, 200, num_samples),
            "flywheel_storage": np.random.uniform(10, 100, num_samples),
            "dac_rate": np.random.uniform(0.1, 1.0, num_samples),
            "carbon_util_rate": np.random.uniform(0.1, 1.0, num_samples),
        })
        sample_data["solar_output"] += 0.5 * sample_data["inverter_eff"] * 100  # Add correlations
        return sample_data

if uploaded_file is not None:
    logging.info("File uploaded successfully.")
    data = load_data(uploaded_file)
else:
    st.warning("No file uploaded. Using default demonstration dataset.")
    data = get_demo_data()

if data.empty:
    st.error("Dataset is empty or invalid. Please upload a valid CSV or use the default dataset.")
    st.stop()

st.subheader("Raw Data")
st.dataframe(data)

# Sidebar: Feature Selection
with st.sidebar.expander("Feature Selection", expanded=True):
    st.sidebar.header("Feature Selection")
    features = st.sidebar.multiselect("Select features for prediction", data.columns.tolist(), default=data.columns.tolist()[:-1])

# Define default target columns
default_target_cols = ["solar_output"]

# Ensure default target columns exist in the dataset
available_target_cols = [col for col in default_target_cols if col in data.columns]

# Sidebar: Target Selection
with st.sidebar.expander("Target Selection", expanded=True):
    target_cols = st.sidebar.multiselect("Select target columns", data.columns.tolist(), default=available_target_cols)

if not features or not target_cols:
    st.error("Please select at least one feature and one target column.")
    st.stop()

# Remove 'date' from features if present
if 'date' in features:
    features.remove('date')

# Check for missing values
if data.isnull().any().any():
    st.warning("Data contains missing values. Consider cleaning the data.")

# Scale the input features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])
X = pd.DataFrame(scaled_features, columns=features)
y = data[target_cols[0]]

# Sidebar: Model Training Parameters
with st.sidebar.expander("Model Training", expanded=True):
    st.sidebar.header("Model Training")
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "XGBoost"])
    n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 50, 10)

# Train the model if button is clicked
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        logging.info(f"Model training started using {model_choice}.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        start_time = time.time()

        try:
            # Hyperparameter tuning using GridSearchCV
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}
            if model_choice == "Random Forest":
                model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring='r2', cv=3)
            elif model_choice == "Gradient Boosting":
                model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, scoring='r2', cv=3)
            else:
                model = GridSearchCV(XGBRegressor(random_state=42), param_grid, scoring='r2', cv=3)

            model.fit(X_train, y_train)
            best_model = model.best_estimator_

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            st.error(f"Error during model training: {e}")
            st.stop()

        training_time = time.time() - start_time
        logging.info("Model training completed.")

        # Save trained model
        model_filename = f"trained_model_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(best_model, model_filename)
        st.success(f"Model saved as {model_filename}")
