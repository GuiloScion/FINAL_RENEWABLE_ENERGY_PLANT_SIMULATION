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
from scipy.stats import shapiro
import h2o
from h2o.automl import H2OAutoML
import json
from io import BytesIO, StringIO

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up Streamlit configuration
st.set_page_config(page_title="Renewable Energy Predictor", layout="wide", initial_sidebar_state="expanded")

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

# Title
st.title(["title"])

# Sidebar: File Upload
with st.sidebar.expander(texts["upload_data"], expanded=True):
    uploaded_file = st.file_uploader(texts["choose_csv"], type="csv")

# Add a default demonstration dataset
def load_demo_data() -> pd.DataFrame:
    demo_data = """
    solar_output,inverter_eff,converter_eff,li_batt_charge,flow_batt_charge,geothermal_output,caes_storage,chp_output,biomass_output,htf_temp,molten_salt_storage,flywheel_storage,dac_rate,carbon_util_rate
    """ + "\n".join([",".join(map(str, [
        np.random.uniform(10, 100),  # solar_output
        np.random.uniform(0.8, 1.0),  # inverter_eff
        np.random.uniform(0.7, 0.9),  # converter_eff
        np.random.uniform(0.2, 1.0),  # li_batt_charge
        np.random.uniform(0.1, 0.9),  # flow_batt_charge
        np.random.uniform(10, 20),  # geothermal_output
        np.random.uniform(5, 50),  # caes_storage
        np.random.uniform(20, 60),  # chp_output
        np.random.uniform(1, 40),  # biomass_output
        np.random.uniform(300, 600),  # htf_temp
        np.random.uniform(100, 200),  # molten_salt_storage
        np.random.uniform(20, 50),  # flywheel_storage
        np.random.uniform(1, 3),  # dac_rate
        np.random.uniform(0.1, 3.0)  # carbon_util_rate
    ])) for _ in range(50)])  # Generate 50 rows of random data
    return pd.read_csv(StringIO(demo_data))

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

# Sidebar: Feature Selection
with st.sidebar.expander(texts["feature_selection"], expanded=True):
    st.sidebar.header(texts["feature_selection"])
    features = st.sidebar.multiselect(texts["select_features"], data.columns.tolist(), default=data.columns.tolist()[:-1])

# Define default target columns
default_target_cols = ["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"]
available_target_cols = [col for col in default_target_cols if col in data.columns]

# Sidebar: Target Selection
with st.sidebar.expander(texts["target_selection"], expanded=True):
    target_cols = st.sidebar.multiselect(
        texts["select_targets"],
        data.columns.tolist(),
        default=available_target_cols
    )

if not target_cols:
    st.warning("No valid target columns selected. Please choose at least one target column.")
    st.stop()

# Preprocess data
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

# Additional Features
st.subheader("Correlation Heatmap")
if st.checkbox("Show Correlation Heatmap"):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.subheader("Descriptive Statistics")
if st.checkbox("Show Descriptive Statistics"):
    st.write(data.describe())

st.subheader("Missing Values")
if st.checkbox("Show Missing Values"):
    missing_values = data.isnull().sum()
    st.bar_chart(missing_values)

if st.sidebar.checkbox("Enable Hyperparameter Tuning"):
    if 'model' not in locals() or model is None:
        st.error("Please train a model first before performing hyperparameter tuning.")
    else:
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
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2")
            grid_search.fit(X, y)
            st.write(f"Best Parameters: {grid_search.best_params_}")
        except Exception as e:
            st.error(f"Error during hyperparameter tuning: {e}")

if model_choice == "AutoML":
    try:
        h2o.init()
        train_data = h2o.H2OFrame(pd.concat([X, y], axis=1))
        aml = H2OAutoML(max_runtime_secs=300)
        aml.train(y=target_cols[0], training_frame=train_data)
        st.write(f"Best AutoML Model: {aml.leader}")
    except Exception as e:
        st.error(f"H2O AutoML encountered an error: {e}")
    finally:
        h2o.shutdown(prompt=False)
