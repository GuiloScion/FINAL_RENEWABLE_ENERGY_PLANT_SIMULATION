import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
import datetime
import seaborn as sns
import psutil
import platform
import logging
import joblib
from datetime import datetime
import torch
from sklearn.model_selection import cross_val_score

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
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return pd.DataFrame()

if uploaded_file is not None:
    logging.info("File uploaded successfully.")
    data = load_data(uploaded_file)
    logging.info("Data loaded successfully.")

    if data.empty:
        st.error("Uploaded file is empty or invalid. Please upload a valid CSV.")
        st.stop()

    st.subheader("Raw Data")
    st.dataframe(data)
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
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])
X = pd.DataFrame(scaled_features, columns=features)
y = data[target_cols] if len(target_cols) > 1 else data[target_cols[0]]

# Sidebar: Model Training Parameters
with st.sidebar.expander("Model Training", expanded=True):
    st.sidebar.header("Model Training")
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "XGBoost"])
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

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
                model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            else:
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            # Cross-validation for model evaluation
            st.subheader("🔄 Cross-Validation Scores")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            st.write("Cross-validation R² scores:", cv_scores)
            st.write("Mean R² score:", np.mean(cv_scores))

            model.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            st.error(f"Error during model training: {e}")
            st.stop()

        training_time = time.time() - start_time
        logging.info("Model training completed.")

        # Save trained model with versioning
        model_filename = f"trained_model_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_filename)
        st.success(f"Model saved as {model_filename}")
        logging.info(f"Model saved as {model_filename}")

        # Model evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Evaluation")
        st.metric("🧮 MAE", f"{mae:.3f}", help="Mean Absolute Error: Measures average magnitude of errors in predictions.")
        st.metric("📉 RMSE", f"{rmse:.3f}", help="Root Mean Squared Error: Measures the square root of average squared differences between observed and predicted values.")
        st.metric("📈 R² Score", f"{r2:.3f}", help="R² Score: Indicates how well the model explains variability in the target variable.")
        st.metric("⏱️ Training Time", f"{training_time:.2f} seconds", help="Time taken to train the model.")

        # Feature Importances
        if hasattr(model, "feature_importances_"):
            st.subheader("🔍 Feature Importances")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.dataframe(importance_df)

        # Predictions Table
        st.subheader("📋 Predictions vs Actual")
        pred_df = pd.DataFrame({
            "Actual": y_test.values.flatten(),
            "Predicted": y_pred.flatten()
        })
        st.dataframe(pred_df)

        # Scatter plot for predictions vs actual
        st.subheader("📈 Predictions vs Actual Scatter Plot")
        fig, ax = plt.subplots()
        ax.scatter(pred_df["Actual"], pred_df["Predicted"], alpha=0.7, label="Predictions")
        ax.plot([pred_df["Actual"].min(), pred_df["Actual"].max()],
                [pred_df["Actual"].min(), pred_df["Actual"].max()], 'k--', color='red', label="Perfect Fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        st.pyplot(fig)

        # Residual Analysis
        st.subheader("Residual Error Analysis")
        residuals = y_test.values.flatten() - y_pred.flatten()
        fig, ax = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, ax=ax)
        ax.set_title("Residuals Distribution")
        st.pyplot(fig)

        # System Resource Usage
        st.subheader("⚙️ System Resource Usage")
        st.write(f"CPU Usage: {psutil.cpu_percent()}%")
        st.write(f"Memory Usage: {psutil.virtual_memory().percent}%")
        st.write(f"System Platform: {platform.system()} {platform.release()}")

        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
            st.write(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        else:
            st.write("GPU: Not Available")
