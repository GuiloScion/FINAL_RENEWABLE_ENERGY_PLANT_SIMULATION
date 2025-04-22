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
import os

st.set_page_config(page_title="Renewable Energy Predictor", layout="wide")
st.title("🔋 Renewable Energy Production Predictor")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(data)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Check available columns in the dataset
st.write("Available Columns in Dataset:", data.columns.tolist())

# Sidebar feature/target selection
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect("Select features for prediction", data.columns.tolist(), default=data.columns.tolist()[:-1])
target_cols = st.sidebar.multiselect("Select target columns", data.columns.tolist(), default=["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"])

# Handle missing target columns by checking if they exist in the dataset
missing_cols = [col for col in target_cols if col not in data.columns]
if missing_cols:
    st.warning(f"The following target columns are missing from the data: {', '.join(missing_cols)}")
    
    # Example: Add logic to create missing columns
    if 'energy_output' in missing_cols:
        # Assuming energy_output can be calculated by summing different energy sources (adjust as necessary)
        if 'solar_output' in data.columns and 'wind_output' in data.columns and 'geothermal_output' in data.columns:
            data['energy_output'] = data['solar_output'] + data['wind_output'] + data['geothermal_output']  # Adjust column names as necessary
        else:
            st.warning("Required columns for calculating energy_output are missing.")
    
    if 'co2_captured' in missing_cols:
        # Assuming CO2 captured is a simple function of energy output (replace with actual logic)
        if 'energy_output' in data.columns:
            emission_factor = 0.5  # Example: adjust emission factor
            data['co2_captured'] = data['energy_output'] * emission_factor
        else:
            st.warning("energy_output is missing, unable to calculate co2_captured.")
            
    # Recheck if missing columns were created
    missing_cols = [col for col in target_cols if col not in data.columns]
    if missing_cols:
        st.error(f"Some target columns are still missing: {', '.join(missing_cols)}")
        st.stop()

if not features or not target_cols:
    st.error("Please select at least one feature and one target column.")
    st.stop()

if 'date' in features:
    features.remove('date')

# Scale the input features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])
X = pd.DataFrame(scaled_features, columns=features)
y = data[target_cols] if len(target_cols) > 1 else data[[target_cols[0]]]

# Sidebar model training parameters
st.sidebar.header("Model Training")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "XGBoost"])
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

if st.sidebar.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start time
    start_time = time.time()

    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test.values.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.values.flatten(), y_pred.flatten()))
    r2 = r2_score(y_test.values.flatten(), y_pred.flatten())

    st.subheader("Model Evaluation")
    st.metric("🧮 MAE", f"{mae:.3f}")
    st.metric("📉 RMSE", f"{rmse:.3f}")
    st.metric("📈 R² Score", f"{r2:.3f}")
    st.metric("⏱️ Training Time", f"{training_time:.2f} seconds")

    # Performance badge
    if r2 >= 0.9:
        st.success("✅ Excellent Model Performance")
    elif r2 >= 0.75:
        st.info("ℹ️ Good Model Performance")
    else:
        st.warning("⚠️ Model Needs Improvement")

    # Model summary report
    st.subheader("📊 Model Summary Report")
    report_data = {
        "Model": model_choice,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Training Time (s)": training_time,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.json(report_data)

    # Residual error analysis
    st.subheader("Residual Error Analysis")
    residuals = y_test.values.flatten() - y_pred.flatten()
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    st.pyplot(fig)

    # Feature importances
    st.subheader("🔍 Feature Importances")
    feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    if feature_importances is not None:
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(importance_df)
        st.bar_chart(importance_df.set_index('Feature'))
    else:
        st.write("Feature importances not available for this model.")

    # Predictions table with timestamp
    st.subheader("📋 Predictions vs Actual")
    pred_df = pd.DataFrame()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if len(target_cols) == 1:
        col = target_cols[0]
        pred_df[f"Actual_{col}"] = y_test.values.flatten()
        pred_df[f"Predicted_{col}"] = y_pred.flatten()
    else:
        for i, col in enumerate(target_cols):
            pred_df[f"Actual_{col}"] = y_test.iloc[:, i].values
            pred_df[f"Predicted_{col}"] = y_pred[:, i]
    pred_df["Timestamp"] = timestamp
    st.dataframe(pred_df)

    # System resource usage
    st.subheader("⚙️ System Resource Usage")
    st.write(f"CPU Usage: {psutil.cpu_percent()}%")
    st.write(f"Memory Usage: {psutil.virtual_memory().percent}%")
    st.write(f"System Platform: {platform.system()} {platform.release()}")
