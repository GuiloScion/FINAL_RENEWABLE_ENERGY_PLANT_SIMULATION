import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
import time
import datetime
import seaborn as sns
import psutil
import platform
import os
import requests
import shap
from sklearn.ensemble import IsolationForest
import mlflow

# Ensure the mlruns directory exists, create if necessary
mlruns_dir = './mlruns'
os.makedirs(mlruns_dir, exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file://{mlruns_dir}")
mlflow.set_experiment("Renewable_Simulation")

st.set_page_config(page_title="Renewable Energy Predictor", layout="wide")
st.title("🔋 Renewable Energy Production Predictor")

# Sidebar for real‑time weather & location
st.sidebar.header("Location & Weather")
lat = st.sidebar.number_input("Latitude", value=39.0, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=-105.5, format="%.6f")
owm_key = st.secrets.get("OWM_KEY", "")
weather = {}
if owm_key:
    resp = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
        f"&appid={owm_key}&units=metric"
    ).json()
    if resp.get("main"):
        weather["temp"]  = resp["main"]["temp"]
        weather["cloud"] = resp["clouds"]["all"]
        weather["wind"]  = resp["wind"]["speed"]
st.sidebar.write("**Weather**")
st.sidebar.write(weather or "No data")

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

# Sidebar for scenario & outage simulation
st.sidebar.header("Scenario & Outage")
battery_avail  = st.sidebar.checkbox("Battery Online", value=True)
inverter_perf  = st.sidebar.slider("Inverter Efficiency", 0.0, 1.0, 1.0)
storm_impact   = st.sidebar.slider("Storm Severity (%)", 0, 100, 0)

# Sidebar feature/target selection
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect(
    "Select features for prediction",
    data.columns.tolist(),
    default=data.columns.tolist()[:-1]
)

# Default target columns
default_target_cols = [
    "cost_per_kWh", "energy_consumption", "energy_output",
    "operating_costs", "co2_captured", "hydrogen_production"
]
available_target_cols = [c for c in default_target_cols if c in data.columns]
target_cols = st.sidebar.multiselect(
    "Select target columns",
    data.columns.tolist(),
    default=available_target_cols
)

# Warn about missing targets
missing = [c for c in target_cols if c not in data.columns]
if missing:
    st.warning(f"The following target columns are missing: {', '.join(missing)}")

if not features or not target_cols:
    st.error("Please select at least one feature and one target column.")
    st.stop()

if 'date' in features:
    features.remove('date')

# Prepare data
# inject weather
if "temp" in weather:
    data["temp"] = weather["temp"]
    features.append("temp")
# outage logic
if not battery_avail and "battery_output" in data.columns:
    data["battery_output"] = 0
if "solar_output" in data.columns:
    data["solar_output"] = data["solar_output"] * (1 - storm_impact/100)

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)

# y as array (1D) or DataFrame (multi-output)
if len(target_cols) == 1:
    y = data[target_cols[0]].values  # 1D array
else:
    y = data[target_cols]            # DataFrame for multi-output

# Sidebar for AutoML button
st.sidebar.header("Optimization")
autoopt = st.sidebar.button("Auto‑Optimize")
best_params = {}
if autoopt:
    with st.spinner("Searching best parameters..."):
        param_dist = {"n_estimators": [50,100,200], "max_depth": [5,10,20]}
        rs = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_distributions=param_dist,
            n_iter=5, cv=3, random_state=42
        )
        rs.fit(X, y)
        best_params = rs.best_params_
    st.sidebar.success(f"Best params: {best_params}")

# Sidebar model parameters
st.sidebar.header("Model Training")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "XGBoost"]
)
n_estimators = best_params.get("n_estimators") or st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth    = best_params.get("max_depth")    or st.sidebar.slider("Max Depth", 1, 20, 10)

if st.sidebar.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Wrap for multi-output if needed
    if len(target_cols) > 1:
        base = {
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
            "XGBoost": XGBRegressor
        }[model_choice]
        model = MultiOutputRegressor(
            base(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        )
    else:
        if model_choice == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        else:
            model = XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_param("model_choice", model_choice)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        if len(target_cols) == 1:
            y_test_arr, y_pred_arr = y_test, y_pred
        else:
            y_test_arr, y_pred_arr = y_test.values, y_pred

        mae  = mean_absolute_error(y_test_arr, y_pred_arr)
        rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
        r2   = r2_score(y_test_arr, y_pred_arr)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

    # SHAP Explainability & Anomaly Detection
    st.subheader("Explainability & Anomalies")
    explainer = shap.TreeExplainer(model.estimators_[0]
                                   if hasattr(model, "estimators_") else model)
    shap_values = explainer.shap_values(X_test)
    fig_shap = shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig_shap)

    iso = IsolationForest(contamination=0.01)
    preds_iso = iso.fit_predict(
        y_pred_arr.reshape(-1,1) if len(target_cols)==1 else y_pred_arr
    )
    anomalies = np.where(preds_iso == -1)[0]
    st.write("Anomalous indices:", anomalies)

    # Model Evaluation
    st.subheader("Model Evaluation")
    st.metric("🧮 MAE", f"{mae:.3f}")
    st.metric("📉 RMSE", f"{rmse:.3f}")
    st.metric("📈 R² Score", f"{r2:.3f}")
    st.metric("⏱️ Training Time", f"{training_time:.2f} seconds")

    # Summary Report
    st.subheader("📊 Model Summary Report")
    st.json({
        "Model": model_choice,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Training Time (s)": training_time,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Residual Error Analysis
    st.subheader("Residual Error Analysis")
    resid = (y_test_arr - y_pred_arr).ravel()
    fig_res, ax = plt.subplots()
    sns.histplot(resid, bins=30, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    st.pyplot(fig_res)

    # Feature Importances Table
    st.subheader("🔍 Feature Importances")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.mean(
            [e.feature_importances_ for e in model.estimators_],
            axis=0
        )
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    st.dataframe(imp_df)

    # Predictions vs Actual
    st.subheader("📋 Predictions vs Actual")
    pred_df = pd.DataFrame(
        y_test_arr, columns=target_cols
    ) if len(target_cols)>1 else pd.DataFrame({target_cols[0]: y_test_arr})
    pred_df[[f"Pred_{c}" for c in target_cols]] = (
        y_pred_arr if len(target_cols)>1 else y_pred_arr.reshape(-1,1)
    )
    pred_df["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(pred_df)

    # System Resource Usage
    st.subheader("⚙️ System Resource Usage")
    st.write(f"CPU Usage: {psutil.cpu_percent()}%")
    st.write(f"Memory Usage: {psutil.virtual_memory().percent}%")
    st.write(f"System Platform: {platform.system()} {platform.release()}")
