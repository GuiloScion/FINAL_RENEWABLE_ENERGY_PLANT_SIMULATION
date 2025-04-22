import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
import mlflow
import time
import datetime
import seaborn as sns
import psutil
import platform
import os

# ─── Optional streamlit_folium import ────────────────────────────────────────
try:
    from streamlit_folium import st_folium
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# ─── APP CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Renewable Energy Predictor", layout="wide")
st.title("🔋 Renewable Energy Production Predictor")

# Initialize MLflow
mlflow.set_tracking_uri("file://./mlruns")
mlflow.set_experiment("Renewable_Simulation")

# ─── 1. REAL‑TIME WEATHER & LOCATION ──────────────────────────────────────────
st.sidebar.header("Location & Weather")

# Fallback inputs for lat/lon
if not FOLIUM_AVAILABLE:
    st.sidebar.warning("Install streamlit_folium for map picker")
    lat = st.sidebar.number_input("Latitude", value=39.0, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=-105.5, format="%.6f")
else:
    # Map picker
    lat0, lon0 = 39.0, -105.5
    m = folium.Map(location=[lat0, lon0], zoom_start=6)
    map_out = st.sidebar.expander("Select Site Location")
    with map_out:
        clicked = st_folium(m, width=300, height=200)
        if clicked and clicked.get("last_clicked"):
            lat = clicked["last_clicked"]["lat"]
            lon = clicked["last_clicked"]["lng"]
        else:
            lat, lon = lat0, lon0

# Fetch weather via OpenWeatherMap
owm_key = st.secrets.get("OWM_KEY", "")
weather = {}
if owm_key:
    try:
        resp = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
            f"&appid={owm_key}&units=metric"
        ).json()
        weather = {
            "temp":  resp["main"]["temp"],
            "cloud": resp["clouds"]["all"],
            "wind":  resp["wind"]["speed"]
        }
    except Exception:
        weather = {}
st.sidebar.write("**Weather**")
st.sidebar.write(weather or "No data")

# ─── 2. DATA UPLOAD ───────────────────────────────────────────────────────────
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
@st.cache_data
def load_data(f): return pd.read_csv(f)

if uploaded_file:
    data = load_data(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(data)
else:
    st.warning("Upload a CSV to proceed.")
    st.stop()

# ─── 3. SCENARIO & OUTAGE ─────────────────────────────────────────────────────
st.sidebar.header("Scenario & Outage")
battery_avail = st.sidebar.checkbox("Battery Online", value=True)
inverter_perf = st.sidebar.slider("Inverter Efficiency", 0.0, 1.0, 1.0)
storm_impact  = st.sidebar.slider("Storm Severity (%)", 0, 100, 0)

# ─── 4. FEATURE / TARGET SELECTION ────────────────────────────────────────────
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect("Features", data.columns.tolist(), default=data.columns.tolist()[:-1])

default_targets = [
    "cost_per_kWh", "energy_consumption", "energy_output",
    "operating_costs", "co2_captured", "hydrogen_production"
]
available_targets = [c for c in default_targets if c in data.columns]
target_cols = st.sidebar.multiselect("Targets", data.columns.tolist(), default=available_targets)
missing = [c for c in target_cols if c not in data.columns]
if missing:
    st.warning(f"Missing targets: {', '.join(missing)}")
if not features or not target_cols:
    st.error("Select at least one feature and one target.")
    st.stop()
if "date" in features:
    features.remove("date")

# ─── 5. DATA PREP & WEATHER INJECTION ─────────────────────────────────────────
# Inject weather into dataframe if available
if weather.get("temp") is not None:
    data["temp"] = weather["temp"]
    features.append("temp")

# Apply outage logic
if not battery_avail and "battery_output" in data.columns:
    data["battery_output"] = 0
# Storm reduces solar output
if "solar_output" in data.columns:
    data["solar_output"] = data["solar_output"] * (1 - storm_impact/100)

# Scale features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)

# Prepare target y
if len(target_cols) == 1:
    y = data[target_cols[0]].values
else:
    y = data[target_cols]

# ─── 6. AUTO‑OPTIMIZE ──────────────────────────────────────────────────────────
st.sidebar.header("Optimization")
autoopt = st.sidebar.button("Auto‑Optimize")
best_params = {}
if autoopt:
    with st.spinner("Searching ..."):
        param_dist = {"n_estimators":[50,100,200], "max_depth":[5,10,20]}
        rs = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_distributions=param_dist,
            n_iter=5, cv=3, random_state=42
        )
        rs.fit(X, y)
        best_params = rs.best_params_
    st.sidebar.success(f"Best params: {best_params}")

# ─── 7. MODEL TRAINING & MLflow ───────────────────────────────────────────────
st.sidebar.header("Model Training")
model_choice = st.sidebar.selectbox("Model", ["Random Forest","Gradient Boosting","XGBoost"])
n_est = best_params.get("n_estimators") or st.sidebar.slider("Trees", 10, 200, 100)
m_depth = best_params.get("max_depth")    or st.sidebar.slider("Max Depth", 1, 20, 10)

if st.sidebar.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build model (multi-output if needed)
    if len(target_cols) > 1:
        base_cls = {
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
            "XGBoost": XGBRegressor
        }[model_choice]
        estimator = base_cls(n_estimators=n_est, max_depth=m_depth, random_state=42)
        model = MultiOutputRegressor(estimator)
    else:
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=n_est, max_depth=m_depth, random_state=42)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=n_est, max_depth=m_depth, random_state=42)
        else:
            model = XGBRegressor(n_estimators=n_est, max_depth=m_depth, random_state=42)

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_param("model_choice", model_choice)
        mlflow.log_param("n_estimators", n_est)
        mlflow.log_param("max_depth", m_depth)

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

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

    # ─── 8. SHAP & ANOMALY DETECTION ──────────────────────────────────────────
    st.subheader("Explainability & Anomalies")
    explainer = shap.TreeExplainer(model.estimators_[0]
                                   if hasattr(model, "estimators_") else model)
    shap_values = explainer.shap_values(X_test)
    fig_shap = shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig_shap)

    iso = IsolationForest(contamination=0.01)
    preds_iso = iso.fit_predict(y_pred_arr.reshape(-1,1) if len(target_cols)==1 else y_pred_arr)
    anomalies = np.where(preds_iso == -1)[0]
    st.write("Anomalous indices:", anomalies)

    # ─── 9. MODEL EVALUATION & REPORT ────────────────────────────────────────
    st.subheader("Model Evaluation")
    st.metric("🧮 MAE", f"{mae:.3f}")
    st.metric("📉 RMSE", f"{rmse:.3f}")
    st.metric("📈 R² Score", f"{r2:.3f}")
    st.metric("⏱️ Training Time", f"{train_time:.2f} s")

    st.subheader("📊 Summary Report")
    st.json({
        "Model": model_choice,
        "Params": {"n_estimators": n_est, "max_depth": m_depth},
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Training Time": f"{train_time:.2f}s",
        "Run Date": datetime.datetime.now().isoformat()
    })

    # ─── 10. RESIDUAL ANALYSIS ────────────────────────────────────────────────
    st.subheader("Residual Error Analysis")
    resid = (y_test_arr - y_pred_arr).ravel()
    fig_res, ax = plt.subplots()
    sns.histplot(resid, bins=30, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    st.pyplot(fig_res)

    # ─── 11. FEATURE IMPORTANCES ─────────────────────────────────────────────
    st.subheader("🔍 Feature Importances")
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    else:
        imps = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
    imp_df = pd.DataFrame({"Feature": features, "Importance": imps}) \
               .sort_values("Importance", ascending=False)
    st.dataframe(imp_df)

    # ─── 12. PREDICTIONS TABLE ────────────────────────────────────────────────
    st.subheader("📋 Predictions vs Actual")
    df_pred = (pd.DataFrame(y_test_arr, columns=target_cols)
               if len(target_cols)>1
               else pd.DataFrame({target_cols[0]: y_test_arr}))
    df_pred[[f"Pred_{c}" for c in target_cols]] = (
        y_pred_arr if len(target_cols)>1 else y_pred_arr.reshape(-1,1)
    )
    df_pred["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(df_pred)

    # ─── 13. SYSTEM RESOURCE USAGE ───────────────────────────────────────────
    st.subheader("⚙️ System Resource Usage")
    st.write(f"CPU: {psutil.cpu_percent()}%")
    st.write(f"Memory: {psutil.virtual_memory().percent}%")
    st.write(f"System: {platform.system()} {platform.release()}")
