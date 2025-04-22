import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
import datetime
import seaborn as sns
import psutil
import platform
import os
import tempfile
import mlflow

# Ensure MLflow can write to a temporary directory if '/absolute/path/to/mlruns' doesn't work
try:
    # Try creating 'mlruns' in the current working directory
    mlruns_dir = './mlruns'  # Use relative path if the absolute one failed
    os.makedirs(mlruns_dir, exist_ok=True)  # Create the directory if it doesn't exist
except PermissionError:
    # Fall back to using a temporary directory
    mlruns_dir = tempfile.mkdtemp()  # Use temporary directory
    st.warning(f"Unable to write to the specified directory. Using a temporary directory: {mlruns_dir}")

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
mlflow.set_experiment("Renewable_Simulation")

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
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)

# y as array (1D) or DataFrame (multi-output)
if len(target_cols) == 1:
    y = data[target_cols[0]].values  # 1D array
else:
    y = data[target_cols]            # DataFrame for multi-output

# Sidebar model parameters
st.sidebar.header("Model Training")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "XGBoost"]
)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

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

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    # Ensure shapes align for metrics
    if len(target_cols) == 1:
        y_test_arr = y_test
        y_pred_arr = y_pred
    else:
        y_test_arr = y_test.values
        y_pred_arr = y_pred

    mae = mean_absolute_error(y_test_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
    r2  = r2_score(y_test_arr, y_pred_arr)

    # Model evaluation display
    st.subheader("Model Evaluation")
    st.metric("🧮 MAE", f"{mae:.3f}")
    st.metric("📉 RMSE", f"{rmse:.3f}")
    st.metric("📈 R² Score", f"{r2:.3f}")
    st.metric("⏱️ Training Time", f"{training_time:.2f} s")

    # Performance badge
    if r2 >= 0.9:
        st.success("✅ Excellent Model Performance")
    elif r2 >= 0.75:
        st.info("ℹ️ Good Model Performance")
    else:
        st.warning("⚠️ Model Needs Improvement")

    # Summary report
    st.subheader("📊 Model Summary Report")
    st.json({
        "Model": model_choice,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Training Time (s)": training_time,
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Residual analysis
    st.subheader("Residual Error Analysis")
    resid = (y_test_arr - y_pred_arr).ravel()
    fig, ax = plt.subplots()
    sns.histplot(resid, bins=30, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    st.pyplot(fig)

    # Feature importances table
    st.subheader("🔍 Feature Importances")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # MultiOutputRegressor stores estimators_
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_],
            axis=0
        )
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    st.dataframe(imp_df)

    # Predictions vs Actual
    st.subheader("📋 Predictions vs Actual")
    pred_df = pd.DataFrame(
        y_test_arr if len(target_cols)==1 else y_test_arr,
        columns=target_cols
    )
    pred_df[[f"Pred_{c}" for c in target_cols]] = (
        y_pred_arr if len(target_cols)==1 else y_pred_arr
    )
    pred_df["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(pred_df)

    # System resource usage
    st.subheader("⚙️ System Resource Usage")
    st.write(f"CPU Usage: {psutil.cpu_percent()}%")
    st.write(f"Memory Usage: {psutil.virtual_memory().percent}%")
    st.write(f"System Platform: {platform.system()} {platform.release()}")
