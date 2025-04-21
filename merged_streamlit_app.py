import streamlit as st
import pandas as pd
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Page config
st.set_page_config(page_title="Renewable Energy Predictor", layout="wide")
st.title("🔋 Renewable Energy Production Simulator")

# Sidebar for file upload
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.subheader("📊 Raw Data")
    st.dataframe(data)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Sidebar for simulation controls
st.sidebar.header("🔧 Simulation Controls")
include_components = st.sidebar.multiselect("Toggle Energy Components", ["Solar", "Wind", "Hydro", "Geothermal", "Battery", "CHP", "Carbon Capture", "AI Optimization"], default=["Solar", "Battery"])
outage_sim = st.sidebar.checkbox("Simulate Grid Outage")
modify_inputs = st.sidebar.slider("Modify Demand (Energy Consumption)", 0.5, 2.0, 1.0, step=0.1)

# Sidebar for feature and target column selection
st.sidebar.header("📈 Feature Selection")
features = st.sidebar.multiselect("Select features for prediction", data.columns.tolist(), default=data.columns.tolist()[:-1])
available_target_cols = ["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"]
target_cols = st.sidebar.multiselect("Select target columns", data.columns.tolist(), default=[col for col in available_target_cols if col in data.columns])

if not features or not target_cols:
    st.error("Please select at least one feature and one target column.")
    st.stop()

# Apply modifications
data['energy_consumption'] *= modify_inputs
if outage_sim:
    data['grid_draw'] = 0

# Scale the input features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])
X = pd.DataFrame(scaled_features, columns=features)

# Ensure that target `y` corresponds to the same rows as `X`
y = data[target_cols].iloc[:X.shape[0]]  # Select only as many rows as X has

# Align by resetting the index (this is a safety measure)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Now check the shapes
st.write("Shape of X:", X.shape)
st.write("Shape of y:", y.shape)

# Sidebar for model training
st.sidebar.header("🤖 Model Training")
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest", "XGBoost", "Stacking", "Deep Learning", "AutoML (TPOT)"])
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

if st.sidebar.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlflow.start_run()
    mlflow.log_param("model_choice", model_choice)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    start_time = time.time()

    # Grid Search for Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_model = RandomForestRegressor(random_state=42)
    grid_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=3, scoring='r2', verbose=2)
    grid_rf.fit(X_train, y_train)
    st.write(f"Best Random Forest parameters: {grid_rf.best_params_}")

    model = grid_rf.best_estimator_  # Use the best model from grid search

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_metric("training_time", time.time() - start_time)
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")

    st.success(f"Model training completed in {time.time() - start_time:.2f} seconds")
    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    st.metric("R²", f"{r2_score(y_test, y_pred):.3f}")

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_df = pd.DataFrame({"Feature": features, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)
        st.subheader("📌 Feature Importances")
        st.dataframe(feature_df)
        st.bar_chart(feature_df.set_index("Feature"))

    # Predictions vs Actual
    pred_df = pd.DataFrame({"Actual": y_test.values.flatten(), "Predicted": y_pred.flatten()})
    st.dataframe(pred_df)
    fig, ax = plt.subplots()
    sns.scatterplot(x=pred_df['Actual'], y=pred_df['Predicted'], ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

    mlflow.end_run()

# Chatbot Explainer
st.subheader("🤖 Chatbot Explainer")
user_input = st.text_input("Ask about the model or simulation")
if user_input:
    if "model" in user_input.lower():
        st.write("We use Random Forest, XGBoost, Deep Learning, and AutoML to predict energy outcomes.")
    elif "simulate" in user_input.lower():
        st.write("Simulation supports component toggling, outage scenarios, and real-time demand changes.")
    elif "feature" in user_input.lower():
        st.write("Features are the inputs used for prediction, such as energy type metrics and system states.")
    elif "target" in user_input.lower():
        st.write("Target columns are the outputs we predict like cost, CO2 captured, or hydrogen produced.")
    else:
        st.write("I'm here to help! Try asking about 'model', 'simulate', 'feature', or 'target'.")
