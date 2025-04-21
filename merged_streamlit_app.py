import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
from tpot import TPOTRegressor
import os
import shap

# Page config
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

# Sidebar for feature and target column selection
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect("Select features for prediction", data.columns.tolist(), default=data.columns.tolist()[:-1])

# Set default target columns dynamically based on the available columns
default_target_cols = ["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"]
target_cols = st.sidebar.multiselect("Select target columns", data.columns.tolist(), default=[col for col in default_target_cols if col in data.columns])

if not features or not target_cols:
    st.error("Please select at least one feature and one target column.")
    st.stop()

# Exclude 'date' column from features if present
if 'date' in features:
    features.remove('date')

# Scale the input features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])
X = pd.DataFrame(scaled_features, columns=features)
y = data[target_cols] if len(target_cols) > 1 else data[[target_cols[0]]]

# Sidebar for model training parameters
st.sidebar.header("Model Training")
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest", "XGBoost", "AutoML (TPOT)"])

n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

# Train the model when button is clicked
if st.sidebar.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLflow logging and experiment tracking
    mlflow.start_run()
    mlflow.log_param("model_choice", model_choice)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Timer to track model training time
    start_time = time.time()

    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_choice == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_choice == "AutoML (TPOT)":
        model = TPOTRegressor( generations=5, population_size=20, random_state=42, verbosity=2)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Log model and metrics with MLflow
    mlflow.log_metric("training_time", time.time() - start_time)
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))

    mlflow.sklearn.log_model(model, "model")

    # Display model training time
    st.write(f"Model training time: {time.time() - start_time:.2f} seconds")

    # Display evaluation results
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.subheader("Model Evaluation")
    st.write(f"MAE: {mae:.3f}")
    st.write(f"RMSE: {rmse:.3f}")
    st.write(f"R² Score: {r2:.3f}")

    # Feature importances (if applicable)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        feature_importances = np.zeros(X_train.shape[1])

    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    st.subheader("Feature Importances")
    st.write(feature_importance_df)
    st.bar_chart(feature_importance_df.set_index('Feature')['Importance'])

    # Prediction vs Actual comparison table
    pred_df = pd.DataFrame({
        'Actual': y_test.values.flatten(),
        'Predicted': y_pred.flatten()
    })

    st.subheader("Predictions vs Actual")
    st.dataframe(pred_df)

    # End MLflow logging
    mlflow.end_run()

    # AutoML Button (optional)
    if st.sidebar.button("Run AutoML"):
        st.write("Running AutoML with TPOT...")
        automl_model = TPOTRegressor( generations=5, population_size=20, random_state=42)
        automl_model.fit(X_train, y_train)
        st.write("Best Model: ", automl_model.fitted_pipeline_)
        st.write("AutoML Training completed.")

    # Chatbot explainer
    st.subheader("Chatbot Explainer")
    st.write("Have questions about the model? Ask below!")
    user_input = st.text_input("Ask a question")
    if user_input:
        st.write(f"Model explanation for '{user_input}': The model uses {model_choice} to predict energy output based on the features you've selected.")
