import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import shap
import time

st.set_page_config(page_title="Renewable Energy Predictor", layout="wide")
st.title("🔋 Renewable Energy Production Predictor")

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

# Set default target columns if they exist in the uploaded data
default_target_cols = ["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"]
target_cols = st.sidebar.multiselect("Select target columns", data.columns.tolist(), default=[col for col in default_target_cols if col in data.columns.tolist()])

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

# Sidebar for model choice and hyperparameter tuning
st.sidebar.header("Model Selection & Hyperparameter Tuning")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "XGBoost", "SVR"])

# Define parameter grids for different models
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# Train the model and evaluate
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test.values.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_test.values.flatten(), y_pred.flatten()))
    r2 = r2_score(y_test.values.flatten(), y_pred.flatten())

    return y_pred, mae, rmse, r2

# Model comparison
metrics = {}

if st.sidebar.button("Train and Compare Models"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }

    for model_name, model in models.items():
        y_pred, mae, rmse, r2 = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        metrics[model_name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
    
    # Display model comparison
    metrics_df = pd.DataFrame(metrics).T
    st.subheader("Model Comparison")
    st.write(metrics_df)

# Hyperparameter tuning
if st.sidebar.button("Train Model with Hyperparameter Tuning"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set up GridSearchCV or RandomizedSearchCV
    if model_choice in param_grids:
        param_grid = param_grids[model_choice]
        if model_choice == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)

        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        
        best_params = search.best_params_
        best_model = search.best_estimator_

        y_pred, mae, rmse, r2 = train_and_evaluate(best_model, X_train, y_train, X_test, y_test)

        st.subheader(f"Best Parameters for {model_choice}: {best_params}")
        st.write(f"MAE: {mae:.3f}")
        st.write(f"RMSE: {rmse:.3f}")
        st.write(f"R² Score: {r2:.3f}")

        # SHAP explainability (only for tree-based models)
        if model_choice in ["Random Forest", "Gradient Boosting"]:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

            # Plot SHAP summary plot
            st.subheader(f"SHAP Summary Plot for {model_choice}")
            shap.summary_plot(shap_values, X_test)

# Show predictions vs actual (labeled)
if st.sidebar.button("Show Predictions vs Actual"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = models.get(model_choice, RandomForestRegressor())  # Default to RandomForest
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame()
    for i, col in enumerate(target_cols):
        pred_df[f"Actual_{col}"] = y_test.iloc[:, i].values
        pred_df[f"Predicted_{col}"] = y_pred[:, i]

    st.subheader("Predictions vs Actual (labeled)")
    st.dataframe(pred_df)

# Residual error analysis
if st.sidebar.button("Residual Error Analysis"):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    st.subheader("Residuals vs Predicted")
    st.write("A residual plot helps us identify non-random patterns in the errors of the model.")
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)

# Model summary
if st.sidebar.button("Model Summary"):
    st.subheader("Model Summary Report")
    st.write("This is a simple model summary, showing key metrics and the hyperparameters used for training.")
    st.write(f"Model Type: {model_choice}")
    st.write(f"Hyperparameters: {param_grids.get(model_choice, {})}")

# Energy efficiency suggestions
def energy_efficiency_suggestions(predictions, threshold=0.8):
    # Ensure necessary columns are in the DataFrame
    if "energy_consumption" in predictions.columns:
        high_consumption = predictions["energy_consumption"] > threshold * predictions["energy_consumption"].max()
        
        if high_consumption.any():
            st.subheader("Energy Efficiency Suggestions:")
            st.write("Based on the predictions, the following actions can help reduce energy consumption:")
            st.write("- Install energy-efficient lighting systems")
            st.write("- Consider investing in renewable energy sources like solar power")
            st.write("- Upgrade insulation in buildings to reduce heating/cooling costs")
        else:
            st.write("Energy consumption is within expected limits.")
    else:
        st.warning("The 'energy_consumption' column is missing in the predictions.")

# Ensure predictions DataFrame contains necessary columns
if 'energy_consumption' in pred_df.columns:
    energy_efficiency_suggestions(pred_df)
else:
    st.warning("Energy consumption data is not available for efficiency suggestions.")
