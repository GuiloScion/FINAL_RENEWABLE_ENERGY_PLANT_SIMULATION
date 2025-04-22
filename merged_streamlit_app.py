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
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return pd.DataFrame()

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data.empty:
        st.error("Uploaded file is empty or invalid. Please upload a valid CSV.")
        st.stop()

    st.subheader("Raw Data")
    st.dataframe(data)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Sidebar: Feature/Target Selection
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect("Select features for prediction", data.columns.tolist(), default=data.columns.tolist()[:-1])

# Define default target columns
default_target_cols = ["cost_per_kWh", "energy_consumption", "energy_output", "operating_costs", "co2_captured", "hydrogen_production"]

# Ensure default target columns exist in the dataset
available_target_cols = [col for col in default_target_cols if col in data.columns]

# Sidebar: Target Selection
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
st.sidebar.header("Model Training")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "XGBoost"])
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

# Train the model if button is clicked
if st.sidebar.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()

    try:
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.stop()

    training_time = time.time() - start_time

    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.metric("🧮 MAE", f"{mae:.3f}")
    st.metric("📉 RMSE", f"{rmse:.3f}")
    st.metric("📈 R² Score", f"{r2:.3f}")
    st.metric("⏱️ Training Time", f"{training_time:.2f} seconds")

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

    # Chatbot Section
    st.sidebar.header("🤖 Chatbot")
    user_input = st.sidebar.text_input("Ask me about the model:")

    def chatbot_response(user_input):
        user_input = user_input.lower()
        if "model" in user_input:
            return "The model predicts renewable energy production based on selected features and target columns. It uses machine learning algorithms to learn patterns from historical data."
        elif "accuracy" in user_input or "performance" in user_input:
            return "The model's performance is evaluated using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score. Higher R² scores indicate better predictive capability."
        elif "features" in user_input:
            return "You can select features like cost per kWh, energy consumption, and more. The choice of features greatly impacts the model's performance."
        elif "predictions" in user_input:
            return "Predictions are generated based on the trained model, which analyzes input features to estimate energy production outcomes."
        elif "training" in user_input:
            return "The model is trained using a portion of the data, typically 80%, while 20% is reserved for testing. This helps to evaluate how well the model generalizes to unseen data."
        elif "parameters" in user_input:
            return "You can adjust parameters like the number of trees in ensemble methods or the maximum depth of the trees to optimize results. Tuning these can significantly affect performance."
        elif "data" in user_input:
            return "The model uses historical data related to renewable energy production, including factors like weather conditions, energy costs, and consumption patterns."
        elif "application" in user_input:
            return "This model can help in predicting energy needs, optimizing costs, and improving energy management, which is essential for sustainable practices."
        elif "improve" in user_input:
            return "To improve the model, consider tuning hyperparameters, adding more features, or using ensemble methods. Cross-validation can also help in assessing model stability."
        elif "challenges" in user_input:
            return "Common challenges include overfitting, underfitting, and data quality issues. Ensuring that the data is clean and representative is crucial for model success."
        elif "explainable ai" in user_input:
            return "Explainable AI (XAI) focuses on making the decision-making process of models transparent. This is important for trust, especially in critical applications like energy management."
        else:
            return "I'm sorry, I can only answer questions about the model, its predictions, and related topics. Please ask something specific!"

    if user_input:
        response = chatbot_response(user_input)
        st.sidebar.text_area("Chatbot Response:", response, height=100)
