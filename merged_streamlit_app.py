# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# Sidebar controls
st.sidebar.title("Model Controls")
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Gradient Boosting"])
n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 50, 5)
learning_rate = st.sidebar.slider("Learning Rate (GB only)", 0.01, 0.5, 0.1)

# Load and scale data
@st.cache_data
def load_data():
    # Dummy data for demo
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"Feature_{i}" for i in range(5)])
    y = pd.DataFrame({
        "grid_draw": np.random.rand(100),
        "energy_output": np.random.rand(100),
        "energy_consumption": np.random.rand(100),
        "cost": np.random.rand(100)
    })
    return X, y

X, y = load_data()
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model initialization
if model_choice == "Random Forest":
    base_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
else:
    base_model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                           max_depth=max_depth, random_state=42)

# FIX: Wrap in MultiOutputRegressor if y has multiple outputs
if y_train.ndim > 1 and y_train.shape[1] > 1:
    model = MultiOutputRegressor(base_model)
else:
    model = base_model

# Training and evaluation
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        r2 = [r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

    return y_pred, mae, rmse, r2

y_pred, mae, rmse, r2 = train_and_evaluate(model, X_train, y_train, X_test, y_test)

# Results
st.subheader("Model Evaluation Metrics")
if isinstance(mae, np.ndarray):
    for i, col in enumerate(y.columns):
        st.write(f"**{col}**")
        st.write(f"MAE: {mae[i]:.4f}")
        st.write(f"RMSE: {rmse[i]:.4f}")
        st.write(f"R²: {r2[i]:.4f}")
else:
    st.write(f"MAE: {mae:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"R²: {r2:.4f}")

# Predictions
pred_df = pd.DataFrame(y_pred, columns=y.columns)
st.subheader("Predicted Outputs")
st.dataframe(pred_df)

# Optional: Plotting
st.subheader("Prediction vs Actual")
for col in y.columns:
    fig, ax = plt.subplots()
    ax.plot(y_test[col].values, label='Actual', marker='o')
    ax.plot(pred_df[col].values, label='Predicted', marker='x')
    ax.set_title(col)
    ax.legend()
    st.pyplot(fig)

# Energy efficiency suggestions
def energy_efficiency_suggestions(df):
    st.subheader("Energy Efficiency Suggestions")
    if 'energy_output' in df.columns and df['energy_output'].mean() < 0.5:
        st.write("🔋 Consider improving renewable energy integration for higher output.")
    if 'grid_draw' in df.columns and df['grid_draw'].mean() > 0.7:
        st.write("⚡ High grid draw detected. Investigate battery storage or load shifting.")
    if 'cost' in df.columns and df['cost'].mean() > 0.6:
        st.write("💰 Cost is relatively high. Explore operational optimization strategies.")

energy_efficiency_suggestions(pred_df)
