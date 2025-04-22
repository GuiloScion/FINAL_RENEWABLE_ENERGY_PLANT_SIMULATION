
# --- Addition: Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import platform
import psutil
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib

# --- Addition: Sidebar Controls ---
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100, 10)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

# --- Original Code Starts Here (Unmodified) ---
#else:
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

# Feature importances table (graph removed)
st.subheader("🔍 Feature Importances")
feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
if feature_importances is not None:
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(importance_df)
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

# --- Additions Start Here ---

# Download predictions
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime='text/csv')

# Prediction error scatter plot
st.subheader("📌 Prediction Error Scatter Plot")
fig2, ax2 = plt.subplots()
ax2.scatter(y_test.values.flatten(), y_pred.flatten(), alpha=0.6)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Actual vs. Predicted")
st.pyplot(fig2)

# Cross-validation
cv_score = cross_val_score(model, X_train, y_train, scoring='r2', cv=5).mean()
st.metric("🧪 Cross-Validated R²", f"{cv_score:.3f}")

# Model save/load
st.subheader("💾 Model Persistence")
col1, col2 = st.columns(2)
with col1:
    if st.button("Save Model"):
        joblib.dump(model, "best_model.pkl")
        st.success("Model saved!")
with col2:
    if st.button("Load Model"):
        model = joblib.load("best_model.pkl")
        st.success("Model loaded!")

# Inference timing
start_pred_time = time.time()
_ = model.predict(X_test)
prediction_time = time.time() - start_pred_time
st.metric("⚡ Prediction Time", f"{prediction_time:.2f} seconds")

# Show raw data toggle
if st.checkbox("Show Raw Test Data"):
    st.dataframe(X_test)
