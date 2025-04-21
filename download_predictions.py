# download_predictions.py
import pandas as pd
import streamlit as st
import numpy as np

def show_and_download_predictions():
    # Example predictions, replace with actual predictions
    predictions = np.random.rand(10, 5)  # Replace with your actual model predictions
    df = pd.DataFrame(predictions, columns=['Prediction 1', 'Prediction 2', 'Prediction 3', 'Prediction 4', 'Prediction 5'])

    # Display the dataframe
    st.write("Predictions:")
    st.dataframe(df)

    # Convert to CSV
    csv = df.to_csv(index=False)

    # Add the download button
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
