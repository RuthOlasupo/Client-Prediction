import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

# Load the trained model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_top5.pkl")  # Load the updated model
        scaler = joblib.load("scaler_top5.pkl")  # Load the scaler
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model()

# Define only the top 5 features
REQUIRED_COLUMNS = [
    "year_month",  # One-hot encoded feature
    "total_visits",
    "avg_days_between_pickups",
    "month",
    "days_since_last_pickup"
]

# Function to preprocess input data
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])

    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0  # Set missing columns to 0

    # Ensure the column order matches model training
    input_df = input_df[REQUIRED_COLUMNS]

    return input_df

# Insights Page 1: Power BI Visualization
def exploratory_data_analysis():
    st.title("Hamper Collection Insights")
    st.header("Power BI Visualization")
    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiMTE4Y2JiYWQtMzNhYS00NGFiLThmMDQtMmIwMDg4YTIzMjI5IiwidCI6ImUyMjhjM2RmLTIzM2YtNDljMy05ZDc1LTFjZTI4NWI1OWM3OCJ9"
    st.components.v1.iframe(powerbi_url, width=800, height=600)

# Insights Page 2: SHAP Summary Plot
def shap_summary_plot():
    st.subheader("SHAP Summary Plot")
    image = Image.open("shap_summary_plot.png")
    st.image(image, caption="SHAP Summary Plot", use_container_width=True)

# Predictions Page
def predictions_page():
    st.title("Hamper Return Prediction App")
    st.write("Enter details to predict if a client will return.")

    # User input fields (matching the top 5 important features)
    year_month = st.selectbox("Year-Month", ["2024-08", "2024-09", "2024-10", "2024-11", "2025-12""])
    total_visits = st.number_input("Total Visits", min_value=1, max_value=100, step=1)
    avg_days_between_pickups = st.number_input("Avg Days Between Pickups", min_value=1.0, max_value=100.0, step=0.1)
    month = st.number_input("Month", min_value=1, max_value=12, step=1)
    days_since_last_pickup = st.number_input("Days Since Last Pickup", min_value=0, step=1)

    # Prepare input data
    input_data = {
        "year_month": year_month,
        "total_visits": total_visits,
        "avg_days_between_pickups": avg_days_between_pickups,
        "month": month,
        "days_since_last_pickup": days_since_last_pickup
    }

    # Prediction button
    if st.button("Predict"):
        if model is None or scaler is None:
            st.error("Model or scaler not loaded. Please check if 'model_top5.pkl' and 'scaler.pkl' exist.")
        else:
            # Preprocess input data
            input_df = preprocess_input(input_data)

            # Scale the input data using the loaded scaler
            input_scaled = scaler.transform(input_df)

            # Make the prediction using the scaled data
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)

            st.subheader("Prediction Result:")
            st.write("✅ Prediction: **Yes**" if prediction[0] == 1 else "❌ Prediction: **No**")
            st.write(f"📊 Probability (Yes): **{probability[0][1]:.4f}**")
            st.write(f"📊 Probability (No): **{probability[0][0]:.4f}**")

# Dashboard Page
def dashboard():
    header_image_url = "https://raw.githubusercontent.com/ChiomaUU/Client-Prediction/refs/heads/main/ifssa_2844cc71-4dca-48ae-93c6-43295187e7ca.avif"
    st.image(header_image_url, use_container_width=True)  # Display the image at the top
    st.title("Hamper Return Prediction App")
    st.write("This app predicts whether a client will return for food hampers.")

# Insights Navigation
def insights_navigation():
    if 'page' not in st.session_state:
        st.session_state.page = 1

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Previous") and st.session_state.page > 1:
            st.session_state.page -= 1
    with col2:
        if st.button("Next ➡️"):
            st.session_state.page += 1

    # Display the current page
    if st.session_state.page == 1:
        exploratory_data_analysis()
    elif st.session_state.page == 2:
        shap_summary_plot()

# Main function to control the app
def main():
    st.sidebar.title("Navigation")
    app_page = st.sidebar.radio("Choose a page", ["Dashboard", "Insights", "Predictions"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Insights":
        insights_navigation()
    elif app_page == "Predictions":
        predictions_page()

# Run the app
if __name__ == "__main__":
    main()
