import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model with caching
@st.cache_resource
def load_model():
    try:
        return joblib.load("model_top5.pkl")  # Load the updated model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define only the top 5 features
REQUIRED_COLUMNS = [
    "year_month_2024-08",  # One-hot encoded feature
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
def exploratory_data_analysis():
    st.subheader("Infograph of Clients")
    st.write("This page is currently not active")
    
def predictions_page():
    # Streamlit app UI
    st.title("Hamper Return Prediction App")
    st.write("Enter details to predict if a client will return.")
    
    # User input fields (matching the top 5 important features)
    year_month = st.selectbox("Year-Month", ["2024-08", "2024-07", "2024-06"])
    total_visits = st.number_input("Total Visits", min_value=1, max_value=100, step=1)
    avg_days_between_pickups = st.number_input("Avg Days Between Pickups", min_value=1.0, max_value=100.0, step=0.1)
    month = st.number_input("Month", min_value=1, max_value=12, step=1)
    days_since_last_pickup = st.number_input("Days Since Last Pickup", min_value=0, step=1)
    
    # Prepare input data
    input_data = {
        "year_month_2024-08": 1 if year_month == "2024-08" else 0,  # One-hot encoding for year-month
        "total_visits": total_visits,
        "avg_days_between_pickups": avg_days_between_pickups,
        "month": month,
        "days_since_last_pickup": days_since_last_pickup,
    }
    
    # Prediction button
    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded. Please check if 'model_top5.pkl' exists.")
        else:
            input_df = preprocess_input(input_data)
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
    
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

# Main function to control the app
def main():
    st.sidebar.title("Navigation")
    app_page = st.sidebar.radio("Choose a page", ["Dashboard", "Infograph", "Predictions"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Infograph":
        exploratory_data_analysis()
    elif app_page == "Predictions":
        predictions_page()

# Run the app
if __name__ == "__main__":
    main()
