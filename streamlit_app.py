import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os


# Load the trained model with caching
@st.cache_resource
def load_model():
    try:
        return joblib.load("./model_top5.pkl")  # Load the updated model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def load_data():
    try:
        return pd.read_csv("./df.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    # Define only the top 5 features
REQUIRED_COLUMNS = [
        "year_month_2024-08", # One-hot encoded feature
        "year_month_2024-06"
        "total_visits",
        "avg_days_between_pickups",
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

# Set the background image (optional)
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def powerbi_dashboard():
    st.title("Power BI Dashboard")

    # Path to the PDF file in the repository
    powerbi_link= "https://app.powerbi.com/view?r=eyJrIjoiMTE4Y2JiYWQtMzNhYS00NGFiLThmMDQtMmIwMDg4YTIzMjI5IiwidCI6ImUyMjhjM2RmLTIzM2YtNDljMy05ZDc1LTFjZTI4NWI1OWM3OCJ9" 
    
    
    # Embed the Power BI dashboard using an iframe
    components.html(
        f"""
        <iframe
            width="100%"
            height="1200"
            src="{powerbi_link}"
            frameborder="0"
            allowFullScreen="true">
        </iframe>
        """,
        height=800,
    )

def prediction_page():
    # Add the header image
    header_image_url = "https://raw.githubusercontent.com/ChiomaUU/Client-Prediction/refs/heads/main/ifssa_2844cc71-4dca-48ae-93c6-43295187e7ca.avif"
    st.image(header_image_url, use_container_width=True)  # Updated parameter

    st.title("Client Return Prediction App")
    st.write("Enter details to predict if a client will return.")

    # Load the dataset
    data = load_data()

    # Display the dataset (optional)
    if st.checkbox("Show raw data"):
        st.write(data)

    # User input fields (matching the top 5 important features)
    year_month = st.selectbox("Year-Month", ["2024-08", "2024-07", "2024-06"])
    total_visits = st.number_input("Total Visits", min_value=1, max_value=100, step=1)
    avg_days_between_pickups = st.number_input("Avg Days Between Pickups", min_value=1.0, max_value=100.0, step=0.1)
    days_since_last_pickup = st.number_input("Days Since Last Pickup", min_value=0, step=1)

    # Prepare input data
    input_data = {
        "year_month_2024-08": 1 if year_month == "2024-08" else 0,  # One-hot encoding for year-month
        "year_month_2024-06": 1 if year_month == "2024-06" else 0,  # One-hot encoding for year-month
        "total_visits": total_visits,
        "avg_days_between_pickups": avg_days_between_pickups,
        "days_since_last_pickup": days_since_last_pickup
    }

    # Prediction button
    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded. Please check if 'model_top5.pkl' exists.")
        else:
            st.write(f"Model Expected Features: {model.n_features_in_}")
            input_df = preprocess_input(input_data)
            st.write("Processed Input Data:")
            st.write(input_df)
            st.write(f"Shape of Input Data: {input_df.shape}")
            st.write(f"Columns in Input Data: {input_df.columns.tolist()}")

            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)

            st.subheader("Prediction Result:")
            st.write("✅ Prediction: **Yes**" if prediction[0] == 1 else "❌ Prediction: **No**")
            st.write(f"📊 Probability (Yes): **{probability[0][1]:.4f}**")
            st.write(f"📊 Probability (No): **{probability[0][0]:.4f}**")

# Main function to handle multi-page navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Power BI Dashboard"])

    if page == "Prediction":
        prediction_page()
    elif page == "Power BI Dashboard":
        powerbi_dashboard()

# Run the app
if __name__ == "__main__":
    main()

