import streamlit as st
import pandas as pd
import joblib
import numpy as np



# Access the stored Google Sheets URL from Streamlit secrets
spreadsheet_url = st.secrets["connections.gsheets"]["spreadsheet"]

# Read data from Google Sheets (ensure the link is a CSV exportable URL)
df = pd.read_csv(https://docs.google.com/spreadsheets/d/1iMtDVXtbdPXm0InGw2NYh9MKa3ixDFsUxzjFaieWGl4/edit?usp=sharing)

# Display the data in Streamlit
st.write("### Google Sheets Data")
st.dataframe(df)


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
    #"year_month_2024-08",  # One-hot encoded feature
    "total_visits",
     "month",
    "avg_days_between_pickups",
    #"days_since_last_pickup",
     #"year_month_2024-06"
   
    #"days_since_last_pickup"
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
    st.subheader("Hamper Collection Insights")
    st.title("Power BI Visualization")
    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiMTE4Y2JiYWQtMzNhYS00NGFiLThmMDQtMmIwMDg4YTIzMjI5IiwidCI6ImUyMjhjM2RmLTIzM2YtNDljMy05ZDc1LTFjZTI4NWI1OWM3OCJ9"
    st.components.v1.iframe(powerbi_url, width=800, height=600)

    
def predictions_page():
    # Streamlit app UI
    st.title("Hamper Return Prediction App")
    st.write("Enter details to predict if a client will return.")
    
    # User input fields (matching the top 5 important features)
    #year_month = st.selectbox("Year-Month", ["2024-08", "2024-07", "2024-06"])
    total_visits = st.number_input("Total Visits", min_value=1, max_value=100, step=1)
    avg_days_between_pickups = st.number_input("Avg Days Between Pickups", min_value=1.0, max_value=100.0, step=0.1)
    month = st.number_input("Month", min_value=1, max_value=12, step=1)
    #days_since_last_pickup = st.number_input("Days Since Last Pickup", min_value=0, step=1)
    #year_month = st.selectbox("Year-Month", ["2024-08", "2024-07", "2024-06"])
    
   # Prepare input data (One-hot encoding for the 'year_month' feature)
    input_data = {
    #"year_month_2024-08": 1 if year_month == "2024-08" else 0,
    #"year_month_2024-06": 1 if year_month == "2024-06" else 0,
    "total_visits": total_visits,
    "avg_days_between_pickups": avg_days_between_pickups,
    #"days_since_last_pickup": days_since_last_pickup,
    "month": month,
        
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
    app_page = st.sidebar.radio("Choose a page", ["Dashboard", "Insights", "Predictions"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Insights":
        exploratory_data_analysis()
    elif app_page == "Predictions":
        predictions_page()

# Run the app
if __name__ == "__main__":
    main()
