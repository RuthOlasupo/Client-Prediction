import streamlit as st
import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    return pd.DataFrame({
        "age": [25, 40, 35],
        "household": [1, 0, 1],
        "time_since_previous_visit": [2, 5, 1],
        "contact_frequency": ["weekly", "monthly", "weekly"],
        "returned": [1, 0, 1]  # Target column
    })

# Prediction function (rule-based for now)
def predict(input_data):
    """Predict client retention likelihood based on input data."""
    if input_data['age'].values[0] > 30 and input_data['household'].values[0] == 1:
        return [1]  # Likely to return
    else:
        return [0]  # Unlikely to return

# Predictions Page
def predictions_page():
    st.title("Client Retention Prediction for Food Bank")
    
    # Load and display dataset
    data = load_data()
    
    if st.checkbox("Show raw data"):
        st.write(data)
    
    # User Input Section
    st.header("Enter Client Information")
    
    # Create input fields
    input_features = {
        'age': st.number_input("Enter Age", min_value=0, max_value=100, value=30),
        'household': st.selectbox("Is the client part of a household?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
        'time_since_previous_visit': st.number_input("Enter Time Since Previous Visit (in months)", min_value=0, value=1),
        'contact_frequency': st.selectbox("Contact Frequency", options=["weekly", "monthly"])
    }
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_features])
    
    # Prediction Button
    if st.button("Predict"):
        prediction = predict(input_df)
    
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("✅ The client is likely to return.")
        else:
            st.error("❌ The client is unlikely to return.")

# Infographic Page
def exploratory_data_analysis():
    st.subheader("Infograph of Clients")
  
    # Display images
    st.image("IFSSA CLEANED DATA_page-0001.jpg", use_container_width=True)

    # Display second image
    st.image("IFSSA CLEANED DATA_page-0002.jpg", use_container_width=True)
    st.markdown("<h4 style='text-align: center;'>Clients Infograph</h4>", unsafe_allow_html=True)

# Dashboard Page
def dashboard():
    header_image_url = "https://raw.githubusercontent.com/ChiomaUU/Client-Prediction/refs/heads/main/ifssa_2844cc71-4dca-48ae-93c6-43295187e7ca.avif"
    st.image(header_image_url, use_container_width=True)  # Display the image at the top

    st.title("Client Return Prediction App (MVP)")
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
if _name_ == "_main_":
    main()
