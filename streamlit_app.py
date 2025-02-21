import streamlit as st
import pandas as pd

# Simulate loading a dataset (for demonstration purposes)
@st.cache_data
def load_data():
    # Create a dummy dataset
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'previous_visits': [1, 2, 3, 4, 5],
        'returned': [1, 0, 1, 0, 1]  # Target column (1 = returned, 0 = did not return)
    })
    return data

# Simulate a prediction (rule-based or random)
def predict(input_data):
    # Example rule: If age > 30 and income > 60000, predict "likely to return"
    if input_data['age'].values[0] > 30 and input_data['income'].values[0] > 60000:
        return [1]  # Likely to return
    else:
        return [0]  # Unlikely to return

# Main function to run the app
def main():
    st.title("Client Return Prediction App (MVP)")
    st.write("This app predicts whether a client will return for food hampers.")

    # Load the dataset
    data = load_data()

    # Display the dataset (optional)
    if st.checkbox("Show raw data"):
        st.write(data)

    # Input fields for user to enter data
    st.sidebar.header("Input Features")
    input_features = {}

    # Create input fields for each feature
    for column in data.columns[:-1]:  # Exclude the target column
        input_features[column] = st.sidebar.number_input(f"Enter {column}", value=0.0)

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_features])

    # Make prediction
    if st.sidebar.button("Predict"):
        prediction = predict(input_df)
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("The client is likely to return.")
        else:
            st.error("The client is unlikely to return.")

# Run the app
if __name__ == "__main__":
    main()

import streamlit as st

# Set the background image
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;  /* Ensures the image covers the entire background */
            background-position: center;  /* Centers the image */
            background-repeat: no-repeat;  /* Prevents the image from repeating */
            background-attachment: fixed;  /* Makes the background image fixed while scrolling */
        }}
        /* Add a semi-transparent overlay to improve readability of text */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.7);  /* White overlay with 70% opacity */
            z-index: -1;
        }}
        /* Ensure Streamlit components are visible */
        .stButton>button, .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {{
            background-color: rgba(255, 255, 255, 0.8) !important;  /* Slightly transparent white background for inputs */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Main function to run the app
def main():
    # Set the background image (replace with your raw GitHub URL)
    background_url = "https://raw.githubusercontent.com/ChiomaUU/Client-Prediction/refs/heads/main/ifssa_2844cc71-4dca-48ae-93c6-43295187e7ca.avif"
    set_background(background_url)

    # Add content to the app
    st.title("Client Return Prediction App")
    st.write("This app predicts whether a client will return for food hampers.")
    st.write("Add your app content here.")

    # Example input fields
    age = st.number_input("Enter Age", min_value=0, max_value=100, value=30)
    income = st.number_input("Enter Income", min_value=0, value=50000)
    st.button("Predict")

# Run the app
if __name__ == "__main__":
    main()
