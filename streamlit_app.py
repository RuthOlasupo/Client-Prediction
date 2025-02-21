import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
@st.cache_data
def load_model():
    model = joblib.load('model.pkl')  # Replace with your model file path
    return model

# Load the dataset (if needed for reference)
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv')  # Replace with your dataset file path
    return data

# Function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Main function to run the app
def main():
    st.title("Client Return Prediction App")
    st.write("This app predicts whether a client will return for food hampers.")

    # Load the model and data
    model = load_model()
    data = load_data()

    # Display the dataset (optional)
    if st.checkbox("Show raw data"):
        st.write(data)

    # Input fields for user to enter data
    st.sidebar.header("Input Features")
    input_features = {}

    for column in data.columns[:-1]:  # Exclude the target column
        input_features[column] = st.sidebar.number_input(f"Enter {column}", value=0.0)

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_features])

    # Make prediction
    if st.sidebar.button("Predict"):
        prediction = predict(model, input_df)
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("The client is likely to return.")
        else:
            st.error("The client is unlikely to return.")

# Run the app
if __name__ == "__main__":
    main()
