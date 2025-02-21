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
