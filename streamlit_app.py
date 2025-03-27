import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib

# --- Configuration ---
st.set_page_config(layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("./model_top5.pkl")
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Define the Top 5 Features ---
TOP_FEATURES = [
    "year_month_2024_08",
    "total_visits",
    "avg_days_between_pickups",
    "days_since_last_pickup",
    "year_month_2024_06"
]

# --- Prediction Page ---
def prediction_page():
    st.title("Client Return Prediction App")
    st.write("Enter details to predict if a client will return.")

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Month selection (will be one-hot encoded)
        month = st.selectbox("Select Month", ["2024_08", "2024_07", "2024_06"])
        
        total_visits = st.number_input("Total Visits", 
                                     min_value=1, 
                                     max_value=100, 
                                     value=5)
        
    with col2:
        avg_days = st.number_input("Average Days Between Pickups", 
                                 min_value=1.0, 
                                 max_value=100.0, 
                                 value=15.0,
                                 step=0.5)
        
        days_since_last = st.number_input("Days Since Last Pickup", 
                                        min_value=0, 
                                        max_value=365,
                                        value=30)

    # --- Prediction Logic ---
    if st.button("Predict", type="primary"):
        if model is None:
            st.error("Model failed to load. Please check the model file.")
            return
            
        try:
            # Create one-hot encoded month features
            input_data = {
                "year_month_2024_08": 1 if month == "2024_08" else 0,
                "year_month_2024_06": 1 if month == "2024_06" else 0,
                "total_visits": total_visits,
                "avg_days_between_pickups": avg_days,
                "days_since_last_pickup": days_since_last
            }
            
            # Create DataFrame with exact feature order
            input_df = pd.DataFrame([input_data])[TOP_FEATURES]
            
            # Debugging output
            with st.expander("Debug Info"):
                st.write("Input features:", input_df.columns.tolist())
                st.write("Input values:", input_df.values)
            
            # Make prediction
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)
            
            # Display results
            st.subheader("Prediction Results")
            result = "✅ LIKELY to return" if prediction[0] == 1 else "❌ UNLIKELY to return"
            st.write(f"**Prediction:** {result}")
            st.write(f"**Probability of returning:** {proba[0][1]:.1%}")
            st.progress(proba[0][1])
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please check that your input values are valid.")

# --- Power BI Dashboard ---
def powerbi_dashboard():
    st.title("Power BI Dashboard")
    powerbi_link = "https://app.powerbi.com/view?r=eyJrIjoiMTE4Y2JiYWQtMzNhYS00NGFiLThmMDQtMmIwMDg4YTIzMjI5IiwidCI6ImUyMjhjM2RmLTIzM2YtNDljMy05ZDc1LTFjZTI4NWI1OWM3OCJ9"
    
    components.html(
        f"""
        <iframe
            width="100%"
            height="800"
            src="{powerbi_link}"
            frameborder="0"
            allowFullScreen="true">
        </iframe>
        """,
        height=800,
    )

# --- Main App Navigation ---
def main():    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Power BI Dashboard"])
    
    if page == "Prediction":
        prediction_page()
    elif page == "Power BI Dashboard":
        powerbi_dashboard()

if __name__ == "__main__":
    main()
