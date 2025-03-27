import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os

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

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv("./df.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Hardcoded Features (UPDATE THESE TO MATCH YOUR MODEL) ---
REQUIRED_FEATURES = [
    "total_visits",
    "avg_days_between_pickups",
    "days_since_last_pickup"
]

# --- Prediction Page ---
def prediction_page():
    # Header Image
    header_image_url = "https://raw.githubusercontent.com/ChiomaUU/Client-Prediction/refs/heads/main/ifssa_2844cc71-4dca-48ae-93c6-43295187e7ca.avif"
    st.image(header_image_url, use_container_width=True)

    st.title("Client Return Prediction App")
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    </style>
    <p class="big-font">Enter details to predict if a client will return.</p>
    """, unsafe_allow_html=True)

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    
    with col1:
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
            # Create input data with only the required features
            input_data = {
                "total_visits": total_visits,
                "avg_days_between_pickups": avg_days,
                "days_since_last_pickup": days_since_last
            }
            
            # Create DataFrame with exactly the required columns
            input_df = pd.DataFrame([input_data])[REQUIRED_FEATURES]
            
            # Debugging output
            with st.expander("Debug Info"):
                st.write("Input data shape:", input_df.shape)
                st.write("Input features:", input_df.columns.tolist())
                st.dataframe(input_df)
            
            # Make prediction
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction[0] == 1:
                st.success(f"✅ Client is LIKELY to return (probability: {proba[0][1]:.1%})")
            else:
                st.error(f"❌ Client is UNLIKELY to return (probability: {proba[0][0]:.1%})")
                
            st.progress(proba[0][1])
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please check that your model expects exactly these features:")
            st.write(REQUIRED_FEATURES)

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

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 Client Prediction App")

if __name__ == "__main__":
    main()
