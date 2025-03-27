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

# --- Background Style ---
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

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

    # Load and show data if requested
    if st.checkbox("Show sample data"):
        data = load_data()
        if data is not None:
            st.dataframe(data.head())

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Dynamically get expected months from model features
        month_features = [col for col in model.feature_names_in_ if col.startswith("year_month_")]
        available_months = [col.replace("year_month_", "") for col in month_features]
        selected_month = st.selectbox("Select Month", available_months)
        
        total_visits = st.number_input("Total Visits", 
                                      min_value=1, 
                                      max_value=100, 
                                      value=5,
                                      help="Total number of client visits")
        
    with col2:
        avg_days = st.number_input("Average Days Between Pickups", 
                                  min_value=1.0, 
                                  max_value=100.0, 
                                  value=15.0,
                                  step=0.5,
                                  help="Average days between client pickups")
        
        days_since_last = st.number_input("Days Since Last Pickup", 
                                         min_value=0, 
                                         max_value=365,
                                         value=30,
                                         help="Days since client's last pickup")

    # --- Prediction Logic ---
    if st.button("Predict", type="primary"):
        if model is None:
            st.error("Model failed to load. Please check the model file.")
            return
            
        try:
            # Create one-hot encoded month features
            input_data = {
                "total_visits": total_visits,
                "avg_days_between_pickups": avg_days,
                "days_since_last_pickup": days_since_last
            }
            
            # Add all month features (0 for unselected)
            for month in available_months:
                input_data[f"year_month_{month}"] = 1 if month == selected_month else 0
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data]).reindex(columns=model.feature_names_in_, fill_value=0)
            
            # Debugging output
            with st.expander("Debug Info"):
                st.write("Model expects features:", model.feature_names_in_)
                st.write("Input data shape:", input_df.shape)
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
                
            # Show probability meter
            st.progress(proba[0][1])
            st.caption(f"Return probability: {proba[0][1]:.1%}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please check that all input values are valid.")

# --- Main App Navigation ---
def main():
    # Background image (optional)
    # set_background("https://example.com/background.jpg")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Power BI Dashboard"])
    
    if page == "Prediction":
        prediction_page()
    elif page == "Power BI Dashboard":
        powerbi_dashboard()

    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 Client Prediction App")

if __name__ == "__main__":
    main()
