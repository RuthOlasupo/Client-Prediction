import streamlit as st
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

# --- Define All Features Expected by Model ---
ALL_FEATURES = [
    "year_month_2024_08",  # One-hot encoded
    "total_visits",        # Scaled
    "avg_days_between_pickups",  # Scaled
    "days_since_last_pickup",    # Scaled
    "year_month_2024_06"   # One-hot encoded
]

# --- Prediction Page ---
def prediction_page():
    st.title("📊 Client Return Prediction")
    st.markdown("Predict if a client will return based on their activity")
    
    # --- User Input Section ---
    with st.form("prediction_form"):
        st.subheader("Client Details")
        
        # Month Selection
        selected_month = st.selectbox(
            "Month of Activity",
            ["2024_08", "2024_07", "2024_06"],
            help="Select the month you're evaluating"
        )
        
        # Numerical Inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            total_visits = st.number_input(
                "Total Visits",
                min_value=0,
                max_value=100,
                value=5,
                step=1
            )
        with col2:
            avg_days = st.number_input(
                "Avg Days Between Visits",
                min_value=0.0,
                max_value=365.0,
                value=15.0,
                step=1.0
            )
        with col3:
            days_since_last = st.number_input(
                "Days Since Last Visit",
                min_value=0,
                max_value=365,
                value=30,
                step=1
            )
        
        # Submit Button
        submitted = st.form_submit_button("Predict Return Probability")
    
    # --- Prediction Logic ---
    if submitted and model:
        try:
            # Create one-hot encoded month features
            month_data = {
                "year_month_2024_08": 1 if selected_month == "2024_08" else 0,
                "year_month_2024_06": 1 if selected_month == "2024_06" else 0
            }
            
            # Combine all features
            input_data = {
                **month_data,
                "total_visits": total_visits,
                "avg_days_between_pickups": avg_days,
                "days_since_last_pickup": days_since_last
            }
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])[ALL_FEATURES]
            
            # Debug View
            with st.expander("🔍 See what's being sent to the model"):
                st.dataframe(input_df)
                st.write(f"Feature count: {len(input_df.columns)}")
                st.write(f"Feature order: {input_df.columns.tolist()}")
            
            # Make prediction
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)
            
            # Display Results
            st.subheader("🎯 Prediction Results")
            
            result_col, prob_col = st.columns(2)
            with result_col:
                if prediction[0] == 1:
                    st.success("✅ Client WILL return")
                else:
                    st.error("❌ Client WON'T return")
            
            with prob_col:
                st.metric(
                    "Return Probability", 
                    f"{proba[0][1]:.1%}",
                    delta=f"{(proba[0][1]-0.5):+.1%}"  # Show difference from 50%
                )
                st.progress(proba[0][1])
            
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")
            st.write("Please verify your model expects these features:")
            st.write(ALL_FEATURES)

# --- Run the App ---
if __name__ == "__main__":
    prediction_page()
