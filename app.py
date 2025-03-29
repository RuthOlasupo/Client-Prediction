import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.inspection import PartialDependenceDisplay

# Load the trained model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_top5.pkl")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Define only the top 5 features
REQUIRED_COLUMNS = [
    "total_visits",
    "month",
    "avg_days_between_pickups",
]

# Initialize SHAP explainer
@st.cache_resource
def load_explainer(model):
    try:
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        return explainer
    except Exception as e:
        st.error(f"❌ Error creating explainer: {e}")
        return None

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

def show_shap_analysis(input_df, prediction, probability):
    st.subheader("🔍 Prediction Explanation")
    
    # Get the preprocessor and classifier from the pipeline
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Process input data
    X_processed = preprocessor.transform(input_df)
    feature_names = [name.split('__')[-1] for name in preprocessor.get_feature_names_out()]
    
    # Convert to dense if sparse
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_processed)
    
    # For binary classification
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Prediction Breakdown", "Feature Effects"])
    
    with tab1:
        st.write("**Global Feature Importance**")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("This shows which features most influence model predictions overall.")
    
    with tab2:
        st.write("**How Each Feature Contributed to This Prediction**")
        
        # Create waterfall plot
        base_value = explainer.expected_value
        contribution = shap_values[0]
        pred_value = base_value + contribution.sum()
        
        # Create dataframe for plot
        df_waterfall = pd.DataFrame({
            'Feature': ['Base Value'] + feature_names + ['Prediction'],
            'Contribution': [base_value] + list(contribution) + [0],
            'Cumulative': [base_value] + list(np.cumsum([base_value] + list(contribution))[1:]) + [pred_value]
        })
        
        # Color based on positive/negative contribution
        df_waterfall['Color'] = df_waterfall['Contribution'].apply(
            lambda x: 'Positive' if x > 0 else 'Negative')
        
        fig = px.bar(df_waterfall, 
                    x='Contribution', 
                    y='Feature', 
                    color='Color',
                    color_discrete_map={'Positive': '#FF0051', 'Negative': '#008BFB'},
                    orientation='h',
                    hover_data={'Cumulative': ':.4f'},
                    title=f"Prediction: {'Return' if prediction[0] == 1 else 'No Return'} (Probability: {probability[0][1]:.2%})")
        
        # Add base value and prediction lines
        fig.add_vline(x=base_value, line_dash="dash", line_color="gray")
        fig.add_vline(x=0.5, line_dash="dot", line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show feature values
        st.write("**Feature Values for This Prediction:**")
        feature_values = {f: v for f, v in zip(feature_names, X_processed[0])}
        st.json(feature_values)
    
    with tab3:
        st.write("**How Changing Features Affects Prediction**")
        selected_feature = st.selectbox("Select feature to analyze", feature_names)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            model,
            input_df,
            features=[feature_names.index(selected_feature)],
            feature_names=feature_names,
            ax=ax
        )
        plt.title(f"Partial Dependence Plot for {selected_feature}")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("This shows how changing this feature would affect the prediction.")

def show_confidence_analysis(probability):
    st.subheader("📊 Prediction Confidence")
    
    # Create confidence gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability[0][1],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        gauge = {
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"},
                {'range': [0.7, 1], 'color': "darkgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability[0][1]}}
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    if probability[0][1] < 0.3 or probability[0][1] > 0.7:
        st.success("✅ High confidence prediction")
        st.write("The model is very confident in this prediction due to clear patterns in the input data.")
    else:
        st.warning("⚠️ Medium confidence prediction")
        st.write("The model has moderate confidence. Consider verifying with additional client information.")

def predictions_page():
    st.title("Hamper Return Prediction App")
    st.write("Enter details to predict if a client will return.")
    
    # User input fields
    col1, col2 = st.columns(2)
    with col1:
        total_visits = st.number_input("Total Visits", min_value=1, max_value=100, step=1, value=5)
        month = st.number_input("Month", min_value=1, max_value=12, step=1, value=6)
    with col2:
        avg_days_between_pickups = st.number_input("Avg Days Between Pickups", 
                                                min_value=1.0, max_value=100.0, 
                                                step=0.1, value=30.0)
    
    input_data = {
        "total_visits": total_visits,
        "avg_days_between_pickups": avg_days_between_pickups,
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
            
            # Show prediction result
            st.subheader("🎯 Prediction Result")
            
            if prediction[0] == 1:
                st.success(f"✅ The client is **likely to RETURN** (probability: {probability[0][1]:.2%})")
            else:
                st.error(f"❌ The client is **NOT likely to RETURN** (probability: {probability[0][0]:.2%})")
            
            # Show SHAP analysis
            show_shap_analysis(input_df, prediction, probability)
            
            # Show confidence analysis
            show_confidence_analysis(probability)
            
            # Recommendation based on prediction
            st.subheader("💡 Recommended Actions")
            if prediction[0] == 1:
                st.markdown("""
                - **Follow up:** Schedule a reminder for their next expected pickup
                - **Retention:** Consider adding them to loyalty communications
                - **Feedback:** Ask about their experience to maintain engagement
                """)
            else:
                st.markdown("""
                - **Outreach:** Consider a check-in call to understand their situation
                - **Incentives:** Offer additional support if appropriate
                - **Feedback:** Learn why they might not be returning
                """)

def dashboard():
    header_image_url = "https://raw.githubusercontent.com/ChiomaUU/Client-Prediction/refs/heads/main/ifssa_2844cc71-4dca-48ae-93c6-43295187e7ca.avif"
    st.image(header_image_url, use_container_width=True)

    st.title("Hamper Return Prediction App")
    st.write("This app predicts whether a client will return for food hampers using machine learning.")
    
    st.markdown("""
    ### Key Features:
    - **Predictions:** Estimate return probability for individual clients
    - **Explainability:** Understand why the model makes each prediction
    - **Confidence Metrics:** See how certain the model is about each prediction
    """)
    
    st.markdown("""
    ### How It Works:
    1. Navigate to the **Predictions** page
    2. Enter client details
    3. Get the prediction with explanations
    4. View recommended actions
    """)

def main():
    st.sidebar.title("Navigation")
    app_page = st.sidebar.radio("Choose a page", ["Dashboard", "Insights", "Predictions"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Insights":
        exploratory_data_analysis()
    elif app_page == "Predictions":
        predictions_page()

    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About This App:**
    - Uses CatBoost machine learning model
    - Provides explainable AI insights
    - Designed for food bank client retention
    """)

if __name__ == "__main__":
    main()
