# claims_triage_deployment.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Claims Triage Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model and artifacts"""
    try:
        artifacts = joblib.load('claims_triage_model_artifacts.joblib')
        return artifacts
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
artifacts = load_model()
if artifacts:
    st.session_state.model_loaded = True
    st.session_state.artifacts = artifacts
    pipeline = artifacts['pipeline']
    feature_names = artifacts.get('feature_names', [])
    target_names = artifacts.get('target_names', ['Cluster_0', 'Cluster_1', 'Cluster_2'])
    
    # Get the actual feature names used by the preprocessor
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        # Get feature names after preprocessing
        encoded_feature_names = preprocessor.get_feature_names_out()
    except:
        encoded_feature_names = []

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=80)
    st.markdown("### 🏥 Claims Triage System")
    st.markdown("---")
    
    if st.session_state.model_loaded:
        st.markdown("#### Model Information")
        st.markdown(f"**Status:** ✅ Loaded")
        st.markdown(f"**Features:** {len(feature_names)}")
    
    st.markdown("---")
    st.markdown("#### Navigation")
    app_mode = st.radio(
        "Select Mode",
        ["🎯 Single Claim Assessment", "📊 Batch Processing", "📈 Model Insights"]
    )

# Main content
st.markdown('<h1 class="main-header">🏥 Claims Triage Classification System</h1>', unsafe_allow_html=True)
st.markdown("**First Mutual Health Zimbabwe • Automated Claims Processing**")

# Check if model is loaded
if not st.session_state.model_loaded:
    st.error("⚠️ Model loading failed. Please check if model files exist.")
    st.stop()

# Helper function to create input dataframe with correct format
def create_input_dataframe(user_inputs, feature_names):
    """Create properly formatted input dataframe for prediction"""
    
    # Start with user inputs
    input_dict = {}
    
    # Add Amount (ensure it's numeric)
    if 'Amount' in user_inputs:
        input_dict['Amount'] = [float(user_inputs['Amount'])]
    else:
        input_dict['Amount'] = [0.0]
    
    # Add categorical features
    categorical_features = ['Pay To', 'Product', 'Self Payer', 'Discipline', 
                           'Reporting Delays Cat', 'Assessment Delays Cat', 'Processing Delays Cat']
    
    for feature in categorical_features:
        if feature in user_inputs:
            input_dict[feature] = [user_inputs[feature]]
        else:
            # Use default values if not provided
            if feature == 'Pay To':
                input_dict[feature] = ['PROVIDER']
            elif feature == 'Product':
                input_dict[feature] = ['SAPPHIRE PLUS GOLD']
            elif feature == 'Self Payer':
                input_dict[feature] = ['N']
            elif feature == 'Discipline':
                input_dict[feature] = ['SPECIALIST']
            elif 'Delays' in feature:
                input_dict[feature] = ['NO DELAY']
    
    # Create dataframe
    input_df = pd.DataFrame(input_dict)
    
    # Ensure all expected columns are present (from original training)
    # This is crucial - the model expects the same columns as during training
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        # For numerical columns, use 0
        if col in ['Amount', 'REPORTING DELAYS', 'ASSESSMENT DELAYS', 'PROCESSING DELAYS']:
            input_df[col] = 0
        # For categorical columns, use the most common value
        else:
            input_df[col] = 'Unknown'
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    return input_df

# Single Claim Assessment Mode
if app_mode == "🎯 Single Claim Assessment":
    st.markdown('<h2 class="sub-header">Single Claim Assessment</h2>', unsafe_allow_html=True)
    
    # Create input form
    with st.form("single_claim_form"):
        st.markdown("### Enter Claim Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input(
                "Amount ($)",
                min_value=0.0,
                max_value=10000.0,
                value=150.0,
                step=10.0,
                help="Total amount being claimed"
            )
            
            pay_to = st.selectbox(
                "Pay To",
                options=["PROVIDER", "MEMBER"],
                index=0
            )
            
            product = st.selectbox(
                "Product",
                options=["SAPPHIRE PLUS GOLD", "AMBER GOLD", "SAPPHIRE SILVER", 
                        "RUBY GOLD", "EMERALD GOLD"],
                index=0
            )
            
            self_payer = st.selectbox(
                "Self Payer",
                options=["N", "Y", "P"],
                index=0
            )
        
        with col2:
            discipline = st.selectbox(
                "Discipline",
                options=["SPECIALIST", "PRESCRIBED DRUGS", "AMBULANCE", 
                        "OPTICAL", "PRIVATE HOSPITALS", "FOREIGN TREATMENT",
                        "GENERAL PRACTITIONER"],
                index=0
            )
            
            reporting_delay = st.selectbox(
                "Reporting Delays Cat",
                options=["NO DELAY", "1-7 DAYS", "8-30 DAYS", "OVER 30 DAYS"],
                index=0
            )
            
            assessment_delay = st.selectbox(
                "Assessment Delays Cat",
                options=["NO DELAY", "1-7 DAYS", "8-30 DAYS", "OVER 30 DAYS"],
                index=0
            )
            
            processing_delay = st.selectbox(
                "Processing Delays Cat",
                options=["NO DELAY", "1-7 DAYS", "8-30 DAYS", "OVER 30 DAYS"],
                index=0
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Classify Claim", type="primary")
    
    if submitted:
        with st.spinner("Analyzing claim..."):
            try:
                # Create user inputs dictionary
                user_inputs = {
                    'Amount': amount,
                    'Pay To': pay_to,
                    'Product': product,
                    'Self Payer': self_payer,
                    'Discipline': discipline,
                    'Reporting Delays Cat': reporting_delay,
                    'Assessment Delays Cat': assessment_delay,
                    'Processing Delays Cat': processing_delay
                }
                
                # Create properly formatted input dataframe
                st.info("Creating input data format...")
                input_df = create_input_dataframe(user_inputs, feature_names)
                
                # Display the input data for debugging
                with st.expander("📋 View Input Data Structure"):
                    st.write("Input DataFrame shape:", input_df.shape)
                    st.write("Columns:", list(input_df.columns))
                    st.dataframe(input_df.head())
                
                st.info("Making prediction...")
                # Make prediction
                prediction = pipeline.predict(input_df)[0]
                probabilities = pipeline.predict_proba(input_df)[0]
                confidence = np.max(probabilities)
                
                # Display results
                st.success("✅ Prediction completed!")
                
                # Cluster information
                cluster_info = {
                    0: {'name': 'High-Complexity', 'color': '#DC2626', 'icon': '🔴'},
                    1: {'name': 'Routine', 'color': '#2563EB', 'icon': '🔵'},
                    2: {'name': 'High-Value', 'color': '#D97706', 'icon': '🟠'}
                }
                
                cluster = cluster_info[prediction]
                
                # Results display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='padding: 1.5rem; border-radius: 10px; background-color: {cluster['color']}20; 
                                border-left: 5px solid {cluster['color']};'>
                        <h3 style='color: {cluster['color']}; margin: 0;'>
                            {cluster['icon']} {cluster['name']}
                        </h3>
                        <p style='font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;'>
                            Cluster {prediction}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Confidence Score")
                    st.markdown(f"# {confidence:.1%}")
                    
                    # Progress bar for confidence
                    st.progress(float(confidence))
                
                with col3:
                    st.markdown("### Recommended Action")
                    if prediction == 0:
                        st.warning("⚠️ Requires expert review")
                        st.markdown("• Manual assessment needed")
                        st.markdown("• Senior adjudicator review")
                    elif prediction == 1:
                        st.success("✅ Suitable for automation")
                        st.markdown("• Standard processing")
                        st.markdown("• Automated approval")
                    else:
                        st.error("🚨 High-value attention")
                        st.markdown("• Senior review required")
                        st.markdown("• Priority processing")
                
                # Probability breakdown
                st.markdown("---")
                st.markdown("### 📊 Probability Breakdown")
                
                prob_data = []
                for i, prob in enumerate(probabilities):
                    prob_data.append({
                        'Cluster': f'Cluster {i}',
                        'Name': cluster_info[i]['name'],
                        'Probability': prob,
                        'Color': cluster_info[i]['color']
                    })
                
                prob_df = pd.DataFrame(prob_data)
                
                fig = px.bar(
                    prob_df,
                    x='Name',
                    y='Probability',
                    color='Name',
                    color_discrete_map={
                        'High-Complexity': '#DC2626',
                        'Routine': '#2563EB',
                        'High-Value': '#D97706'
                    },
                    title='Prediction Probabilities'
                )
                fig.update_layout(
                    yaxis_tickformat=',.0%',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for debugging
                with st.expander("🔍 Technical Details"):
                    st.write("Prediction probabilities:", probabilities)
                    st.write("Model used:", type(pipeline.named_steps['classifier']).__name__)
                    st.write("Input features used:", len(feature_names))
                    
            except Exception as e:
                st.error(f"Error processing claim: {str(e)}")
                st.error("This might be due to missing features or incorrect data format.")
                
                # Debug information
                with st.expander("🔧 Debug Information"):
                    st.write("Feature names expected:", feature_names)
                    st.write("Model pipeline steps:", list(pipeline.named_steps.keys()))
                    
                    if 'preprocessor' in pipeline.named_steps:
                        st.write("Preprocessor transformers:", 
                                pipeline.named_steps['preprocessor'].transformers)

# Batch Processing Mode
elif app_mode == "📊 Batch Processing":
    st.markdown('<h2 class="sub-header">Batch Claims Processing</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with claims data",
        type=['csv'],
        help="The CSV should contain columns matching the training data features"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display preview
            st.markdown(f"### Data Preview ({len(df)} claims)")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check for required columns
            missing_cols = set(feature_names) - set(df.columns)
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {list(missing_cols)}")
                st.info("Adding missing columns with default values...")
                
                # Add missing columns with default values
                for col in missing_cols:
                    if col in ['Amount', 'REPORTING DELAYS', 'ASSESSMENT DELAYS', 'PROCESSING DELAYS']:
                        df[col] = 0
                    else:
                        df[col] = 'Unknown'
            
            # Process button
            if st.button("🚀 Process All Claims", type="primary"):
                with st.spinner(f"Processing {len(df)} claims..."):
                    try:
                        # Ensure correct column order
                        df = df[feature_names]
                        
                        # Make predictions
                        predictions = pipeline.predict(df)
                        probabilities = pipeline.predict_proba(df)
                        confidence_scores = np.max(probabilities, axis=1)
                        
                        # Add results to dataframe
                        result_df = df.copy()
                        result_df['Predicted_Cluster'] = predictions
                        result_df['Prediction_Confidence'] = confidence_scores
                        
                        # Cluster names
                        cluster_names = {
                            0: 'High-Complexity',
                            1: 'Routine',
                            2: 'High-Value'
                        }
                        result_df['Triage_Category'] = result_df['Predicted_Cluster'].map(cluster_names)
                        
                        # Store results
                        st.session_state.predictions = result_df
                        
                        # Display summary
                        st.success(f"✅ Successfully processed {len(result_df)} claims!")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            cluster_0 = (result_df['Predicted_Cluster'] == 0).sum()
                            st.metric("High-Complexity", cluster_0)
                        
                        with col2:
                            cluster_1 = (result_df['Predicted_Cluster'] == 1).sum()
                            st.metric("Routine", cluster_1)
                        
                        with col3:
                            cluster_2 = (result_df['Predicted_Cluster'] == 2).sum()
                            st.metric("High-Value", cluster_2)
                        
                        with col4:
                            avg_confidence = result_df['Prediction_Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                        
                        # Visualization
                        st.markdown("### 📈 Distribution")
                        
                        fig = px.pie(
                            result_df,
                            names='Triage_Category',
                            title='Claim Distribution',
                            color='Triage_Category',
                            color_discrete_map={
                                'High-Complexity': '#DC2626',
                                'Routine': '#2563EB',
                                'High-Value': '#D97706'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        st.markdown("### 📥 Download Results")
                        
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Full Results (CSV)",
                            data=csv,
                            file_name="claims_triage_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing batch: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Model Insights Mode
elif app_mode == "📈 Model Insights":
    st.markdown('<h2 class="sub-header">Model Insights</h2>', unsafe_allow_html=True)
    
    # Model information
    st.markdown("### 🏗️ Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Pipeline Steps")
        steps = list(pipeline.named_steps.keys())
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. **{step}**")
            if step == 'classifier':
                st.markdown(f"   - Type: {type(pipeline.named_steps[step]).__name__}")
    
    with col2:
        st.markdown("#### Feature Information")
        st.markdown(f"**Total Features:** {len(feature_names)}")
        
        # Count categorical vs numerical
        categorical_count = len([f for f in feature_names if 'Cat' in f or f in ['Pay To', 'Product', 'Self Payer', 'Discipline']])
        numerical_count = len(feature_names) - categorical_count
        
        st.markdown(f"**Numerical Features:** {numerical_count}")
        st.markdown(f"**Categorical Features:** {categorical_count}")
    
    # Sample predictions table
    st.markdown("---")
    st.markdown("### 🎯 Sample Predictions")
    
    sample_data = [
        {"Amount": 1500, "Discipline": "AMBULANCE", "Prediction": "High-Value", "Confidence": "95%"},
        {"Amount": 45, "Discipline": "PRESCRIBED DRUGS", "Prediction": "Routine", "Confidence": "88%"},
        {"Amount": 350, "Discipline": "FOREIGN TREATMENT", "Prediction": "High-Complexity", "Confidence": "92%"},
        {"Amount": 120, "Discipline": "SPECIALIST", "Prediction": "Routine", "Confidence": "85%"},
        {"Amount": 2200, "Discipline": "PRIVATE HOSPITALS", "Prediction": "High-Value", "Confidence": "96%"},
    ]
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    
    # Performance metrics (if available in artifacts)
    if 'performance_metrics' in artifacts:
        st.markdown("---")
        st.markdown("### 📊 Performance Metrics")
        
        metrics = artifacts['performance_metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'accuracy' in metrics:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        
        with col2:
            if 'precision' in metrics:
                st.metric("Precision", f"{metrics['precision']:.3f}")
        
        with col3:
            if 'recall' in metrics:
                st.metric("Recall", f"{metrics['recall']:.3f}")
        
        with col4:
            if 'f1_score' in metrics:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <p style="font-size: 0.9rem;">
        <strong>Claims Triage Classification System</strong> • First Mutual Health Zimbabwe
    </p>
    <p style="font-size: 0.8rem;">
        MSc in Big Data Analytics Research • Chinhoyi University of Technology • 2026
    </p>
</div>
""", unsafe_allow_html=True)