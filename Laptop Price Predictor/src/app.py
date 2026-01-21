import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Advanced Page Config
st.set_page_config(page_title="Laptop AI Predictor", layout="wide")

# 2. Inject Custom CSS for "Advanced" Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { color: #00d4ff; font-size: 40px; }
    .stButton>button { 
        width: 100%; border-radius: 20px; 
        background-color: #00d4ff; color: black; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model and Scaler
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')

# 4. UI Layout
st.title("💻 Laptop Price AI Predictor")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configure Specs")
    ram = st.slider("RAM (GB)", 4, 64, 8)
    ssd = st.select_slider("Storage (SSD GB)", options=[128, 256, 512, 1024, 2048])
    weight = st.number_input("Weight (kg)", 0.5, 4.0, 1.4)
    screen = st.number_input("Screen Size (Inches)", 10.0, 20.0, 13.3)
    
    predict_btn = st.button("Calculate Market Value")

with col2:
    st.subheader("Market Analysis")
    if predict_btn:
        # Process Input
        input_data = np.array([[ram, ssd, weight, screen]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        # Display Metric
        st.metric(label="Predicted Price (USD)", value=f"${prediction:,.2f}")
        
        # Fun Visualization
        chart_data = pd.DataFrame(
            np.random.randn(10, 1) * 100 + prediction,
            columns=['Market Fluctuations']
        )
        st.line_chart(chart_data)
    else:
        st.info("Adjust the sliders on the left to see the AI prediction.")