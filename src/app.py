import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "laptop_pipeline.joblib")

st.set_page_config(page_title="Laptop AI Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { color: #00d4ff; font-size: 40px; }
    .stButton>button { 
        width: 100%; border-radius: 20px; 
        background-color: #00d4ff; color: black; font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #008fb3;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource 
def load_pipeline():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Did you run train.py and push the 'models' folder to GitHub?")
    st.stop()

st.title("Laptop Price AI Predictor")
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
        input_df = pd.DataFrame(
            [[ram, ssd, weight, screen]], 
            columns=['ram', 'ssd', 'weight', 'screen']
        )
        
        prediction = pipeline.predict(input_df)[0]

        if prediction < 150:
            prediction = 150.0
        
        # Display Metric
        st.metric(label="Predicted Price (USD)", value=f"${prediction:,.2f}")
        
        st.write("### Value Comparison Chart")
        chart_data = pd.DataFrame(
            np.random.randn(10, 1) * 50 + prediction,
            columns=['Market Price Range']
        )
        st.area_chart(chart_data)
    else:
        st.info("Adjust the sliders on the left and click 'Calculate' to see the AI prediction.")
