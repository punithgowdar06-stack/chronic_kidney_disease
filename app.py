import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load model ---
MODEL_PATH = "ckd_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please upload 'ckd_model.pkl' to your repo.")
    st.stop()

model = joblib.load(MODEL_PATH)

scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

# --- App title ---
st.title("🩺 Chronic Kidney Disease Prediction")
st.write("Enter patient details to check risk of CKD.")

# --- Input fields ---
age = st.number_input("Age", 1, 120, 45)
bp = st.number_input("Blood Pressure", 50, 200, 80)
sg = st.number_input("Specific Gravity", 1.0, 1.05, 1.02, format="%.2f")
al = st.number_input("Albumin", 0, 5, 0)
su = st.number_input("Sugar", 0, 5, 0)
bgr = st.number_input("Blood Glucose Random", 50, 500, 100)
bu = st.number_input("Blood Urea", 1, 300, 40)
sc = st.number_input("Serum Creatinine", 0.1, 15.0, 1.2, format="%.2f")
hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.0, format="%.1f")
pcv = st.number_input("Packed Cell Volume", 10, 60, 40)
wc = st.number_input("White Blood Cell Count", 2000, 25000, 8000)
rc = st.number_input("Red Blood Cell Count", 2.0, 8.0, 5.0, format="%.1f")

# Convert inputs to dataframe
input_data = pd.DataFrame([[age, bp, sg, al, su, bgr, bu, sc, hemo, pcv, wc, rc]],
    columns=["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "hemo", "pcv", "wc", "rc"]
)

# Scale if scaler is available
if scaler:
    input_data = scaler.transform(input_data)

# --- Prediction ---
if st.button("🔍 Predict CKD"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Patient is at risk of Chronic Kidney Disease")
    else:
        st.success("✅ Patient is NOT at risk of Chronic Kidney Disease")
