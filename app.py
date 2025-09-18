import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("ckd_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="CKD Prediction App", layout="centered")
st.title("🩺 Chronic Kidney Disease Prediction")
st.markdown("Enter patient details to check CKD risk.")

# --- Input fields ---
age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=80)
sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.025, step=0.001, value=1.015)
al = st.number_input("Albumin", min_value=0, max_value=5, value=0)
su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
ba = st.selectbox("Bacteria", ["notpresent", "present"])
bgr = st.number_input("Blood Glucose Random (mg/dl)", min_value=50, max_value=500, value=120)
bu = st.number_input("Blood Urea (mg/dl)", min_value=1, max_value=400, value=40)
sc = st.number_input("Serum Creatinine (mg/dl)", min_value=0.1, max_value=20.0, value=1.2)
sod = st.number_input("Sodium (mEq/L)", min_value=100, max_value=200, value=140)
pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=10.0, value=4.5)
hemo = st.number_input("Hemoglobin (g/dl)", min_value=3.0, max_value=20.0, value=15.0)
pcv = st.number_input("Packed Cell Volume", min_value=20, max_value=55, value=40)
wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=2000, max_value=25000, value=8000)
rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.0, max_value=7.0, value=5.0)
htn = st.selectbox("Hypertension", ["no", "yes"])
dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
appet = st.selectbox("Appetite", ["good", "poor"])
pe = st.selectbox("Pedal Edema", ["no", "yes"])
ane = st.selectbox("Anemia", ["no", "yes"])

# --- Encode categorical features ---
mapping = {
    "normal": 0, "abnormal": 1,
    "notpresent": 0, "present": 1,
    "no": 0, "yes": 1,
    "good": 0, "poor": 1
}

rbc = mapping[rbc]
pc = mapping[pc]
pcc = mapping[pcc]
ba = mapping[ba]
htn = mapping[htn]
dm = mapping[dm]
cad = mapping[cad]
appet = mapping[appet]
pe = mapping[pe]
ane = mapping[ane]

# --- Prepare input ---
input_data = np.array([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu,
                        sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad,
                        appet, pe, ane]])

input_scaled = scaler.transform(input_data)

# --- Prediction ---
if st.button("Predict CKD"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ Patient is likely to have Chronic Kidney Disease.")
    else:
        st.success("✅ Patient is unlikely to have CKD.")
