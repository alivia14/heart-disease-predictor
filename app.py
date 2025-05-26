import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🫀 Heart Disease Prediction App")
st.write("Enter the patient details to predict heart disease risk.")

# Input fields
age = st.number_input("Age", 20, 100, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200)
chol = st.number_input("Serum Cholesterol (chol)", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["True", "False"])
restecg = st.selectbox("Resting ECG (restecg)", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 220)
exang = st.selectbox("Exercise Induced Angina (exang)", ["Yes", "No"])
oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope of ST segment (slope)", ["Upsloping", "Flat", "Downsloping"])
ca = st.slider("Number of major vessels (ca)", 0, 4)
thal = st.selectbox("Thalassemia (thal)", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encode categorical variables
sex = 1 if sex == "Male" else 0
cp = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(cp)
fbs = 1 if fbs == "True" else 0
restecg = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
exang = 1 if exang == "Yes" else 0
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

# Collect into DataFrame
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
])

# Scale the data
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({prob*100:.1f}% probability)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({prob*100:.1f}% probability)")
