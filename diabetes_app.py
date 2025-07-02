import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Title
st.title("Diabetes Prediction System")
st.write("Enter patient details below to predict the risk of diabetes.")

# Load Dataset
try:
    data = pd.read_csv("diabetes.csv")  # Load from local file
except FileNotFoundError:
    st.error("The file 'diabetes.csv' was not found. Please make sure it is in the same directory.")
    st.stop()

# Preprocess Data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# User Input
st.sidebar.header("Patient Details")

def user_input():
    Pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
    Glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    BloodPressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    SkinThickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    Insulin = st.sidebar.slider("Insulin", 0.0, 900.0, 80.0)
    BMI = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    Age = st.sidebar.slider("Age", 10, 100, 30)

    user_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }

    return pd.DataFrame(user_data, index=[0])

input_df = user_input()

# Standardize user input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("Prediction Result")
result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
st.write(f"The model predicts the patient is **{result}**")
st.write("Prediction Probability:")
st.write(f"Not Diabetic: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Diabetic: {prediction_proba[0][1]*100:.2f}%")

# Visual: Confusion Matrix
st.subheader("Confusion Matrix")
st.dataframe(confusion_matrix(y_test, y_pred))

# Save Model (Optional)
# joblib.dump(model, 'diabetes_model.pkl')
