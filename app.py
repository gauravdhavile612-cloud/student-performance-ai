import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Student Predictor", page_icon="🎓", layout="wide")

st.title("🎓 AI-Based Student Performance Predictor")

# Load dataset
data = pd.read_csv("student_data.csv")

X = data.drop("result", axis=1)
y = data["result"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.title("📊 Input Student Data")
attendance = st.sidebar.slider("Attendance (%)", 0, 100)
study_hours = st.sidebar.slider("Study Hours per Day", 0, 10)
internal_marks = st.sidebar.slider("Internal Marks", 0, 100)
assignments = st.sidebar.selectbox("Assignments Submitted", [0, 1])
previous_marks = st.sidebar.slider("Previous Semester Marks", 0, 100)

if st.sidebar.button("🚀 Predict Now"):
    input_data = np.array([[attendance, study_hours, internal_marks, assignments, previous_marks]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    confidence = max(probability[0]) * 100

    st.subheader("🔍 Prediction Result")

    if prediction[0] == "Pass":
        st.success("🎉 Student is likely to PASS")
    else:
        st.error("⚠ Student is likely to FAIL")

    st.progress(int(confidence))
    st.info(f"Model Confidence: {confidence:.2f}%")

    st.subheader("📈 Feature Importance")

    feature_names = ["Attendance", "Study Hours", "Internal Marks", "Assignments", "Previous Marks"]
    importances = model.feature_importances_

    for name, importance in zip(feature_names, importances):
        st.write(name)
        st.progress(int(importance * 100))
