import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('random_forest_model_NF.pkl')

def predict_g3(sex, age, address, Medu, Fedu, Mjob, Fjob, reason, guardian,
               traveltime, studytime, failures, schoolsup, higher, internet, romantic, 
               goout, Dalc, Walc, health, absences, G1, G2):
    """Predict student's final grade"""
    # Create features array
    features = np.array([[sex, age, address, Medu, Fedu, Mjob, Fjob, reason, guardian,
                         traveltime, studytime, failures, schoolsup, higher, internet, romantic,
                         goout, Dalc, Walc, health, absences, G1, G2]])  # Total 23 features
    
    # Make prediction
    prediction = model.predict(features)
    return float(prediction[0])

# Streamlit app

# Title
st.title("Student Grade Predictor")

# Description
st.write("Enter student characteristics to predict their final grade.")

# Input fields
sex = st.selectbox("Sex", options=["Male", "Female"])  # Categorical options
sex = 0 if sex == "Male" else 1  # Encode for prediction

age = st.slider("Age", min_value=15, max_value=22, value=15, step=1)
address = st.selectbox("Address", options=["Urban", "Rural"])  # Categorical options
address = 0 if address == "Urban" else 1  # Encode for prediction

Medu = st.slider("Mother's Education (Medu)", min_value=0, max_value=4, value=0, step=1)
Fedu = st.slider("Father's Education (Fedu)", min_value=0, max_value=4, value=0, step=1)

# Categorical features
Mjob = st.selectbox("Mother's Job", options=["teacher", "health", "services", "at_home", "other"])  
Mjob_dict = {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4}
Mjob = Mjob_dict[Mjob]  # Encode for prediction

Fjob = st.selectbox("Father's Job", options=["teacher", "health", "services", "at_home", "other"])  
Fjob_dict = {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4}
Fjob = Fjob_dict[Fjob]  # Encode for prediction

reason = st.selectbox("Reason for Choosing School", options=["home", "reputation", "course", "other"])  
reason_dict = {"home": 0, "reputation": 1, "course": 2, "other": 3}
reason = reason_dict[reason]  # Encode for prediction

guardian = st.selectbox("Guardian", options=["mother", "father", "other"])  
guardian_dict = {"mother": 0, "father": 1, "other": 2}
guardian = guardian_dict[guardian]  # Encode for prediction

traveltime = st.slider("Travel Time", min_value=1, max_value=4, value=1, step=1)
studytime = st.slider("Study Time", min_value=1, max_value=4, value=1, step=1)
failures = st.slider("Number of Past Failures", min_value=0, max_value=3, value=0, step=1)

# Additional boolean features
schoolsup = st.selectbox("Extra Educational Support", options=["No", "Yes"])  
schoolsup = 0 if schoolsup == "No" else 1  # Encode for prediction

higher = st.selectbox("Wants to take Higher Education", options=["No", "Yes"])  
higher = 0 if higher == "No" else 1  # Encode for prediction

internet = st.selectbox("Internet Access", options=["No", "Yes"])  
internet = 0 if internet == "No" else 1  # Encode for prediction

romantic = st.selectbox("In a Romantic Relationship", options=["No", "Yes"])  
romantic = 0 if romantic == "No" else 1  # Encode for prediction

goout = st.slider("Going Out", min_value=1, max_value=5, value=1, step=1)
Dalc = st.slider("Weekend Alcohol Consumption", min_value=1, max_value=5, value=1, step=1)
Walc = st.slider("Weekday Alcohol Consumption", min_value=1, max_value=5, value=1, step=1)
health = st.slider("Health Status", min_value=1, max_value=5, value=1, step=1)
absences = st.slider("Number of Absences", min_value=0, max_value=75, value=0, step=1)
G1 = st.slider("G1 Grade", min_value=0, max_value=20, value=0, step=1)
G2 = st.slider("G2 Grade", min_value=0, max_value=20, value=0, step=1)

# Predict button
if st.button("Predict Grade"):
    # Get prediction
    prediction = predict_g3(sex, age, address, Medu, Fedu, Mjob, Fjob, reason, guardian,
                             traveltime, studytime, failures, schoolsup, higher, internet, romantic, 
                             goout, Dalc, Walc, health, absences, G1, G2)  # Total 23 features
    
    # Display result
    st.success(f"Predicted Final Grade: {prediction}")
