import pickle 
import numpy as np 
import streamlit as st 

st.title("Diabetes Prediction")

# Inputs
pregnancies = st.number_input("Enter Pregnancies", key=1, value=6)	
glucose = st.number_input("Enter Glucose Level", key=2, value=148)	
bloodPressure = st.number_input("Enter Blood Pressure Level", key=3, value=72)	
skinThickness = st.number_input("Enter the Skin Thickness", key=4, value=35)	
insulin = st.number_input("Enter the Insulin Level", key=5, value=0)	
BMI = st.number_input("Enter the BMI", key=6, value=33.6)	
diabetesPedigreeFunction = st.number_input("Enter the Diabetes Pedigree Function", key=7, value=0.627)	
age = st.number_input("Enter the Age", key=8, value=50)

predict = st.button("Predict")
# load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

if predict:
    a = model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    if a[0] == 0:
        st.text("You are not Prone to Diabetes")
    elif a[0] == 1:
        st.text("You are Prone to be diabetic")


st.text("Disclaimer : This is a Machine Learning Model predictions, So please take a medical advice before taking any serious steps")