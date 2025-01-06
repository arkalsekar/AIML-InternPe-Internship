import os
import pickle
import streamlit as st
import numpy as np 
import pandas as pd 

st.title("Car Price Predictor")
name = st.text_input("Enter the Name of the Car", value="Maruti Suzuki Wagon", key=1)
company = st.selectbox("Select the Company Name", ('Maruti', 'Mahindra', 'Ford', 'Hyndai', 'Skoda', 'Audi', 'Toyota',
       'Renault', 'Honda', 'Datsun', 'Mitsubishi', 'Tata', 'Volkswagen',
       'Chevrolet', 'Mini', 'BMW', 'Nissan', 'Hindustan', 'Fiat', 'Force',
       'Mercedes', 'Land', 'Jaguar', 'Jeep', 'Volvo'), key=2)
year = st.selectbox("Select the Year", (2007, 2006, 2014, 2012, 2013, 2016, 2015, 2010, 2017, 2008, 2018,
       2011, 2019, 2009, 2005, 2000, 2003, 2004, 1995, 2002, 2001), key=3)
kms_driven = st.number_input("Enter the KMS Driven", value=200, key=4)	
fuel_type = st.selectbox("Select the fuel type", ("Petrol", "Diesel", "LPG"), key=5)
predict = st.button("Predict", )


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

if predict:
    price = model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
    data=np.array([str(name), str(company), int(year), int(kms_driven), str(fuel_type)]).reshape(1,5)))
    st.text(f"The Predicted Price for the given details could be {round(price[0])}")
