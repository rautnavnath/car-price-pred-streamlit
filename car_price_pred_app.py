import streamlit as st
import pandas as pd

import datetime


st.header('Cars24 car price', divider='rainbow')
#st.header('_Streamlit_ is :blue[cool] :sunglasses:')
#st.write('Daily Close chart')
#df=pd.read_csv()
col1, col2=st.columns(2)
with col1:
    fuel_type = st.selectbox(
        "Select Fuel type:",
        ("Petrol", "Diesel", "CNG", 'Electric'))
    Engine = st.slider("Engine CC", 900, 1800, step=100)
with col2:
    trans_type = st.selectbox(
        "Select trans type:",
        ("Auto", "Manual"))
    Seats = st.slider("Seats", 1, 10, step=1)

encode_dict= { 'fuel_type':{'Petrol':1, 'Diesel':2, 'CNG':3, 'Electric':4}, 'trans_type':{'manual':1, 'Auto':2}}
import pickle
def model_pred(fuel_encoded, trans_encoded):
    with open('car_pred','rb') as file:
        reg_model=pickle.load(file)
    imput_features=[[2012,1,120000,fuel_encoded,trans_encoded, 19.7, Engine, 47,Seats]]

    price=reg_model.predict(imput_features)
    return price
#st.button("Reset", type="primary")
if st.button("Predict"):
    #st.write("Why hello there")
    fuel_encoded=encode_dict['fuel_type'][fuel_type]
    trans_encoded=encode_dict['trans_type'][trans_type]
    price=model_pred(fuel_encoded, trans_encoded)
    st.text('Predicted Price' +str(price))
#else:
    #st.write("Goodbye")