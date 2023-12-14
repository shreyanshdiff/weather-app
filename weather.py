import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

model = pkl.load(open('model.pkl', 'rb'))

st.title("Weather Prediction Application")

cols = ['precipitation', 'temp_max', 'temp_min', 'wind']

precipitation = st.slider("Precipitation", 0, 30)
temp_max = st.slider("Maximum Temperature", 0, 20)
temp_min = st.slider("Minimum Temperature", 0, 10)
wind = st.slider("Wind", 0, 10)


def predict():
    row = np.array([precipitation, temp_max, temp_min, wind])
    X = pd.DataFrame([row], columns=cols)
    return X


predict_button = st.button("Predict")
if predict_button:
    input_data = predict()
    weather = model.predict(input_data)
    st.subheader(f"Weather is {weather[0]}")
