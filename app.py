import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('models/G07_model_pickle.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Mvua Alert: Rain Tomorrow Predictor")
st.write("Group G7: Lawrence, Felix, Martin, Winfred, Swynneystone")

st.header("Enter Today's Weather:")
min_temp = st.number_input("MinTemp (°C)", value=15.0)
max_temp = st.number_input("MaxTemp (°C)", value=25.0)
rainfall = st.number_input("Rainfall (mm)", value=0.0)
humidity3pm = st.number_input("Humidity3pm (%)", value=50.0)
pressure3pm = st.number_input("Pressure3pm (hPa)", value=1015.0)
wind_gust = st.number_input("WindGustSpeed (km/h)", value=30.0)
rain_today = st.selectbox("RainToday", ["No", "Yes"])
location = st.selectbox("Location", ["Albury", "Sydney", "Melbourne", "Darwin", "Perth"])

if st.button("Predict Rain Tomorrow"):
    input_data = pd.DataFrame({
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'Humidity3pm': [humidity3pm],
        'Pressure3pm': [pressure3pm],
        'WindGustSpeed': [wind_gust],
        'RainToday': [0 if rain_today == "No" else 1],
        'Location': [location]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.error(f"🌧️ **Rain Expected Tomorrow!** (Probability: {prob:.2f})")
            st.write("**SMS Text:** Mvua Alert: Rain is expected tomorrow. Please take precautions.")
        else:
            st.success(f"☀️ **No Rain Expected Tomorrow.** (Probability: {prob:.2f})")
            st.write("**SMS Text:** Mvua Alert: No rain expected tomorrow. Have a safe day.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
