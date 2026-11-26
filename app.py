import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Cache the data loading and training so it's fast
@st.cache_resource
def load_and_train_model():
    # 1. Load the data from the repo
    df = pd.read_csv('weatherAUS.csv')
    df = df.dropna(subset=['RainTomorrow'])
    df = df.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'RISK_MM'], errors='ignore')
    df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

    X = df.drop(columns=['Date', 'RainTomorrow'])
    y = df['RainTomorrow']

    # 2. Build a quick pipeline
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)])

    # Use a small model to train fast
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42))])

    pipe.fit(X, y)
    return pipe

# Show a spinner while it trains
with st.spinner("Training model on startup (this takes 10 seconds)..."):
    model = load_and_train_model()

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
