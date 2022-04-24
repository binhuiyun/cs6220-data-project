from calendar import weekday
import streamlit as st
import pandas as pd
import joblib
import math

clf = joblib.load('./model/random-forest.joblib')

st.title("SLU Parking prediction App")
st.sidebar.title("Features")

def user_input_features():
    Key = st.sidebar.slider('Key', 8401, 134965, 79026)
    ParkingSpaceCount = st.sidebar.slider('ParkingSpaceCount',1, 29, 7)
    WeekDay = st.sidebar.slider('Weekday', 0, 6, 5)
    Year = st.sidebar.slider('Year',2022, 2024, 2022)
    Month = st.sidebar.slider('Month', 1, 12, 4)
    Day= st.sidebar.slider('Day', 1, 31, 16)
    Hour = st.sidebar.slider('Hour', 8, 18, 14)
    Minute= st.sidebar.slider('Minute',0, 60, 1)
   
    data = {'Key': Key,
            'ParkingSpaceCount': ParkingSpaceCount,
            'WeekDay': WeekDay,
            'Year': Year,
            'Month': Month,
            'Day': Day,
            'Hour': Hour,
            'Minute':Minute,
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)

st.write('---')
prediction = clf.predict(df)
st.header('Prediction of Paid Occupancy')
st.write(prediction)

st.write('---')
if df['ParkingSpaceCount'][0] >math.ceil(prediction[0]):
    st.header('Parking is available!')
else:
    st.header("Parking is not available!")