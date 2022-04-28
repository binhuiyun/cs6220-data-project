from calendar import weekday
from this import s
import streamlit as st
import pandas as pd
import joblib
import math
import datetime

clf = joblib.load('./random-forest.joblib')

st.set_page_config(layout='wide')
st.markdown(
    
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 465px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 465px;
        margin-left: -465px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SLU Parking prediction App")
st.sidebar.title("Features")

def user_input_features():
    data = pd.read_csv('../parking/data/paystub_location.csv')
    blocks =data['BlockfaceName'].unique().tolist()
    blocks = sorted(blocks)
    location_selector = st.sidebar.selectbox(
        "Select a location",
        blocks
    )

    df_street = data[data['BlockfaceName'] == location_selector]
    directions = df_street['SideOfStreet'].tolist()

    direction_selector = st.sidebar.selectbox(
        "Select a direction",
        directions
    )
    d = st.sidebar.date_input(
     "Select a date",
    datetime.date(2022, 3, 22))
    lat = data[(data['BlockfaceName'] == location_selector) & (data['SideOfStreet'] == direction_selector)]['Latitude']
    lon = data[ (data['BlockfaceName'] == location_selector) & (data['SideOfStreet'] == direction_selector)]['Longitude']
    
    spaceCount = data[(data['BlockfaceName'] == location_selector) & (data['SideOfStreet'] == direction_selector)]['ParkingSpaceCount']

    lat = lat.values[0]
    lon = lon.values[0]
    spaceCount = spaceCount.values[0]

    Hour = st.sidebar.slider('Hour', 8, 18, 14)
    Minute= st.sidebar.slider('Minute',0, 60, 1)
   
    data = {   
            'ParkingSpaceCount': spaceCount,
            'WeekDay': weekday(d.year,d.month,d.day),
            'Year': d.year,
            'Month':d.month,
            'Day': d.day,
            'Hour': Hour,
            'Minute':Minute,
            'Latitude': lat,
            'Longitude': lon,
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

def user_confirm():
    res = st.sidebar.button("See Result")
    return res
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)

if user_confirm():
    st.write('---')
    prediction = clf.predict(df)
    st.header('Prediction of Paid Occupancy')
    occupied_spots =round(prediction[0])
    st.write(occupied_spots)

    st.write('---')
    available_spots = df['ParkingSpaceCount'][0] - occupied_spots
    if available_spots > 0:
       
        st.write(available_spots,'parking slots is available!')
    else:
        st.header("Parking is not available!")
