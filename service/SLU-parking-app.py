from calendar import weekday
import streamlit as st
import pandas as pd
import joblib
import math
import datetime

#clf = joblib.load('./model/random-forest.joblib')

st.title("SLU Parking prediction App")
st.sidebar.title("Features")

def user_input_features():
    data = pd.read_csv('slu-Paid_Parking_Occupancy__Last_30_Days_.csv')
    blocks =data['BlockfaceName'].unique().tolist()
    blocks = sorted(blocks)
    location_selector = st.sidebar.selectbox(
        "Select a location",
        blocks
    )
    directions = data["SideOfStreet"].unique().tolist()
    direction_selector = st.sidebar.selectbox(
        "Select a direction",
        directions
    )
    d = st.sidebar.date_input(
     "Select a date",
    datetime.date(2022, 3, 22))


 #   Key = st.sidebar.slider('Key', 8401, 134965, 79026)
    
   # ParkingSpaceCount = st.sidebar.slider('ParkingSpaceCount',1, 29, 7)


    Hour = st.sidebar.slider('Hour', 8, 18, 14)
    Minute= st.sidebar.slider('Minute',0, 60, 1)
   
    data = {#'Key': Key,
            'Blockface': location_selector,
            'SideOfStreet': direction_selector,
            #'ParkingSpaceCount': ParkingSpaceCount,
            'WeekDay': weekday(d.year,d.month,d.day),
            'Year': d.year,
            'Month':d.month,
            'Day': d.day,
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

# st.write('---')
# prediction = clf.predict(df)
# st.header('Prediction of Paid Occupancy')
# st.write(prediction)

# st.write('---')
# if df['ParkingSpaceCount'][0] >math.ceil(prediction[0]):
#     st.header('Parking is available!')
# else:
#     st.header("Parking is not available!")