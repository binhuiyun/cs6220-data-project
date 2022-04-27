from calendar import weekday
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import datetime
import pydeck as pdk

# clf = joblib.load('./model/random-forest.joblib')
from sklearn.preprocessing import LabelEncoder

from parking.interpret import read_clean_data

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

st.title("SLU Parking prediction App")
st.sidebar.title("Features")

# def user_input_features():
#     df = read_clean_data()
#
#     data = pd.read_csv('data/slu-Paid_Parking_Occupancy__Last_30_Days_.csv')
#     blocks = data['BlockfaceName'].unique().tolist()
#     blocks = sorted(blocks)
#     location_selector = st.sidebar.selectbox(
#         "Select a location",
#         blocks
#     )
#     directions = data["SideOfStreet"].unique().tolist()
#     direction_selector = st.sidebar.selectbox(
#         "Select a direction",
#         directions
#     )
#     d = st.sidebar.date_input(
#         "Select a date",
#         datetime.date(2022, 3, 22))
#
#     #   Key = st.sidebar.slider('Key', 8401, 134965, 79026)
#
#     # ParkingSpaceCount = st.sidebar.slider('ParkingSpaceCount',1, 29, 7)
#
#     Hour = st.sidebar.slider('Hour', 8, 18, 14)
#     Minute = st.sidebar.slider('Minute', 0, 60, 1)
#
#     data = {  # 'Key': Key,
#         'Blockface': location_selector,
#         'SideOfStreet': direction_selector,
#         # 'ParkingSpaceCount': ParkingSpaceCount,
#         'WeekDay': weekday(d.year, d.month, d.day),
#         'Year': d.year,
#         'Month': d.month,
#         'Day': d.day,
#         'Hour': Hour,
#         'Minute': Minute,
#     }
#     features = pd.DataFrame(data, index=[0])
#     return features
#
#
# df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
# st.write(df)


df = read_clean_data()


# CALCULATE MIDPOINT FOR GIVEN SET OF DATA
@st.experimental_memo
def mid_point(lat, lon):
    return np.average(lat), np.average(lon)


midpoint = mid_point(df["Latitude"], df["Longitude"])

print(midpoint)


def map(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["Longitude", "Latitude"],
                    radius=10,
                    elevation_scale=4,
                    elevation_range=[0, 100],
                    pickable=True,
                    extruded=True,
                ),
            ],
        )
    )

label_encoder = LabelEncoder()
for column_name in df.columns:
    if column_name == 'BlockNameWithStreetSide':
        df[column_name] = label_encoder.fit_transform(
            df[column_name])

df = df.dropna()
print(df)

# print(df['BlockNameWithStreetSide'].corr(df['SourceElementKey']))

print(df.corr())

map(df[(df['DateTimeHour'] == 15) & (df['AvailableSpaceCount'] > 1)], midpoint[0], midpoint[1], 12)

# st.write('---')
# prediction = clf.predict(df)
# st.header('Prediction of Paid Occupancy')
# st.write(prediction)

# st.write('---')
# if df['ParkingSpaceCount'][0] >math.ceil(prediction[0]):
#     st.header('Parking is available!')
# else:
#     st.header("Parking is not available!")
