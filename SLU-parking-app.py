import streamlit as st
import pandas as pd
import datetime
import pydeck as pdk
from os import path
import joblib

# Method to display date side bar and gather user data time input
def user_input_date():
    date = st.sidebar.date_input(
        "Select a date",
        datetime.date(2022, 3, 22)
    )
    hour = st.sidebar.slider('Hour', 8, 20, 8)
    minute= st.sidebar.slider('Minute',0, 60, 0)

    day_of_the_week = date.weekday()
    minute_of_the_day = hour * 60 + minute

    return (day_of_the_week, minute_of_the_day)

# Method to display block and street side drop down menu and gather user street input
def user_input_street(data):
    areas = ['South Lake Union', 'Denny Triangle', 'First Hill', 'Capitol Hill']

    area_selector = st.sidebar.selectbox(
        "Select an area",
        areas
    )

    df_area = data.loc[data['PaidParkingArea'] == area_selector]
    blocks = df_area['BlockfaceName'].unique().tolist()
    blocks = sorted(blocks)

    location_selector = st.sidebar.selectbox(
        "Select a street name",
        blocks
    )

    df_street = df_area.loc[df_area['BlockfaceName'] == location_selector]
    sides = df_street['SideOfStreet'].tolist()
    if len(sides) > 0:
        sides = sorted(sides)
        side_selector = st.sidebar.selectbox(
            "Select a street side",
            sides
        )
        df = df_street.loc[df_street['SideOfStreet'] == side_selector]
    else:
        df = df_street

    lat = df['Latitude']
    lon = df['Longitude']
    count = df['ParkingSpaceCount'].iloc[0]

    return lat, lon, count

# Method to display "See Result" and gather on click response
def user_confirm():
    res = st.sidebar.button("See Result")
    return res

# Method to predict parking spaces for a given coordinates
def predict_street(model, day, minute, Latitude, Longitude):
    col = ['DayOfTheWeek', 'MinuteOfTheDay', 'Latitude', 'Longitude']
    user_in = [day, minute, Latitude, Longitude]
    user_df = pd.DataFrame(user_in).T
    user_df.columns = col
    res = model.predict(user_df)
    return res

# Method to generate a data frame that contains all data points
# for model prediction 
def generate_model_input(day, minute, df):
    res = df[['Latitude', 'Longitude']]
    df_length = len(df)

    day_list, minute_list = [], []
    day_list += df_length * [day]
    minute_list += df_length * [minute]
    res['DayOfTheWeek'] = day_list
    res['MinuteOfTheDay'] = minute_list

    res = res[['DayOfTheWeek','MinuteOfTheDay','Latitude', 'Longitude']]
    return res

# Method to predict parking spaces for all coordinates in df
# return a dataframe that will be used in pydeck layer
def predict_map(model, day, minute, df):
    map_df = generate_model_input(day, minute, df)
    map_out = model.predict(map_df)
    map_out = [round(i) for i in map_out]
    res = pd.DataFrame(columns=['lng', 'lat'])

    map_out = [round(i) for i in map_out]
    res = pd.DataFrame(columns=['lng', 'lat'])
    for i in range(len(map_out)):
        for j in range(map_out[i]):
            to_append = [df['Longitude'][i], df['Latitude'][i]]
            a_series = pd.Series(to_append, index = res.columns)
            res = res.append(a_series, ignore_index=True)
    return res


# Main Panel
model = joblib.load('data/rf-model.joblib')
paystubs = pd.read_csv('data/paystub_location.csv')

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

st.title("Seattle Paid Parking Prediction")

day, minute = user_input_date()
lat, lon, count = user_input_street(paystubs)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=-122.3344,
    latitude=47.615,
    zoom=13,
    min_zoom=5,
    max_zoom=20,
    pitch=50
)

r = pdk.Deck(initial_view_state=view_state, map_style='mapbox://styles/mapbox/streets-v11')

if user_confirm():

    pts = predict_map(model, day, minute, paystubs)
    layer = pdk.Layer(
        'HexagonLayer',  # `type` positional argument is here
        pts,
        get_position=['lat','lng'],
        radius=18,
        elevation_scale=4, 
        elevation_range=[0, 100], 
        pickable=True, 
        extruded=True
    )

    r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/streets-v11')

    num = predict_street(model, day, minute, lat, lon)
    data = [[day, minute, lat.iloc[0], lon.iloc[0]]]

    inputs = pd.DataFrame(data, columns=['DayOfTheWeek','MinuteOfTheDay','Latitude','Longitude'])
    st.write("Inputs for machine learning model")
    st.write(inputs)

    if num[0] > count:
        st.write(count, "parking spaces are available on your selected street.")
    elif count > num[0] > 0:
        st.write(round(num[0]), "parking spaces are available on your selected street.")
    else:
        st.write('Sorry, no available parking on your selected street.')

st.pydeck_chart(pydeck_obj=r, use_container_width=False)
