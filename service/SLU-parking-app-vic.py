from pickle import NONE
import streamlit as st
import joblib
import datetime
import pandas as pd
from streamlit_folium import folium_static
import folium
from folium import plugins


def user_input_date():
    date = st.sidebar.date_input(
        "Select a date",
        datetime.date(2022, 3, 22)
    )
    hour = st.sidebar.slider('Hour', 8, 18, 0)
    minute= st.sidebar.slider('Minute',0, 60, 0)

    day_of_the_week = date.weekday()
    minute_of_the_day = hour * 60 + minute

    return (day_of_the_week, minute_of_the_day)

def user_confirm():
    res = st.sidebar.button("See result")
    return res

def generate_model_input_map(day, minute, coordinates):
    num_coor = len(coordinates)
    day_list, minute_list = [], []
    day_list += num_coor * [day]
    minute_list += num_coor * [minute]
    coordinates['DayOfTheWeek'] = day_list
    coordinates['MinuteOfTheDay'] = minute_list
    coordinates = coordinates[['DayOfTheWeek','MinuteOfTheDay','Latitude', 'Longitude']]
    return coordinates

def predict_map(model, day, minute, coordinates):
    map_df = generate_model_input_map(day, minute, coordinates)
    map_out = model.predict(map_df)

    new_df = map_df[['Longitude','Latitude']]
    new_df = pd.concat([new_df, pd.DataFrame(map_out, columns=['count'])], axis = 1)
    points = new_df.to_numpy()
    return points

def generate_model_input_point(day, minute, Latitude, Longitude):
    col = ['DayOfTheWeek', 'MinuteOfTheDay', 'Latitude', 'Longitude']
    user_in = [day, minute, Latitude, Longitude]
    user_df = pd.DataFrame(user_in).T
    user_df.columns = col
    return user_df

def map(array):
    m = folium.Map(location=[47.6256, -122.3344], zoom_start=14)
    if any(array):
        plugins.HeatMap(array).add_to(m)
    folium_static(m)


# Main Panel
model = joblib.load('../parking/random-forest.joblib')
coordinates = pd.read_csv('../parking/data/paystub_coordinates.csv')
day, minute = user_input_date()

m = folium.Map(location=[47.6256, -122.3344], zoom_start=15)
if user_confirm():
    predict_map(model, day, minute, coordinates)

    points = predict_map(model, day, minute, coordinates)
    plugins.HeatMap(points).add_to(m)
folium_static(m)
