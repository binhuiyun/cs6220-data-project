import streamlit as st
import joblib
import datetime
import pandas as pd

# model = joblib.load('../parking/random-forest.joblib')

def user_input_features():
    date = st.sidebar.date_input(
        "Select a date",
        datetime.date(2022, 3, 22)
    )
    hour = st.sidebar.slider('Hour', 8, 18, 14)
    minute= st.sidebar.slider('Minute',0, 60, 1)

    day_of_the_week = date.day_of_week
    minute_of_the_day = hour * 60 + minute

    return (day_of_the_week, minute_of_the_day)


def generate_map_weight(day, minute, coordinates):
    coordinates

def main():
    coordinates = pd.read_csv('../parking/data/paystub_coordinates.csv')
    print(len(coordinates))
main()
