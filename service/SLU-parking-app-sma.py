from os import path
from calendar import weekday
import datetime
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from joblib import load
from pandas import read_csv

st.set_page_config(layout='wide')


@st.experimental_singleton
def load_model_file(input_filename):
    file_path = path.join(path.abspath(path.dirname(__file__)), '..', 'joblibs', input_filename)
    print(file_path)
    return load(file_path)


@st.experimental_singleton
def load_data_file(input_filename):
    file_path = path.join(path.abspath(path.dirname(__file__)), '..', 'data', input_filename)
    print(file_path)
    return read_csv(file_path)


@st.experimental_singleton
def extract_columns(dataframe):
    print(dataframe)
    return dataframe['Latitude'].tolist(), dataframe['Longitude'].tolist()


@st.experimental_memo
def calculate_mid_point(lat, lon):
    return np.average(lat), np.average(lon)


def render_map(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style='mapbox://styles/mapbox/streets-v11',
            initial_view_state={'latitude': lat, 'longitude': lon, 'zoom': zoom, 'pitch': 50},
            layers=[
                pdk.Layer('HexagonLayer', data=data, get_position=['Longitude', 'Latitude'], radius=18,
                          elevation_scale=4, elevation_range=[0, 100], pickable=True, extruded=True)
            ]
        )
    )


def predict(model, cols, day, hour, minute):
    locations = list(set(zip(cols[0], cols[1])))
    prediction_input_rows = [[day, hour, minute, location[0], location[1]] for location in locations]

    prediction_input_df = pd.DataFrame(prediction_input_rows)
    prediction_input_df.columns = ['DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute', 'Latitude', 'Longitude']

    space_counts = [int(count) for count in model.predict(prediction_input_df)]
    result_rows = [[space_counts[i], day, hour, minute, location[0], location[1]]
                   for i, location in enumerate(locations)]

    print(space_counts[0:5])
    print(result_rows[0:5])

    # transform results
    transformed_result_rows = []
    for result_row in result_rows:
        for i in range(result_row[0]):
            transformed_result_rows.append([1, result_row[1], result_row[2], result_row[3],
                                            result_row[4], result_row[5]])

    result_df = pd.DataFrame(transformed_result_rows)
    result_df.columns = ['AvailableSpaceCount', 'DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute',
                         'Latitude', 'Longitude']
    return result_df


# data model initialization
input_model_list = [load_model_file('../joblibs/out_South_Lake_Union_random_forest.joblib.gz'),
                    load_model_file('../joblibs/out_Capitol_Hill_random_forest.joblib.gz'),
                    load_model_file('../joblibs/out_Denny_Triangle_random_forest.joblib.gz'),
                    load_model_file('../joblibs/out_First_Hill_random_forest.joblib.gz')]

df_list = [load_data_file('out_South_Lake_Union_Paid_Parking_Occupancy__Last_30_Days_.csv.gz'),
           load_data_file('out_Capitol_Hill_Paid_Parking_Occupancy__Last_30_Days_.csv.gz'),
           load_data_file('out_Denny_Triangle_Paid_Parking_Occupancy__Last_30_Days_.csv.gz'),
           load_data_file('out_First_Hill_Paid_Parking_Occupancy__Last_30_Days_.csv.gz')]

cols_list = [extract_columns(df_list[0]), extract_columns(df_list[1]),
             extract_columns(df_list[2]), extract_columns(df_list[3])]

row_0_0, row_0_1 = st.columns((1, 1))
row_1_0, row_1_1 = st.columns((3, 2))
row_2_0, row_2_1 = st.columns((1, 1))
row_3_0, row_3_1 = st.columns((1, 1))

with row_0_0:
    st.title('Seattle City Paid Parking Prediction')

with row_1_0:
    st.write(
        """
    ##
 The City of Seattle has created an on-street paid parking occupancy data set and is providing access to this data set
 for public use for research and entrepreneurial purposes under the City’s Open Data Program. In 2017-2018, 
 the Seattle Information Technology (Seattle IT) and Seattle Department of Transportation (SDOT) worked collaboratively 
 on a project to determine the rate of paid occupancy in the city’s paid parking system. The City is providing data to 
 researchers and programmers for analysis and to develop applications that might help improve parking management 
 conditions. SDOT’s interest is to make data-driven parking decisions and to help people access parking information 
 so that they can find a space easier and spend less time circling, stuck in traffic.
    """
    )

with row_1_1:
    
    d = st.date_input(
     "Select a date",
    datetime.date(2022, 4, 28))
    input_day =  weekday(d.year,d.month,d.day)
    input_hour = st.slider('Select hour', 8, 18, value=10)
    input_minute = st.slider('Select minute', 0, 59, value=30)


def render_map_handler(index):
    midpoint = calculate_mid_point(df_list[index]['Latitude'], df_list[index]['Longitude'])
    predicted_df = predict(input_model_list[index], cols_list[index], input_day, input_hour, input_minute)
    print('Predicted with attributes => day: {0}, hour: {1}, minute: {2}\n'.format(input_day, input_hour, input_minute))
    render_map(predicted_df, midpoint[0], midpoint[1], 14)


with row_2_0:
    render_map_handler(0)

with row_2_1:
    render_map_handler(1)

with row_3_0:
    render_map_handler(2)

with row_3_1:
    render_map_handler(3)
