import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from joblib import load

from parking.interpret import service_read_data

st.set_page_config(layout='wide')


@st.experimental_singleton
def load_data(filename):
    return service_read_data(filename)


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
                          elevation_scale=4, elevation_range=[0, 50], pickable=True, extruded=True)
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
input_model_list = [load('../joblibs/slu_random_forest.joblib'),
                    load('../joblibs/fh_random_forest.joblib')]

df_list = [load_data('processed-slu-Paid_Parking_Occupancy__Last_30_Days_.csv'),
           load_data('processed-fh-Paid_Parking_Occupancy__Last_30_Days_.csv')]

cols_list = [extract_columns(df_list[0]), extract_columns(df_list[1])]

row_0_0, row_0_1 = st.columns((1, 1))
row_1_0, row_1_1 = st.columns((3, 2))
row_2_0, row_2_1 = st.columns((1, 1))

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
    input_day = st.slider('Select day of the week', 1, 7, value=2)
    input_hour = st.slider('Select hour', 8, 18, value=10)
    input_minute = st.slider('Select minute', 0, 59, value=30)

with row_2_0:
    midpoint = calculate_mid_point(df_list[0]['Latitude'], df_list[0]['Longitude'])
    predicted_df = predict(input_model_list[0], cols_list[0], input_day, input_hour, input_minute)
    print('Predicted with attributes => day: {0}, hour: {1}, minute: {2}'.format(input_day, input_hour, input_minute))
    render_map(predicted_df, midpoint[0], midpoint[1], 14)

with row_2_1:
    midpoint = calculate_mid_point(df_list[1]['Latitude'], df_list[1]['Longitude'])
    predicted_df = predict(input_model_list[1], cols_list[1], input_day, input_hour, input_minute)
    print('Predicted with attributes => day: {0}, hour: {1}, minute: {2}'.format(input_day, input_hour, input_minute))
    render_map(predicted_df, midpoint[0], midpoint[1], 14)
