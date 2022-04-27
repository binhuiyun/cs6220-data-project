from os import path
from pandas import read_csv

SLU_PARKING_30_CSV = 'slu-Paid_Parking_Occupancy__Last_30_Days_.csv'
PARKING_30_CSV = 'Paid_Parking_Occupancy__Last_30_Days_.csv'
CLEAN_SLU_PARKING_30_CSV = 'available-space-clean-slu-Paid_Parking_Occupancy__Last_30_Days_.csv'


def read_data():
    directory_path = path.abspath(path.dirname(__file__))

    file_path = path.join(directory_path, '..', 'data', SLU_PARKING_30_CSV)

    return read_csv(file_path, verbose=False)


def read_clean_data():
    directory_path = path.abspath(path.dirname(__file__))

    file_path = path.join(directory_path, '..', 'data', CLEAN_SLU_PARKING_30_CSV)

    return read_csv(file_path, nrows=100000,  verbose=False)


def save_data(df):
    directory_path = path.abspath(path.dirname(__file__))

    file_path = path.join(directory_path, '..', 'data', 'available-space-clean-' + SLU_PARKING_30_CSV)

    df.to_csv(file_path, encoding='utf-8', index=False)


def filter_data():
    directory_path = path.abspath(path.dirname(__file__))

    file_path = path.join(directory_path, '..', 'data', PARKING_30_CSV)

    dataframe = read_csv(file_path, verbose=True)

    dataframe = dataframe[dataframe['PaidParkingArea'] == 'South Lake Union']

    dataframe.to_csv('slu-new-file', encoding='utf-8', index=False)

    return dataframe
