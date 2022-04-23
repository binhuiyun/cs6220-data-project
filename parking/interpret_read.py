from os import path
from pandas import read_csv

from constants import EXT_CSV, FILENAME_DICT, FILENAME_YEAR_DICT


def interpret_data():
    directory_path = path.abspath(path.dirname(__file__))
    data_path = path.join(directory_path)

    file_name_with_ext = 'slu-Paid_Parking_Occupancy__Last_30_Days_.csv'

    return read_csv(file_name_with_ext, verbose=False)
