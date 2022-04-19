from os import path
from pandas import read_csv

from constants import EXT_CSV, FILENAME_DICT, FILENAME_YEAR_DICT


def interpret_data():
    directory_path = path.abspath(path.dirname(__file__))
    data_path = path.join(directory_path, '../data/states')

    def __interpret_helper(folder_path, file_name, file_key):
        df = read_csv(path.join(folder_path, file_name + EXT_CSV))
        df['YEAR'] = FILENAME_YEAR_DICT[file_key]
        return df

    return {file_key: __interpret_helper(data_path, file_name, file_key)
            for file_key, file_name in FILENAME_DICT.items()}
