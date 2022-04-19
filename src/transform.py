from os import path
from pandas import read_excel

from constants import EXT_CSV, EXT_XLSX, FILENAME_DICT


def transform_data():
    directory_path = path.abspath(path.dirname(__file__))
    data_path = path.join(directory_path, '../data/states')

    def read_helper(file_with_ext):
        return read_excel(file_with_ext, engine='openpyxl')

    df_dict = dict()
    for file_key, file_name in FILENAME_DICT.items():
        if path.isfile(path.join(data_path, file_name + EXT_CSV)):
            continue
        else:
            df_dict[file_key] = read_helper(
                path.join(data_path, file_name + EXT_XLSX))

    for file_key, file_df in df_dict.items():
        file_df.to_csv(path.join(data_path, FILENAME_DICT[file_key] + EXT_CSV),
                       encoding='utf-8', index=False)
