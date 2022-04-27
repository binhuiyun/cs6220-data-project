from os import path
from pandas import read_csv


def read_data_by_filename(input_filename):
    directory_path = path.abspath(path.dirname(__file__))

    file_path = path.join(directory_path, '..', 'data', input_filename)

    return read_csv(file_path, verbose=True)


def service_read_data(input_filename):
    directory_path = path.abspath(path.dirname(__file__))

    file_path = path.join(directory_path, '..', 'data', input_filename)

    return read_csv(file_path, nrows=200000, verbose=False)


def save_data(df, output_filename):
    directory_path = path.abspath(path.dirname(__file__))

    file_path = path.join(directory_path, '..', 'data', output_filename)

    df.to_csv(file_path, encoding='utf-8', index=False)
