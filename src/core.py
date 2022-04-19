from time import time

from transform import transform_data
from interpret import interpret_data
from models import linear_regression, random_forest_regression
from extract import extract_data_by_occ_title


def driver_callback(**kwargs):
    start_time = time()

    # transform data format
    kwargs['transform_data']()

    # load data into dataframe
    df = kwargs['interpret_data']()

    # extract methods
    consolidated_df = kwargs['extract_data'](df)

    # process models
    for process_model in kwargs['process_models']:
        process_model(consolidated_df)

    print("--- %s seconds ---" % (time() - start_time))


def main():
    driver_callback(transform_data=transform_data,
                    interpret_data=interpret_data,
                    extract_data=extract_data_by_occ_title,
                    process_models=[linear_regression, random_forest_regression])


if __name__ == "__main__":
    main()
