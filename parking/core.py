from time import time

from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from pandas import to_datetime
from parse import parse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mutual_info_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from interpret import save_data, read_data_by_filename

SOURCE_FILENAME = 'Paid_Parking_Occupancy__Last_30_Days_.csv'
TARGET_AREA = 'South Lake Union'


def evaluate_accuracy(model, x_train, y_train, x_test, y_test):
    prediction = model.predict(x_test)
    r_square_score = r2_score(y_test, prediction)
    print('{0} R-Squared: {1}'.format(model.__class__.__name__, r_square_score))
    print(cross_val_score(model, x_train, y_train, cv=6))
    return r_square_score


def gradient_boosting_regression(x_train, x_test, y_train, y_test):
    model = GradientBoostingRegressor(max_depth=16)
    model.fit(x_train, y_train)

    r_square_score = evaluate_accuracy(model, x_train, y_train, x_test, y_test)
    dump(model, str(r_square_score) + '_out_' + TARGET_AREA.replace(' ', '_') + '_gradient_boosting.joblib')


def random_forest_regression(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor(max_depth=16, n_jobs=-1)
    model.fit(x_train, y_train)

    r_square_score = evaluate_accuracy(model, x_train, y_train, x_test, y_test)
    dump(model, str(r_square_score) + '_out_' + TARGET_AREA.replace(' ', '_') + '_random_forest.joblib')


def k_neighbors_regression(x_train, x_test, y_train, y_test):
    model = KNeighborsRegressor(n_neighbors=16)
    model.fit(x_train, y_train)

    r_square_score = evaluate_accuracy(model, x_train, y_train, x_test, y_test)
    dump(model, str(r_square_score) + '_out_' + TARGET_AREA.replace(' ', '_') + '_k_neighbors.joblib')


def linear_regression(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)

    r_square_score = evaluate_accuracy(model, x_train, y_train, x_test, y_test)
    dump(model, str(r_square_score) + '_out_' + TARGET_AREA.replace(' ', '_') + '_linear.joblib')


def expand_data(dataframe):
    # create new feature columns
    dataframe['FormedOccupancyDateTime'] = to_datetime(dataframe['OccupancyDateTime'], format='%m/%d/%Y %I:%M:%S %p')
    dataframe['DateTimeWeekDay'] = dataframe['FormedOccupancyDateTime'].dt.dayofweek
    dataframe['DateTimeHour'] = dataframe['FormedOccupancyDateTime'].dt.hour
    dataframe['DateTimeMinute'] = dataframe['FormedOccupancyDateTime'].dt.minute
    dataframe['DateTimeSecond'] = dataframe['FormedOccupancyDateTime'].dt.second
    dataframe['LocationStr'] = dataframe['Location'].astype('str')
    dataframe['AvailableSpaceCount'] = dataframe["ParkingSpaceCount"] - dataframe["PaidOccupancy"]
    dataframe = dataframe[dataframe['AvailableSpaceCount'] >= 0]

    # feature creation helpers
    def __point_extractor(point_str):
        parsed = parse('POINT ({} {})', point_str)
        return parsed[0], parsed[1]

    dataframe['Latitude'] = dataframe['LocationStr'].apply(lambda point_str: __point_extractor(point_str)[1])
    dataframe['Longitude'] = dataframe['LocationStr'].apply(lambda point_str: __point_extractor(point_str)[0])

    dataframe['BlockNameWithStreetSide'] = dataframe["BlockfaceName"] + ' - ' + dataframe["SideOfStreet"]

    # clean up unused columns
    dataframe = dataframe[['BlockNameWithStreetSide', 'AvailableSpaceCount', 'Latitude', 'Longitude',
                           'DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute']]

    # remove missing data
    dataframe = dataframe.dropna()

    print(dataframe)
    print(dataframe.columns)
    print(dataframe.corr())

    return dataframe


def train_models(dataframe):
    x_features = ['DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute', 'Latitude', 'Longitude']
    y_feature = 'AvailableSpaceCount'

    # prepare training dataset
    x, y = dataframe[x_features], dataframe[y_feature]

    # under-sampling
    random_under_sampler = RandomUnderSampler()
    x, y = random_under_sampler.fit_resample(x, y)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # execute model training
    for function in [linear_regression, k_neighbors_regression, random_forest_regression, gradient_boosting_regression]:
        function(x_train, x_test, y_train, y_test)


def main():
    start_time = time()

    # perform source dataset pre-processing and generate enhanced dataset (can be skipped)
    raw_df = read_data_by_filename(SOURCE_FILENAME)
    filtered_df = raw_df[raw_df['PaidParkingArea'] == TARGET_AREA]
    expanded_df = expand_data(filtered_df)
    save_data(expanded_df, 'out_' + TARGET_AREA.replace(' ', '_') + '_' + SOURCE_FILENAME)

    # execute model training
    clean_df = read_data_by_filename('out_' + TARGET_AREA.replace(' ', '_') + '_' + SOURCE_FILENAME)
    train_models(clean_df)

    print("--- %s seconds ---" % (time() - start_time))


if __name__ == "__main__":
    main()
