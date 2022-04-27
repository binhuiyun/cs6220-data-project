from time import time
import re

from interpret import read_data, save_data, read_clean_data

from parse import parse

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mutual_info_score
from sklearn.feature_extraction import DictVectorizer

from imblearn.under_sampling import RandomUnderSampler

from joblib import dump

from pandas import to_datetime


def evaluate_accuracy(model, x_train, y_train, x_test, y_test):
    prediction = model.predict(x_test)
    print('{0} R-Squared: {1}'.format(model.__class__.__name__, r2_score(y_test, prediction)))
    # print(cross_val_score(model, x_train, y_train, cv=6))


def random_forest_regression(dataframe, x_features, y_feature):
    # # process string values
    # label_encoder = LabelEncoder()
    # for column_name in dataframe.columns:
    #     print('column name:', column_name)
    #     if dataframe[column_name].dtype == object:
    #         dataframe[column_name] = label_encoder.fit_transform(
    #             dataframe[column_name])

    # x = dataframe.drop(['OccupancyDateTime', 'FormedOccupancyDateTime', 'PaidOccupancy',
    #                     'ParkingTimeLimitCategory', 'PaidParkingRate'], axis=1)
    # y = dataframe['PaidOccupancy']

    x = dataframe[x_features]
    y = dataframe[y_feature]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    random_under_sampler = RandomUnderSampler(random_state=42)
    x, y = random_under_sampler.fit_resample(x, y)

    model = RandomForestRegressor(max_depth=8, random_state=0, n_jobs=-1)
    model.fit(x_train, y_train)

    # dump(model, 'random-forest.joblib')

    evaluate_accuracy(model, x_train, y_train, x_test, y_test)


def k_neighbors_regression(dataframe, x_features, y_feature):
    label_encoder = LabelEncoder()
    # for column_name in dataframe.columns:
    #     if dataframe[column_name].dtype == object:
    #         dataframe[column_name] = label_encoder.fit_transform(
    #             dataframe[column_name])

    x = dataframe[x_features]
    y = dataframe[y_feature]

    random_under_sampler = RandomUnderSampler(random_state=42)
    x, y = random_under_sampler.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    model = KNeighborsRegressor(n_neighbors=8)
    model.fit(x_train, y_train)

    evaluate_accuracy(model, x_train, y_train, x_test, y_test)


def linear_regression(dataframe, x_features, y_feature):
    label_encoder = LabelEncoder()
    # for column_name in dataframe.columns:
    #     if dataframe[column_name].dtype == object:
    #         dataframe[column_name] = label_encoder.fit_transform(
    #             dataframe[column_name])

    x = dataframe[x_features]
    y = dataframe[y_feature]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    random_under_sampler = RandomUnderSampler(random_state=42)
    x, y = random_under_sampler.fit_resample(x, y)

    model = LinearRegression()
    model.fit(x_train, y_train)

    print('linear model score:', model.score(x_train, y_train))

    evaluate_accuracy(model, x_train, y_train, x_test, y_test)


def check_correlation(dataframe):
    print(dataframe.corr())


def process(dataframe):
    # filter by block face name
    # block_face_name = 'JOHN ST BETWEEN WESTLAKE AVE N AND TERRY AVE N'
    # dataframe = dataframe[dataframe['BlockfaceName'] == block_face_name]
    # return block_dataframe

    dataframe = dataframe.dropna(subset=['Location'])

    print(dataframe.head(10))

    # dataframe.to_csv('terry.csv', encoding='utf-8', index=False)
    # dataframe = dataframe.dropna()

    dataframe['FormedOccupancyDateTime'] = to_datetime(
        dataframe['OccupancyDateTime'], format='%m/%d/%Y %I:%M:%S %p')
    dataframe['DateTimeWeekDay'] = dataframe['FormedOccupancyDateTime'].dt.dayofweek
    dataframe['DateTimeHour'] = dataframe['FormedOccupancyDateTime'].dt.hour
    dataframe['DateTimeMinute'] = dataframe['FormedOccupancyDateTime'].dt.minute
    dataframe['DateTimeSecond'] = dataframe['FormedOccupancyDateTime'].dt.second
    dataframe['LocationStr'] = dataframe['Location'].astype('str')

    print(dataframe.dtypes)

    def extract_latitude_helper(point_str):
        result = re.findall('\\(.*?\\)', point_str)
        print(result)
        return result

    format_point = 'POINT ({} {})'

    def extractor(point_str):
        parsed = parse(format_point, point_str)
        return parsed[0], parsed[1]

    dataframe['Latitude'] = dataframe['LocationStr'].apply(lambda point_str: extractor(point_str)[1])
    dataframe['Longitude'] = dataframe['LocationStr'].apply(lambda point_str: extractor(point_str)[0])

    dataframe['BlockNameWithStreetSide'] = dataframe["BlockfaceName"] + ' - ' + dataframe["SideOfStreet"]

    print(dataframe)
    print(dataframe.columns)

    df = dataframe[['BlockNameWithStreetSide', 'SourceElementKey', 'ParkingSpaceCount', 'PaidOccupancy',
                    'Latitude', 'Longitude',
                    'DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute', 'ParkingTimeLimitCategory']]

    save_data(df)

    label_encoder = LabelEncoder()
    for column_name in dataframe.columns:
        if dataframe[column_name].dtype == object:
            dataframe[column_name] = label_encoder.fit_transform(
                dataframe[column_name])
    # vec = DictVectorizer()

    # label_encoder = LabelEncoder()
    # for column_name in dataframe.columns:
    #     print('column name:', column_name)
    #     if dataframe[column_name].dtype == object:
    #         dataframe[column_name] = label_encoder.fit_transform(
    #             dataframe[column_name])

    check_correlation(dataframe)

    # # 'OccupancyDateTime'
    # x_features = ['BlockfaceName', 'SideOfStreet', 'ParkingSpaceCount', 'PaidParkingArea', 'ParkingTimeLimitCategory',
    #               'DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute', 'DateTimeSecond']
    # y_feature = 'PaidOccupancy'
    #
    # linear_regression(dataframe, x_features, y_feature)
    # k_neighbors_regression(dataframe, x_features, y_feature)
    # random_forest_regression(dataframe, x_features, y_feature)

    return dataframe


def process_available(dataframe):
    # filter by block face name
    # block_face_name = 'JOHN ST BETWEEN WESTLAKE AVE N AND TERRY AVE N'
    # dataframe = dataframe[dataframe['BlockfaceName'] == block_face_name]
    # return block_dataframe

    dataframe = dataframe.dropna(subset=['Location'])

    print(dataframe.head(10))

    # dataframe.to_csv('terry.csv', encoding='utf-8', index=False)
    # dataframe = dataframe.dropna()

    dataframe['FormedOccupancyDateTime'] = to_datetime(
        dataframe['OccupancyDateTime'], format='%m/%d/%Y %I:%M:%S %p')
    dataframe['DateTimeWeekDay'] = dataframe['FormedOccupancyDateTime'].dt.dayofweek
    dataframe['DateTimeHour'] = dataframe['FormedOccupancyDateTime'].dt.hour
    dataframe['DateTimeMinute'] = dataframe['FormedOccupancyDateTime'].dt.minute
    dataframe['DateTimeSecond'] = dataframe['FormedOccupancyDateTime'].dt.second
    dataframe['LocationStr'] = dataframe['Location'].astype('str')

    print(dataframe.dtypes)

    def extract_latitude_helper(point_str):
        result = re.findall('\\(.*?\\)', point_str)
        print(result)
        return result

    format_point = 'POINT ({} {})'

    def extractor(point_str):
        parsed = parse(format_point, point_str)
        return parsed[0], parsed[1]

    dataframe['Latitude'] = dataframe['LocationStr'].apply(lambda point_str: extractor(point_str)[1])
    dataframe['Longitude'] = dataframe['LocationStr'].apply(lambda point_str: extractor(point_str)[0])

    dataframe['AvailableSpaceCount'] = dataframe["ParkingSpaceCount"] - dataframe["PaidOccupancy"]
    dataframe['BlockNameWithStreetSide'] = dataframe["BlockfaceName"] + ' - ' + dataframe["SideOfStreet"]

    print(dataframe)
    print(dataframe.columns)

    df = dataframe[['BlockNameWithStreetSide', 'AvailableSpaceCount',
                    'Latitude', 'Longitude',
                    'DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute', 'ParkingTimeLimitCategory']]

    df = df[df['AvailableSpaceCount'] >= 0]

    df = df.dropna()

    save_data(df)

    label_encoder = LabelEncoder()
    for column_name in dataframe.columns:
        if dataframe[column_name].dtype == object:
            dataframe[column_name] = label_encoder.fit_transform(
                dataframe[column_name])
    # vec = DictVectorizer()

    # label_encoder = LabelEncoder()
    # for column_name in dataframe.columns:
    #     print('column name:', column_name)
    #     if dataframe[column_name].dtype == object:
    #         dataframe[column_name] = label_encoder.fit_transform(
    #             dataframe[column_name])

    check_correlation(dataframe)

    # # 'OccupancyDateTime'
    # x_features = ['BlockfaceName', 'SideOfStreet', 'ParkingSpaceCount', 'PaidParkingArea', 'ParkingTimeLimitCategory',
    #               'DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute', 'DateTimeSecond']
    # y_feature = 'PaidOccupancy'
    #
    # linear_regression(dataframe, x_features, y_feature)
    # k_neighbors_regression(dataframe, x_features, y_feature)
    # random_forest_regression(dataframe, x_features, y_feature)

    return dataframe


def handle(df):
    x_features = ['BlockNameWithStreetSide', 'SourceElementKey', 'ParkingSpaceCount', 'Latitude', 'Longitude',
                  'DateTimeWeekDay', 'DateTimeHour', 'DateTimeMinute', 'ParkingTimeLimitCategory']
    y_feature = 'PaidOccupancy'

    label_encoder = LabelEncoder()
    for column_name in df.columns:
        # if df[column_name].dtype == object:
        if column_name in ['BlockNameWithStreetSide']:
            df[column_name] = label_encoder.fit_transform(
                df[column_name])

    linear_regression(df, x_features, y_feature)
    k_neighbors_regression(df, x_features, y_feature)
    random_forest_regression(df, x_features, y_feature)


def main():
    start_time = time()

    dataframe = read_data()

    process_available(dataframe)


    # df = read_clean_data()
    #
    # handle(df)


    print("--- %s seconds ---" % (time() - start_time))


if __name__ == "__main__":
    main()
