from time import time

from interpret_read import interpret_data

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_extraction import DictVectorizer

from joblib import dump

from pandas import to_datetime


def process(dataframe):
    # filter by block face name
    block_face_name = 'JOHN ST BETWEEN WESTLAKE AVE N AND TERRY AVE N'
    dataframe = dataframe[dataframe['BlockfaceName'] == block_face_name]
    # return block_dataframe

    print(dataframe.head(10))

    # dataframe.to_csv('terry.csv', encoding='utf-8', index=False)
    # dataframe = dataframe.dropna()

    dataframe['FormedOccupancyDateTime'] = to_datetime(
        dataframe['OccupancyDateTime'], format='%m/%d/%Y %I:%M:%S %p')
    dataframe['DateTimeWeekDay'] = dataframe['FormedOccupancyDateTime'].dt.dayofweek
    dataframe['DateTimeHour'] = dataframe['FormedOccupancyDateTime'].dt.hour
    dataframe['DateTimeMinute'] = dataframe['FormedOccupancyDateTime'].dt.minute
    dataframe['DateTimeSecond'] = dataframe['FormedOccupancyDateTime'].dt.second

    print(dataframe)
    print(dataframe.columns)

    # print(dataframe['DateTimeDatum'])

    vec = DictVectorizer()

    # process string values
    label_encoder = LabelEncoder()
    for column_name in dataframe.columns:
        if dataframe[column_name].dtype == object:
            dataframe[column_name] = label_encoder.fit_transform(
                dataframe[column_name])

    X = dataframe.drop(
        ['FormedOccupancyDateTime', 'PaidOccupancy', 'ParkingTimeLimitCategory', 'PaidParkingRate'], axis=1)
    y = dataframe['PaidOccupancy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)

    model = RandomForestRegressor(max_depth=16, random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)

    dump(model, 'random-forest.joblib')

    evaluate_accuracy(model, X_test, y_test)


def evaluate_accuracy(model, X_test, y_test):
    y_train_pred = model.predict(X_test)
    print('{0} R-Squared: {1}'.format(
        model.__class__.__name__, r2_score(y_test, y_train_pred)))


def main():
    start_time = time()

    dataframe = interpret_data()

    # print(dataframe.head(10))
    process(dataframe)

    print("--- %s seconds ---" % (time() - start_time))


if __name__ == "__main__":
    main()
