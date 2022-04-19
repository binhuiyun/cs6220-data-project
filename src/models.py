from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def evaluate_accuracy(model, X_test, y_test):
    y_train_pred = model.predict(X_test)
    print('Model: {0} R-Squared: {1}'.format(
        model.__class__.__name__, r2_score(y_test, y_train_pred)))


def linear_regression(consolidated_df):
    label_encoder = LabelEncoder()

    for column_name in consolidated_df.columns:
        if consolidated_df[column_name].dtype == object:
            consolidated_df[column_name] = label_encoder.fit_transform(
                consolidated_df[column_name])

    X = consolidated_df.drop(['A_MEAN'], axis=1)
    y = consolidated_df['A_MEAN']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    evaluate_accuracy(model, X_test, y_test)


def random_forest_regression(consolidated_df):
    label_encoder = LabelEncoder()

    for column_name in consolidated_df.columns:
        if consolidated_df[column_name].dtype == object:
            consolidated_df[column_name] = label_encoder.fit_transform(
                consolidated_df[column_name])

    X = consolidated_df.drop(['A_MEAN'], axis=1)
    y = consolidated_df['A_MEAN']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    model = RandomForestRegressor(max_depth=16, random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)

    evaluate_accuracy(model, X_test, y_test)
