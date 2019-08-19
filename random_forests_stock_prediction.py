import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import features

data_to_predict = np.nan
y = []


def create_label(row):
    row['target'] = 1 if row['Adj Close'] < row['shifted_value'] else -1
    return row


def get_fresh_data_for_prediction(df: pd.DataFrame):
    result = df.where(df['shifted_value'].isna())
    result.dropna(thresh=1, inplace=True)
    result.drop(columns=['shifted_value'], inplace=True)
    return result


def random_forest_classifier(n_estimators, max_depth, random_state):
    window = 10
    df = pd.read_csv('data/TITAN.NS.csv')

    df.drop(columns=["Date"], inplace=True)
    df.dropna(inplace=True)
    df = features.simpleMA(df, window)
    df = features.weightedMA(df, window)
    df = features.EMA(df, window)
    df = features.momentum(df, window)
    print(df.head())

    # create label and save rows with labels for prediction task
    df['shifted_value'] = df['Adj Close'].shift(-10)
    data_to_predict = get_fresh_data_for_prediction(df)
    df = df.apply(lambda x: create_label(x), axis=1)
    df.dropna(inplace=True)
    df.drop(columns=['shifted_value'], inplace=True)

    # convert dataframe to numpy array
    data = df.to_numpy()
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    print(clf.predict(data_to_predict))
    return score


random_forest_classifier(200, 10, 1)
