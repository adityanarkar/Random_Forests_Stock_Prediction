import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_to_predict = np.nan
y = []


def create_label(row):
    row['target'] = 1 if row['Adj Close'] < row['shifted_value'] else -1
    return row


def get_fresh_data_for_prediction(df: pd.DataFrame):
    result = df.where(df['shifted_value'].isna())
    result.dropna(thresh=1, inplace=True)
    return result


df = pd.read_csv('data/TITAN.NS.csv')
df.drop(columns=["Date"], inplace=True)
df.dropna(inplace=True)

# create label and save rows with labels for prediction task
df['shifted_value'] = df['Adj Close'].shift(-10)
data_to_predict = get_fresh_data_for_prediction(df)
df = df.apply(lambda x: create_label(x), axis=1)
df.dropna(inplace=True)

# convert dataframe to numpy array
data = df.to_numpy()
X = np.asarray(list(map(lambda row: row[:-1], data)))
y = np.asarray(list(map(lambda row: row[-1], data)))
print(X)

# training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


