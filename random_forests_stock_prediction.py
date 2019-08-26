import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import data_preparation as dp


def random_forest_classifier(n_estimators, max_depth, random_state):
    df, data_to_predict = dp.data_preparation('data/TITAN.NS.csv', 10).data_frame_with_features()

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
