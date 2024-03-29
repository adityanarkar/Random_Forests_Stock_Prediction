import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


def random_forest_classifier(data, n_estimators, max_depth, no_of_features):
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf = RFE(clf, no_of_features, step=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    # print(selector.support_)

    return clf, score

