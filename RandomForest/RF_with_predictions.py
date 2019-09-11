import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


def random_forest_classifier(data, n_estimators, max_depth):
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    selector = RFE(clf, 14, step=1)
    selector = selector.fit(X_train, y_train)
    score = selector.score(X_test, y_test)
    print(selector.support_)

    return selector, score

