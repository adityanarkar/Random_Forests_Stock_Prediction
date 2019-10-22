import numpy as np
from sklearn import dummy
from sklearn.model_selection import train_test_split


def zr(data_for_algos):
    model = dummy.DummyClassifier(strategy="most_frequent")
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)
