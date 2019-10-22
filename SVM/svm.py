import numpy as np
from sklearn import svm
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.utils.testing import ignore_warnings


def scale_data(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler.transform(X)


@ignore_warnings(category=ConvergenceWarning)
def svm_classifier(data, features_to_select, C):
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    X = scale_data(X)
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=10000)
    clf = RFE(clf, 25, step=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    return clf, score
