import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.testing import ignore_warnings


@ignore_warnings(category=ConvergenceWarning)
def svm_classifier(data, features_to_select, C, kernel, degree):
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    selector = 0
    if features_to_select != -1:
        X_train, X_test, selector = feature_selection(features_to_select, C, X_train, y_train, X_test)
    clf = SVC(C=C, kernel=kernel, gamma='scale', degree=degree)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    return clf, score, selector


def feature_selection(features_to_select, C, X_train, y_train, X_test):
    clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=10000)
    clf = RFE(clf, features_to_select, step=1)
    X_train = clf.fit_transform(X_train, y_train)
    X_test = clf.transform(X_test)
    return X_train, X_test, clf
