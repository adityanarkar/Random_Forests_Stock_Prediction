import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.testing import ignore_warnings


@ignore_warnings(category=ConvergenceWarning)
def svm_classifier(dict_of_params):
    data = dict_of_params['data']
    features_to_select = dict_of_params['no_of_features']
    C = dict_of_params['C']
    degree = dict_of_params['degree']
    kernel = dict_of_params['kernel']
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
    dict_of_params.update(
        {'clf': clf, 'score': score, 'selector': selector})
    dict_of_params.pop('data')
    return dict_of_params


def feature_selection(features_to_select, C, X_train, y_train, X_test):
    clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=10000)
    clf = RFE(clf, features_to_select, step=1)
    X_train = clf.fit_transform(X_train, y_train)
    X_test = clf.transform(X_test)
    return X_train, X_test, clf


def get_our_test_score(clf, selector, data_predict, test_classes):
    if selector != 0:
        data_predict = selector.transform(data_predict)
    predictions = clf.predict(data_predict)
    our_test_score = sum(
        [1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
    our_test_score = 0 if our_test_score is None else our_test_score
    return our_test_score
