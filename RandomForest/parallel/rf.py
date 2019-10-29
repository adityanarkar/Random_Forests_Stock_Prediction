import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


def random_forest_classifier(dict_of_params: dict):
    data = dict_of_params['data']
    n_estimators = dict_of_params['estimators']
    max_depth = dict_of_params['max_depth']
    no_of_features = dict_of_params['no_of_features']
    actual_data = dict_of_params['actual_data']
    test_classes = dict_of_params['test_classes']
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))
    print(no_of_features, n_estimators, max_depth)

    # training and testing
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = X
    y_train = y
    X_test = actual_data
    y_test = test_classes
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    if no_of_features != -1:
        clf = RFE(clf, no_of_features, step=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    dict_of_params.update({'clf': clf, 'score': -1, 'our_test_score': score})
    dict_of_params.pop('data')
    return dict_of_params
