import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_classifier(dict_of_params: dict):
    data = dict_of_params['data']
    neighbors = dict_of_params['neighbors']
    no_of_features = dict_of_params['no_of_features']
    metric = dict_of_params['metric']
    actual_data = dict_of_params['actual_data']
    test_classes = dict_of_params['test_classes']
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = X
    y_train = y
    X_test = actual_data
    y_test = test_classes
    clf = KNeighborsClassifier(n_neighbors=neighbors, metric=metric)
    selector = SelectKBest(mutual_info_classif, k=no_of_features)
    X_new = selector.fit_transform(X_train, y_train)
    clf.fit(X_new, y_train)
    X_test_new = selector.transform(X_test)
    score = clf.score(X_test_new, y_test)
    dict_of_params.update({'our_test_score': score, 'score': -1, 'selector': selector, 'clf': clf})
    dict_of_params.pop('data')
    # print(score)
    return dict_of_params
