from data_prep import k_splits
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Prediction import score
from sklearn.feature_selection import RFE


def random_forest_classifier(data, n_estimators, max_depth, no_of_features, future_day):
    sum_score = 0
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))
    # training and testing
    train_indices, test_indices = k_splits.get_max_k_splits(X, k=10, size_of_each_split=future_day)

    for train_index, test_index in train_indices, test_indices:
        X_train = X[train_index[0]:train_index[1]]
        y_train = y[train_index[0]:train_index[1]]
        X_test = X[test_index[0]:test_index[1]]
        y_test = y[test_index[0]:test_index[1]]

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        selector = RFE(clf, no_of_features, step=1)
        selector = selector.fit(X_train, y_train)
        predict_score = score.get_score(selector, X_test, y_test)
        sum_score += predict_score
    return selector, (sum_score / len(train_indices))
