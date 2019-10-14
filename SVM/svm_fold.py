import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Prediction import score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE

from data_prep import k_splits


def scale_data(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler.transform(X)



def svm_classifier(data, features_to_select, C, future_day):
    scores = []
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    X = scale_data(X)
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    train_indices, test_indices = k_splits.get_max_k_splits(X, k=10, size_of_each_split=future_day)
    clf = 0
    for train_index, test_index in zip(train_indices, test_indices):
        X_train, y_train, X_test, y_test = k_splits.get_train_test_set(X, y, train_index, test_index)
        clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=10000)
        clf = RFE(clf, features_to_select, step=1)
        clf.fit(X_train, y_train)
        predict_score = score.get_score(clf, X_test, y_test)
        scores.append(predict_score)
        print(predict_score)
    return clf, np.mean(scores)
