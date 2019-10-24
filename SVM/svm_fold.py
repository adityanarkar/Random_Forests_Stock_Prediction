import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Prediction import score
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import RFE
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from data_prep import k_splits


@ignore_warnings(category=ConvergenceWarning)
def svm_classifier(data, features_to_select, C, future_day, kernel, degree):
    scores = []
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    train_indices, test_indices = k_splits.get_max_k_splits(X, k=10, size_of_each_split=future_day)
    clf = 0
    predict_score = -1
    for train_index, test_index in zip(train_indices, test_indices):
        X_train, y_train, X_test, y_test = k_splits.get_train_test_set(X, y, train_index, test_index)
        clf = LinearSVC(random_state=0, tol=1e-5, C=C, max_iter=10000)
        clf = RFE(clf, features_to_select, step=1)
        X_new = clf.fit_transform(X_train, y_train)
        X_test_new = clf.transform(X_test)
        clf = SVC(C=C, kernel=kernel, gamma='scale', degree=degree)
        clf.fit(X_new, y_train)
        predict_score = score.get_score(clf, X_test_new, y_test)
        scores.append(predict_score)
    mean_score = np.mean(scores[:-1])
    mean_score = -1 if np.isnan(mean_score) else mean_score
    return clf, mean_score, predict_score
