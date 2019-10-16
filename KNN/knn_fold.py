import multiprocessing as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from Prediction import score
from data_prep import k_splits


def min_max_transform(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def knn_classifier(data, metric: str, neighbors: int, future_day, no_of_features: int):
    scores = []
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    X = min_max_transform(X)
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    train_indices, test_indices = k_splits.get_max_k_splits(X, k=10, size_of_each_split=future_day)
    clf = 0
    predict_score = -1
    print("Started training...")
    for train_index, test_index in zip(train_indices, test_indices):
        X_train, y_train, X_test, y_test = k_splits.get_train_test_set(X, y, train_index, test_index)
        clf = KNeighborsClassifier(n_neighbors=neighbors, weights='distance', metric=metric)
        X_train = SelectKBest(chi2, k=no_of_features).fit_transform(X_train, y)
        clf.fit(X_train, y_train)
        predict_score = score.get_score(clf, X_test, y_test)
        scores.append(predict_score)
    print("Finished training...")
    return clf, np.mean(scores), predict_score
