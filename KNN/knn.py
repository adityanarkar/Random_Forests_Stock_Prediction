import multiprocessing as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def min_max_transform(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def knn_classifier(data, metric: str, neighbors: int):
    n_jobs = mp.cpu_count() - 2

    X = np.asarray(list(map(lambda row: row[:-1], data)))
    X = min_max_transform(X)
    y = np.asarray(list(map(lambda row: row[-1], data)))

    # training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=n_jobs, weights='distance', metric=metric)
    print("Started training...")
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Finished training...")
    return clf, score
