import numpy as np
from sklearn import dummy
from Prediction import score
from data_prep import k_splits


def zr(data, future_day):
    scores = []
    X = np.asarray(list(map(lambda row: row[:-1], data)))
    y = np.asarray(list(map(lambda row: row[-1], data)))

    train_indices, test_indices = k_splits.get_max_k_splits(X, k=10, size_of_each_split=future_day)
    model = dummy.DummyClassifier(strategy="most_frequent")
    predict_score = -1
    for train_index, test_index in zip(train_indices, test_indices):
        X_train, y_train, X_test, y_test = k_splits.get_train_test_set(X, y, train_index, test_index)
        model.fit(X_train, y_train)
        predict_score = score.get_score(model, X_test, y_test)
        scores.append(predict_score)
    mean_score = np.mean(scores[:-1])
    mean_score = -1 if np.isnan(mean_score) else mean_score
    return model, mean_score, predict_score
