import json
import os
from functools import reduce
import definitions
import numpy as np
import sklearn.dummy as dummy
import KNN.knn as knn

import data_prep.data_collection as dc
from RandomForest import RF_with_predictions as rf
from SVM import svm
from data_prep import data_preparation as dp

future_day_start = 10
future_day_stop = 110
estimator_end = 60
depth = 60
initial_no_of_features = 22
max_features = 23
feature_window_size = 50
discretize = True


def selectTop3(top3, x):
    if x["our_test_score"] > top3[-1]["our_test_score"]:
        top3[-1] = x
    return sorted(top3, key=lambda y: y["our_test_score"], reverse=True)


def selectTop(top, x):
    print("x ", x, "top ", top)
    if x["our_test_score"] > top["our_test_score"]:
        return x
    return top


def getInitial():
    initial = []
    for i in range(3):
        initial.append({"our_test_score": 0})
    return initial


def get_top_rf_result_csv_format(STOCK, top, shuffle):
    top_estimator = top["estimators"]
    top_depth = top['max_depth']
    top_model_score = top['model_score']
    top_future_day = top['future_day']
    top_our_test_score = top['our_test_score']
    result = result_in_csv(STOCK, 'RF', top_estimator, top_depth, No_of_features=shuffle,
                           Model_Score=top_model_score, Future_day=top_future_day, Our_test_score=top_our_test_score)
    return result


def result_in_csv(STOCK, Algo, Estimator=0, Depth=0, Distance_function=0, No_of_features=False, Model_Score=0, Future_day=0,
                  Our_test_score=0):
    return f"{STOCK},{Algo},{Estimator},{Depth},{Distance_function},{No_of_features},{Model_Score},{Future_day},{Our_test_score}\n "


def testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes, shuffle,
                      actual_data_to_predict):
    n_estimators = range(10, estimator_end, 10)
    max_depth = range(10, depth, 10)
    combinations = []

    for i in n_estimators:
        for j in max_depth:
            try:
                model_score = rf.random_forest_classifier(data_for_algos, i, j, shuffle)
            except:
                continue
            predictions = model_score[0].predict(data_to_predict_for_algos)
            our_test_score = get_test_score(predictions, test_classes)
            # print(our_test_score)
            tuple_to_save = {"estimators": i, "max_depth": j, "model_score": model_score[1],
                             "future_day": future_day,
                             "our_test_score": our_test_score}
            combinations.append(tuple_to_save)

    top = reduce(lambda acc, x: selectTop(acc, x), combinations,
                 {"our_test_score": -1, 'max_depth': -1, 'model_score': -1, 'future_day': -1, 'estimators': -1})
    result = get_top_rf_result_csv_format(STOCK, top, shuffle)
    print(f"final Result RF: {result}")
    return result


def testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes):
    try:
        model = dummy.DummyClassifier(strategy="most_frequent")
        X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
        y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
        model.fit(X, y)
        predictions = model.predict(data_to_predict_for_algos)
        our_test_score = get_test_score(predictions, test_classes)
        result = result_in_csv(STOCK, 'ZR', Future_day=future_day, Our_test_score=our_test_score)
    except:
        result = result_in_csv(STOCK, 'ZR', Future_day=future_day, Our_test_score=-1)
    print(result)
    return result


def create_dir_and_store_result(dir_to_create, result_path, result):
    if not os.path.isdir(dir_to_create):
        os.mkdir(dir_to_create)
        with open(result_path, 'w') as f:
            f.write(json.dumps(result))


def get_prepared_data(STOCK_FILE, window_size, feature_window_size, discretize):
    df, actual_data_to_predict = dp.data_preparation(os.path.join(f"{definitions.ROOT_DIR}", f"data/{STOCK_FILE}"),
                                                     window_size=window_size,
                                                     feature_window_size=feature_window_size,
                                                     discretize=discretize).data_frame_with_features()
    df.drop(columns=['open', 'high', 'low', 'close'], inplace=True)
    complete_data = df.to_numpy()

    data_for_algos, data_to_predict_for_algos, test_classes = complete_data[:-window_size], complete_data[
                                                                                            -window_size:,
                                                                                            :-1], complete_data[
                                                                                                  -window_size:,
                                                                                                  -1]
    return data_for_algos, data_to_predict_for_algos, test_classes, actual_data_to_predict


def write_result_to_file(lock, RESULT_FILE, result):
    lock.acquire()
    try:
        open(RESULT_FILE, 'a').write(result)
    except:
        print("Error while writing to a file.")
    lock.release()


def make_missing_dirs(path):
    head, tail = os.path.split(path)
    if not os.path.exists(head):
        os.makedirs(head)


def add_headers(RESULT_FILE):
    with open(RESULT_FILE, 'w') as f:
        f.write("Stock,Algorithm,Estimators,Depth,Distance_function,Shuffle,Model_Score,Future_day,"
                "Our_test_score\n")


def testKNN(STOCK, data_for_algos, data_to_predict_for_algos, test_classes, future_day):
    combinations = []
    algos = ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']
    for n_neighbors in [3, 5, 7, 9, 11]:
        for metric in algos:
            try:
                clf, score = knn.knn_classifier(data_for_algos, metric, n_neighbors)
            except:
                continue
            data_to_predict_for_algos = knn.min_max_transform(data_to_predict_for_algos)
            predictions = clf.predict(data_to_predict_for_algos)
            our_test_score = get_test_score(predictions, test_classes)
            dict_to_save = {'metric': metric, 'score': score, 'neighbors': n_neighbors, 'our_test_score': our_test_score}
            combinations.append(dict_to_save)
    top = reduce(lambda acc, x: selectTop(acc, x), combinations, {"our_test_score": -1, 'metric': 'error', 'score': -1})
    print(top)
    return result_in_csv(STOCK, 'KNN', Distance_function=top["metric"], Model_Score=top['score'], Future_day=future_day,
                         Our_test_score=top['our_test_score'])


def get_test_score(predictions, test_classes):
    our_test_score = sum([1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
    return 0 if our_test_score is None else our_test_score


def runExperiment(lock, STOCK_FILE, RESULT_FILE, algos):
    STOCK = STOCK_FILE.split(".csv")[0]
    print(STOCK)

    result = ""
    for future_day in range(future_day_start, future_day_stop, 10):
        try:
            data_for_algos, data_to_predict_for_algos, test_classes, actual_data_to_predict = get_prepared_data(
                STOCK_FILE, future_day, feature_window_size, discretize)
        except:
            continue

        if 'KNN' in algos:
            result = testKNN(STOCK, data_for_algos, data_to_predict_for_algos, test_classes, future_day)

        if 'RF' in algos:
            # for no_of_features in range(initial_no_of_features, max_features, 1):
            for shuffle in [True, False]:
                # print(f"Predicting for future days: {future_day} No of features: {no_of_features}")
                print(f"Predicting for future days: {future_day} Shuffle: {shuffle}")
                result += testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
                                            test_classes, shuffle, actual_data_to_predict)

        if 'ZR' in algos:
            result += testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
                               test_classes)

        write_result_to_file(lock, RESULT_FILE, result)


def collect_data(no_of_symbols: int, filepath: str):
    dc.sample_data(no_of_symbols, filepath)
    dc.collect_data(filepath)


def get_requested_tickrs(filepath):
    result = []
    with open(filepath) as f:
        for line in f.readlines():
            if not line.startswith("#"):
                result.append(line.replace("\n", ""))
    return result
