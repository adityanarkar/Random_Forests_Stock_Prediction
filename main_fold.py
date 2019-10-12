import json
import os
from functools import reduce
import definitions
import numpy as np
import sklearn.dummy as dummy
import KNN.knn as knn

import data_prep.data_collection as dc
from RandomForest import rf_fold as rf
from SVM import svm
from data_prep import data_preparation as dp


# future_day_start = 10
# future_day_stop = 110
# estimator_end = 60
# depth = 60
# initial_no_of_features = 22
# max_features = 23
# feature_window_size = 50
# discretize = True


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


def result_in_csv(STOCK, Algo, Estimator=0, Depth=0, Distance_function='0', No_of_features=0, Model_Score=0,
                  Future_day=0, C=-1,
                  Our_test_score=-1):
    return f"{STOCK},{Algo},{Estimator},{Depth},{Distance_function},{No_of_features},{Model_Score},{Future_day},{C},{Our_test_score}\n "


def testRandomForests(STOCK, future_day, data_for_algos, estimator_start, estimator_stop,
                      depth_start, depth_stop,
                      initial_no_of_features, max_features):
    n_estimators = range(estimator_start, estimator_stop, 10)
    max_depth = range(depth_start, depth_stop, 10)
    top = get_initial_top_rf()
    for no_of_features in [10, 15, 20, 25, 28]:
        print(f"{STOCK} {no_of_features}")
        for i in n_estimators:
            for j in max_depth:
                try:
                    selector, score = rf.random_forest_classifier(data_for_algos, i, j, no_of_features, future_day=future_day)
                except:
                    continue
                if score > top['model_score']:
                    top = get_top_rf(estimators=i, max_depth=j, model_score=score, future_day=future_day,
                                     no_of_features=no_of_features)
    result = get_top_rf_result_csv_format(STOCK, top)
    print(f"final Result RF: {result}")
    return result


def get_initial_top_rf():
    return {"estimators": -1, 'max_depth': -1, 'model_score': -1, 'future_day': -1, 'no_of_features': -1}


def get_top_rf(estimators, max_depth, model_score, future_day, no_of_features):
    return {'estimators': estimators, 'max_depth': max_depth, 'model_score': model_score, 'future_day': future_day,
            'no_of_features': no_of_features}


def get_top_rf_result_csv_format(STOCK, top):
    top_estimator = top["estimators"]
    top_depth = top['max_depth']
    top_model_score = top['model_score']
    top_future_day = top['future_day']
    top_no_of_features = top['no_of_features']
    result = result_in_csv(STOCK, 'RF', top_estimator, top_depth, No_of_features=top_no_of_features,
                           Model_Score=top_model_score, Future_day=top_future_day)
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
    data_for_algos = df.to_numpy()
    return data_for_algos, actual_data_to_predict


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
        f.write("Stock,Algorithm,Estimators,Depth,Distance_function,No_of_features,Model_Score,Future_day,"
                "C,Our_test_score\n")


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
            dict_to_save = {'metric': metric, 'score': score, 'neighbors': n_neighbors,
                            'our_test_score': our_test_score}
            combinations.append(dict_to_save)
    top = reduce(lambda acc, x: selectTop(acc, x), combinations, {"our_test_score": -1, 'metric': 'error', 'score': -1})
    print(top)
    return result_in_csv(STOCK, 'KNN', Distance_function=top["metric"], Model_Score=top['score'], Future_day=future_day,
                         Our_test_score=top['our_test_score'])


def testSVM(STOCK, data_for_algos, data_to_predict_for_algos, test_classes, future_day, initial_no_of_features,
            max_features, C):
    combinations = []
    for no_of_features in [10, 15, 20, 25, 28]:
        print(f"{STOCK} {future_day}")
        for c_val in C:
            try:
                clf, score = svm.svm_classifier(data_for_algos, no_of_features, c_val)
            except:
                continue
            # return result_in_csv(STOCK, 'SVM', C=C, Our_test_score=-1)
            data_to_predict_for_algos = svm.scale_data(data_to_predict_for_algos)
            predictions = clf.predict(data_to_predict_for_algos)
            our_test_score = get_test_score(predictions, test_classes)
            dict_to_save = {'C': c_val, 'score': score, 'future_day': future_day, 'no_of_features': no_of_features,
                            'our_test_score': our_test_score}
            combinations.append(dict_to_save)
    top = reduce(lambda acc, x: selectTop(acc, x), combinations, get_svm_initial_top(future_day))
    result = get_svm_top_result_csv(STOCK, top)
    print(result)
    return result


def get_svm_top_result_csv(STOCK, top):
    return result_in_csv(STOCK, 'SVM', No_of_features=top['no_of_features'], Distance_function="Linear", C=top['C'],
                         Model_Score=top['score'], Future_day=top['future_day'],
                         Our_test_score=top['our_test_score'])


def get_svm_initial_top(future_day):
    return {'C': -1, 'score': -1, 'future_day': future_day, 'no_of_features': -1,
            'our_test_score': -1}


def get_test_score(predictions, test_classes):
    our_test_score = sum([1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
    return 0 if our_test_score is None else our_test_score


def runExperiment(lock, STOCK_FILE, RESULT_FILE, algos, future_day_start, future_day_stop, estimator_start,
                  estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, feature_window_size,
                  discretize, C):
    STOCK = STOCK_FILE.split(".csv")[0]
    print(STOCK)

    result = ""
    for future_day in range(future_day_start, future_day_stop, 10):
        try:
            data_for_algos, actual_data_to_predict = get_prepared_data(
                STOCK_FILE, future_day, feature_window_size, discretize)
        except:
            continue

        if 'KNN' in algos:
            result += testKNN(STOCK, data_for_algos, future_day)

        if 'RF' in algos:
            print(f"Predicting {STOCK} for future days: {future_day} using RF")
            result += testRandomForests(STOCK, future_day, data_for_algos, estimator_start,
                                        estimator_stop,
                                        depth_start, depth_stop, initial_no_of_features, max_features)

        if 'SVM' in algos:
            result += testSVM(STOCK, data_for_algos, future_day, initial_no_of_features, max_features, C)

        if 'ZR' in algos:
            result += testZeroHour(STOCK, future_day, data_for_algos)

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