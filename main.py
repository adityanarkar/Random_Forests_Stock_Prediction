import collections
import json

import numpy as np

from data_prep import data_preparation as dp
from RandomForest import RF_with_predictions as rf
import sklearn.dummy as dummy
import ZeroHour.zerohour as zh
from functools import reduce
import os
import data_prep.data_collection as dc

future_day_start = 10
future_day_stop = 110
estimator_end = 100
depth = 100
initial_no_of_features = 10
max_features = 23
common_path = "Results/Selection"


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


def get_top_rf_result_csv_format(STOCK, top, no_of_features):
    top_estimator = top["estimators"]
    top_depth = top['max_depth']
    top_model_score = top['model_score']
    top_future_day = top['future_day']
    top_our_test_score = top['our_test_score']
    result = f"{STOCK},RF,{top_estimator},{top_depth},{no_of_features},{top_model_score},{top_future_day},{top_our_test_score}\n"
    return result


def testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes, no_of_features):
    n_estimators = range(10, estimator_end, 10)
    max_depth = range(10, depth, 10)
    combinations = []

    for i in n_estimators:
        for j in max_depth:
            model_score = rf.random_forest_classifier(data_for_algos, i, j, no_of_features)
            predictions = model_score[0].predict(data_to_predict_for_algos)
            our_test_score = collections.Counter(
                predictions * test_classes).get(1)
            our_test_score = 0 if our_test_score is None else our_test_score
            tuple_to_save = {"estimators": i, "max_depth": j, "model_score": model_score[1],
                             "future_day": future_day,
                             "our_test_score": our_test_score}
            combinations.append(tuple_to_save)

    top = reduce(lambda acc, x: selectTop(acc, x), combinations, {"our_test_score": 0})
    result = get_top_rf_result_csv_format(STOCK, top, no_of_features)
    print(f"final Result RF: {result}")
    return result


def testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes):
    model = dummy.DummyClassifier(strategy="most_frequent")
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    model.fit(X, y)
    predictions = model.predict(data_to_predict_for_algos)
    our_test_score = collections.Counter(
        predictions[0:future_day] * test_classes[0:future_day]).get(1)
    our_test_score = 0 if our_test_score is None else our_test_score

    result = f"{STOCK},ZR,0,0,0,0,{future_day},{our_test_score}\n"
    print(result)
    return result


def create_dir_and_store_result(dir_to_create, result_path, result):
    if not os.path.isdir(dir_to_create):
        os.mkdir(dir_to_create)
        with open(result_path, 'w') as f:
            f.write(json.dumps(result))


def get_prepared_data(STOCK_FILE, window_size):
    df, actual_data_to_predict = dp.data_preparation(f"data/{STOCK_FILE}",
                                                     window_size=window_size).data_frame_with_features()
    complete_data = df.to_numpy()

    data_for_algos, data_to_predict_for_algos, test_classes = complete_data[:-window_size], complete_data[
                                                                                            -window_size:,
                                                                                            :-1], complete_data[
                                                                                                  -window_size:,
                                                                                                  -1]
    return data_for_algos, data_to_predict_for_algos, test_classes


def write_result_to_file(RESULT_FILE, result):
    with open(RESULT_FILE, 'a') as f:
        f.write(result)


def add_headers(RESULT_FILE):
    with open(RESULT_FILE, 'w') as f:
        f.write("Stock,Algorithm,Estimators,Depth,No_of_features,Model_Score,Future_day,Our_test_score\n")

def runExperiment(tickrs):
    RESULT_FILE = "Results/result.csv"
    add_headers(RESULT_FILE)
    for STOCK_FILE in os.listdir("data/"):
        STOCK = STOCK_FILE.split(".csv")[0]
        if STOCK in tickrs:
            print(f"*** Started computations for {STOCK} ***")

            for future_day in range(future_day_start, future_day_stop, 10):
                result = ""
                # try:
                data_for_algos, data_to_predict_for_algos, test_classes = get_prepared_data(STOCK_FILE, future_day)
                # except:
                #     continue
                for no_of_features in range(initial_no_of_features, max_features, 1):
                    print(f"Predicting for future days: {future_day} No of features: {no_of_features}")
                    try:
                        result += testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
                                                    test_classes, no_of_features)
                    except:
                        result += f"{STOCK},RF,0,0,0,0,{future_day},0\n"
                        continue

                try:
                    result += testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
                                           test_classes)
                except:
                    result += f"{STOCK},ZR,0,0,0,0,{future_day},0\n"

                try:
                    write_result_to_file(RESULT_FILE, result)
                except:
                    print("Error while writing to a file.")
                    continue

        else:
            pass


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


tickr_file = "data/TICKR.txt"
# collect_data(300, tickr_file)
runExperiment(get_requested_tickrs(tickr_file))
