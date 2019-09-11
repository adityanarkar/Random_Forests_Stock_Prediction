import collections
import json
from data_prep import data_preparation as dp
from RandomForest import RF_with_predictions as rf
import ZeroHour.zerohour as zh
from functools import reduce
import os
import data_prep.data_collection as dc

# window_size = 100
future_day_start = 10
future_day_stop = 110
estimator_end = 50
depth = 50
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


def testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes, no_of_features):
    n_estimators = range(10, estimator_end, 10)
    max_depth = range(10, depth, 10)
    combinations = []

    results = {}
    final = []

    for i in n_estimators:
        for j in max_depth:
            model_score = rf.random_forest_classifier(data_for_algos, i, j, no_of_features)
            predictions = model_score[0].predict(data_to_predict_for_algos)
            our_test_score = collections.Counter(
                predictions[0:future_day] * test_classes[0:future_day]).get(1)
            our_test_score = 0 if our_test_score is None else our_test_score
            tuple_to_save = {"estimators": i, "max_depth": j, "model_score": model_score[1],
                             "future_days": future_day,
                             "our_test_score": our_test_score}
            combinations.append(tuple_to_save)

    top = reduce(lambda acc, x: selectTop(acc, x), combinations, {"our_test_score": 0})
    results["stock"] = STOCK
    results["model"] = top
    print(results)
    return results


def testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes):
    model = zh.zeroHour()
    model.fit(data_for_algos)
    print(model.prediction_class)
    predictions = model.predict(data_to_predict_for_algos)
    our_test_score = collections.Counter(
        predictions[0:future_day] * test_classes[0:future_day]).get(1)
    our_test_score = 0 if our_test_score is None else our_test_score
    result = {"STOCK: ": STOCK, "future_day": future_day, "our_test_score": our_test_score}
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


def runExperiment(tickrs):
    for STOCK_FILE in os.listdir("data/"):
        zhResults = []
        rfResults = []
        STOCK = STOCK_FILE.split(".csv")[0]
        if STOCK in tickrs:
            print(f"*** Started computations for {STOCK} ***")

            for future_day in range(future_day_start, future_day_stop, 10):
                try:
                    data_for_algos, data_to_predict_for_algos, test_classes = get_prepared_data(STOCK_FILE, future_day)
                except:
                    continue
                for no_of_features in range(17, 22, 1):
                    print(f"Predicting for future days: {future_day} No of features: {no_of_features}")
                    try:
                        finalRF = testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
                                                    test_classes, no_of_features)
                        rfResults.append(finalRF)
                    except:
                        rfResults.append({"our_test_score": 0, "cause": "error"})
                        continue

                    try:
                        finalZH = testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
                                               test_classes)
                        zhResults.append(finalZH)
                    except:
                        zhResults.append({"our_test_score": 0, "cause": "error"})

                    rf_path = f"{common_path}-{no_of_features}-MD_E_{depth}-302_TICKS-FD_{future_day}/RF/"
                    zh_path = f"{common_path}-{no_of_features}-MD_E_{depth}_302_TICKS-FD_{future_day}/ZH/"

                    if not os.path.exists(rf_path):
                        os.makedirs(rf_path)
                    with open(f"{rf_path}/{STOCK}.JSON", 'w') as f:
                        f.write(json.dumps(rfResults))

                    if not os.path.exists(zh_path):
                        os.makedirs(zh_path)
                    with open(f"{zh_path}/{STOCK}.JSON", 'w') as f:
                        f.write(json.dumps(zhResults))

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
