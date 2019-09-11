import collections
import json
from data_prep import data_preparation as dp
from RandomForest import RF_with_predictions as rf
import ZeroHour.zerohour as zh
from functools import reduce
import os
import data_prep.data_collection as dc

window_size = 100
future_day_start = 100
future_day_stop = 110


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


def testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes):
    n_estimators = range(10, 40, 10)
    max_depth = range(10, 40, 10)
    combinations = []

    results = {}
    final = []

    for i in n_estimators:
        for j in max_depth:
            model_score = rf.random_forest_classifier(data_for_algos, i, j)
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


def runExperiment(tickrs):
    for STOCK_FILE in os.listdir("data/"):
        zhResults = []
        rfResults = []
        STOCK = STOCK_FILE.split(".csv")[0]
        if STOCK in tickrs:
            print(f"*** Started computations for {STOCK} ***")
            try:
                df, actual_data_to_predict = dp.data_preparation(f"data/{STOCK_FILE}",
                                                                 window_size=window_size).data_frame_with_features()
                complete_data = df.to_numpy()

                data_for_algos, data_to_predict_for_algos, test_classes = complete_data[:-window_size], complete_data[
                                                                                                    -window_size:,
                                                                                                    :-1], complete_data[
                                                                                                          -window_size:,
                                                                                                          -1]
            except:
                continue
            for future_day in range(future_day_start, future_day_stop, 10):
                print(f"Predicting for future days: {future_day}")
                try:
                    finalRF = testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
                                                test_classes)
                    rfResults.append(finalRF)
                except:
                    rfResults.append({"our_test_score": 0, "cause": "error"})
                    continue
                try:
                    finalZH = testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes)
                    zhResults.append(finalZH)
                except:
                    zhResults.append({"our_test_score": 0, "cause": "error"})

            with open(f"Results/RF/{STOCK}.JSON", 'w') as f:
                f.write(json.dumps(rfResults))

            with open(f"Results/ZH/{STOCK}.JSON", 'w') as f:
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
