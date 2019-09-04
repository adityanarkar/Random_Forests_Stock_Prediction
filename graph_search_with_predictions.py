import collections

import json
from data_prep import data_preparation as dp
from RandomForest import RF_with_predictions as rf
from functools import reduce
import os


def selectTop3(top3, x):
    if x["our_test_score"] > top3[-1]["our_test_score"]:
        top3[-1] = x
    return sorted(top3, key=lambda y: y["our_test_score"], reverse=True)


def getInitial():
    initial = []
    for i in range(3):
        initial.append({"our_test_score": 0})
    return initial


for STOCK_FILE in os.listdir("data/"):
    STOCK = STOCK_FILE.split(".csv")[0]
    print(f"*** Started computations for {STOCK}***")

    df, actual_data_to_predict = dp.data_preparation(f"data/{STOCK_FILE}", 100).data_frame_with_features()
    complete_data = df.to_numpy()
    data_for_algos, data_to_predict_for_algos, test_classes = complete_data[:-100], complete_data[-100:,
                                                                                    :-1], complete_data[-100:, -1]

    n_estimators = range(10, 210, 10)
    max_depth = range(10, 210, 10)
    combinations = []

    results = {}

    for future_day in range(10, 110, 10):
        for i in n_estimators:
            for j in max_depth:
                model_score = rf.random_forest_classifier(data_for_algos, i, j)
                predictions = model_score[0].predict(data_to_predict_for_algos)
                our_test_score = collections.Counter(
                    predictions[0:future_day] * test_classes[0:future_day]).get(1)
                print(predictions[0:future_day])
                print(test_classes[0:future_day])
                tuple_to_save = {"estimators": i, "max_depth": j, "model_score": model_score[1],
                                 "our_test_score": our_test_score}
                print(tuple_to_save)
                combinations.append(tuple_to_save)

        top3 = reduce(lambda acc, x: selectTop3(acc, x), combinations, getInitial())
        results["stock"] = STOCK
        results[f"{future_day}"] = top3

    with open(f"Results/{STOCK}-results.JSON", 'w') as f:
        f.write(json.dumps(results))
