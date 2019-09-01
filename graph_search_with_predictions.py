import collections

import pandas as pd
from data_prep import data_preparation as dp
import numpy as np
from RandomForest import RF_with_predictions as rf

df, actual_data_to_predict = dp.data_preparation('data/NKE.csv', 100).data_frame_with_features()
complete_data = df.to_numpy()
data_for_algos, data_to_predict_for_algos, test_classes = complete_data[:-100], complete_data[-100:,
                                                                                :-1], complete_data[-100:, -1]

n_estimators = np.arange(start=10, stop=210, step=10)
max_depth = np.arange(start=10, stop=210, step=10)
combinations = []

# model_score = rf.random_forest_classifier(data_for_algos, data_to_predict_for_algos, 10, 10)

for i in n_estimators:
    for j in max_depth:
        model_score = rf.random_forest_classifier(data_for_algos, i, j)
        our_test_score = collections.Counter(model_score[0].predict(data_to_predict_for_algos) * test_classes).get(1)
        tuple_to_save = (i, j, model_score[1], our_test_score)
        print(tuple_to_save)
        combinations.append(tuple_to_save)
