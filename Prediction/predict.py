import pandas as pd
import main_fold as main
import RandomForest.RF_with_predictions as rf
import definitions
import os
import numpy as np
import collections

from KNN.knn import knn_classifier
from SVM.svm import svm_classifier
from ZeroR.zr import zr


def clean_dataframe(df: pd.DataFrame):
    df = df[df['Our_test_score'] != -1]
    df["Stock"] = df["Stock"].apply(lambda x: x.strip())
    return df[df['Our_test_score'] != 'error']


def get_dataframe(filepath):
    return pd.read_csv(filepath)


def filter_dataframe_with_pred_rate(df: pd.DataFrame, algo: str, pred_rate: float):
    df = df[(df['Algorithm'] == algo)]
    df["Our_test_score"] = df["Our_test_score"].apply(lambda x: int(x))
    df["Future_day"] = df["Future_day"].apply(lambda x: int(x))
    df["Model_Score"] = df["Model_Score"].apply(lambda x: float(x))
    df["Pred"] = df["Our_test_score"] / df["Future_day"]
    df = df[df["Pred"] > pred_rate]
    # df = df[df["Stock"].isin(["HUN"])]
    # df = df[~df["Stock"].isin(["ABR", "ASA", "ATO", "AFL", "AGN", "APH", "AVNS", "BBF", "BCE", "BGX", "BFO", "BIP", "BIG", "BQH", "BSMX", "BHP", "CII", "DTF", "CUZ", "CWEN", "NEA", "MNP", "HIO", "ESRT"])]
    return df
    # return df[df["Pred"] >= pred_rate]


def create_stock_and_future_day_list(filepath: str):
    df = clean_dataframe(get_dataframe(filepath))
    print(df.head())
    df = filter_dataframe_with_pred_rate(df, 'RF', 0.9)
    return df


def get_data(x: pd.Series, discretize: bool):
    STOCKFILE = f"{x['Stock'].strip()}.csv"
    future_day = int(x['Future_day'])
    return main.get_prepared_data(STOCKFILE, future_day, 50, discretize)


def predict(x: pd.Series):
    print(x)
    STOCKFILE = f"{x['Stock']}.csv"
    data_for_algos, actual_data_to_predict = get_data(x, True)
    if x['Algorithm'] == 'RF':
        estimators, depth, no_of_features, future_day, model_score = get_important_stuff(x)
        return predict_rf(data_for_algos, actual_data_to_predict, STOCKFILE, estimators, depth, no_of_features, future_day, model_score)
    elif x['Algorithm'] == 'SVM':
        kernel, C, degree, future_day, model_score, no_of_features = get_important_stuff(x)
        return predict_svm(data_for_algos, actual_data_to_predict, STOCKFILE, no_of_features, C, kernel, degree, future_day, model_score)
    elif x['Algorithm'] == 'KNN':
        metric, neighbors, no_of_features, future_day, model_score = get_important_stuff(x)
        return predict_knn(data_for_algos, actual_data_to_predict, STOCKFILE, metric, neighbors, no_of_features, future_day, model_score)
    elif x['Algorithm'] == 'ZR':
        future_day, model_score = get_important_stuff(x)
        return predict_zr(data_for_algos, actual_data_to_predict, STOCKFILE, future_day, model_score)


def predict_rf(data_for_algos, actual_data_to_predict, STOCKFILE, estimators, depth, no_of_features, future_day, model_score):
    print("Stock: ", STOCKFILE, "Estimators: ", estimators, "Depth: ", depth, "no_of_features: ", no_of_features)
    selector, score = rf.random_forest_classifier(data_for_algos, estimators, depth, no_of_features)
    actual_data = actual_data_to_predict.to_numpy()
    prediction = selector.predict(actual_data)
    print(prediction)
    return f"Stock:{STOCKFILE},Estimators:{estimators},Depth:{depth},no_of_features:{no_of_features},future_day:{future_day},model_score:{model_score},prediction:{prediction},Result:{collections.Counter(prediction).most_common(1)}"


def predict_svm(data_for_algos, actual_data_to_predict, STOCKFILE, no_of_features, C, kernel, degree, future_day, model_score):
    clf, score = svm_classifier(data_for_algos, no_of_features, C, kernel, degree)
    actual_data = actual_data_to_predict.to_numpy()
    prediction = clf.predict(actual_data)
    print(prediction)
    return f"Stock:{STOCKFILE},no_of_features:{no_of_features},future_day:{future_day},model_score:{model_score},C:{C},kernel:{kernel},degree:{degree},prediction:{prediction},Result:{collections.Counter(prediction).most_common(1)}"


def predict_knn(data_for_algos, actual_data_to_predict, STOCKFILE, metric, neighbors, no_of_features, future_day, model_score):
    clf, selector, score = knn_classifier(data_for_algos, metric, neighbors, no_of_features)
    actual_data = actual_data_to_predict.to_numpy()
    actual_data = selector.transform(actual_data)
    prediction = clf.predict(actual_data)
    print(prediction)
    return f"Stock:{STOCKFILE},no_of_features:{no_of_features},future_day:{future_day},model_score:{model_score},Distance_Function:{metric},No_of_neighbors:{neighbors},prediction:{prediction},Result:{collections.Counter(prediction).most_common(1)}"


def predict_zr(data_for_algos, actual_data_to_predict, STOCKFILE, future_day, model_score):
    clf, score = zr(data_for_algos)
    prediction = clf.predict(actual_data_to_predict)
    return f"Stock:{STOCKFILE},future_day:{future_day},model_score:{model_score},prediction:{prediction},Result:{collections.Counter(prediction).most_common(1)}"


def get_important_stuff(x):
    if x['Algorithm'] == 'RF':
        return get_important_rf_stuff(x)
    elif x['Algorithm'] == 'SVM':
        return get_important_svm_stuff(x)
    elif x['Algorithm'] == 'KNN':
        return get_important_knn_stuff(x)
    elif x['Algorithm'] == 'ZR':
        return get_important_zr_stuff(x)

def get_important_rf_stuff(x):
    estimators = x['Estimators']
    depth = x['Depth']
    no_of_features = x['No_of_features']
    future_day = x['Future_day']
    model_score = x['Model_Score']
    return estimators, depth, no_of_features, future_day, model_score


def get_important_svm_stuff(x):
    kernel = x['Distance_function']
    C = x['C']
    degree = x['degree']
    future_day = x['Future_day']
    model_score = x['Model_Score']
    no_of_features = x['No_of_features']
    return kernel, C, degree, future_day, model_score, no_of_features


def get_important_knn_stuff(x):
    metric = x['Distance_function']
    neighbors = x['No_of_neighbors']
    no_of_features = x['No_of_features']
    future_day = x['Future_day']
    model_score = x['Model_Score']
    return metric, neighbors, no_of_features, future_day, model_score


def get_important_zr_stuff(x):
    future_day = x['Future_day']
    model_score = x['Model_Score']
    return future_day, model_score


def write_predictions(pred):
    with open('../Predictions/Shuffle/FS/All_Predictions.txt', 'a') as file:
        file.write(pred+"\n")

def run(filepath):
    df = create_stock_and_future_day_list(filepath=filepath)
    print(df.head())
    res = df.apply(lambda x: predict(x), axis=1)
    res.apply(lambda x: write_predictions(x))


# run(os.path.join(definitions.ROOT_DIR, 'Results/Profit_Loss/Discretize/result_parallel_profit_loss.csv'))
run(os.path.join(definitions.ROOT_DIR, 'Results/EndGame/Shuffle/FS/final_result.csv'))
