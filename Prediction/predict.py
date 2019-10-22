import pandas as pd
import main_fold as main
import RandomForest.RF_with_predictions as rf
import definitions
import os
import numpy as np
import collections


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
    estimators = x['Estimators']
    depth = x['Depth']
    no_of_features = x['No_of_features']
    print("Stock: ", STOCKFILE, "Estimators: ", estimators, "Depth: ", depth, "no_of_features: ", no_of_features)
    selector, score = rf.random_forest_classifier(data_for_algos, estimators, depth, no_of_features)
    actual_data = actual_data_to_predict.to_numpy()
    prediction = selector.predict(actual_data)
    print(prediction)
    return f"Stock:{STOCKFILE},Estimators:{estimators},Depth:{depth},no_of_features:{no_of_features},future_day:{x['Future_day']},model_score:{x['Model_Score']},prediction:{prediction},Result:{collections.Counter(prediction).most_common(1)}"

def write_predictions(pred):
    with open('../Predictions/RF_Shuffle_Enhanced.txt', 'a') as file:
        file.write(pred+"\n")

def run(filepath):
    df = create_stock_and_future_day_list(filepath=filepath)
    print(df.head())
    res = df.apply(lambda x: predict(x), axis=1)
    res.apply(lambda x: write_predictions(x))


# run(os.path.join(definitions.ROOT_DIR, 'Results/Profit_Loss/Discretize/result_parallel_profit_loss.csv'))
run(os.path.join(definitions.ROOT_DIR, 'Results/EndGame/Shuffle/FS/result.csv'))
