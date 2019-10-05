import pandas as pd
import matplotlib.pyplot as plt
import main_updated

RESULT_FILE = "../Results/Test/Feature_Selection-10-15/Discretize/result_parallel_up_down.csv"
min_features = main_updated.initial_no_of_features
max_features = main_updated.max_features
future_day_start = main_updated.future_day_start
future_day_stop = main_updated.future_day_stop

def remove_errors(df: pd.DataFrame):
    df = df[df['Our_test_score'] != 'error']
    df = df[df['Our_test_score'] != -1]
    return df


def clean_dataframe(df: pd.DataFrame):
    df = df.copy()
    df = remove_errors(df)
    df['Our_test_score'] = df['Our_test_score'].apply(lambda x: int(x))
    df['Future_day'] = df['Future_day'].apply(lambda x: int(x))
    return df


def get_mean_for_future_days(df: pd.DataFrame, future_days, algo):
    df = df.copy()
    df = df[(df['Algorithm'] == algo)]
    scores = df[df['Future_day'] == future_days]
    # scores['Our_test_score'] = scores['Our_test_score'].apply(lambda x: int(x))
    mean_score = (scores['Our_test_score'].mean() / future_days) * 100
    print(mean_score)
    return mean_score


def get_mean_of_no_features_future_day(df: pd.DataFrame, no_of_features: int, future_day: int):
    df = df.copy()
    df = df[df['Future_day'] == future_day]
    df = df[df['No_of_features'] == no_of_features]
    return (df['Our_test_score'].mean() / future_day) * 100


def get_mean_for_feature_selection(df: pd.DataFrame):
    results = []
    for i in range(min_features, max_features, 1):
        result_for_feature = []
        for j in range(future_day_start, future_day_stop, 10):
            result_for_feature.append(get_mean_of_no_features_future_day(df, i, j))
        results.append({'no_of_features': i, 'result':result_for_feature})
    return results


def get_means_rf(df: pd.DataFrame):
    df = df.copy()
    return [get_mean_for_future_days(df, future_days, 'RF') for future_days in range(10, 110, 10)]


def get_means_zr(df: pd.DataFrame):
    df = df.copy()
    return [get_mean_for_future_days(df, future_days, 'ZR') for future_days in range(10, 110, 10)]


def get_means_knn(df: pd.DataFrame):
    df = df.copy()
    return [get_mean_for_future_days(df, future_days, 'KNN') for future_days in range(10, 110, 10)]


def plot_data():
    df = pd.read_csv("../Results/Profit_Loss/Discretize/result_parallel_profit_loss.csv")
    df = clean_dataframe(df)
    x, y_rf, y_zr, y_knn = gather_data(df)
    plt.plot(x, y_rf, label='RF', color='red')
    plt.plot(x, y_zr, label='ZR', color='blue')
    # plt.plot(x, y_knn, label='KNN', color='orange')
    plt.axis([0, 100, 0, 100])
    plt.legend()
    plt.show()


def gather_data(df: pd.DataFrame):
    y_rf = get_means_rf(df)
    y_zr = get_means_zr(df)
    # y_knn = get_means_knn(df)
    x = [i for i in range(10, 110, 10)]

    return x, y_rf, y_zr, 0


def plot_feature_selection():
    df = pd.read_csv(RESULT_FILE)
    df = clean_dataframe(df)
    x = [i for i in range(future_day_start, future_day_stop, 10)]
    results = get_mean_for_feature_selection(df)
    for result in results:
        plt.plot(x, result['result'], label=result['no_of_features'])
    plt.legend(loc='best')
    plt.show()

# plot_data()
plot_feature_selection()