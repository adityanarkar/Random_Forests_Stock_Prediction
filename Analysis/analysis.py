import pandas as pd
import matplotlib.pyplot as plt


def clean_dataframe(df: pd.DataFrame):
    return df[df['Our_test_score'] != 'error']


def get_mean_for_future_days(df: pd.DataFrame, future_days, algo):
    df = df[(df['Algorithm'] == algo)]
    scores = df[df['Future_day'] == future_days]
    scores['Our_test_score'] = scores['Our_test_score'].apply(lambda x: int(x))
    mean_score = (scores['Our_test_score'].mean() / future_days) * 100
    print(mean_score)
    return mean_score


def get_means_rf(df: pd.DataFrame):
    return [get_mean_for_future_days(df, future_days, 'RF') for future_days in range(10, 110, 10)]


def get_means_zr(df: pd.DataFrame):
    return [get_mean_for_future_days(df, future_days, 'ZR') for future_days in range(10, 110, 10)]


def plot_data():
    df = pd.read_csv('../Results/result-parallel.csv')
    df = clean_dataframe(df)
    x, y_rf, y_zr = gather_data(df)
    plt.plot(x, y_rf, label='RF', color='red')
    plt.plot(x, y_zr, label='ZR', color='blue')
    plt.axis([0, 100, 0, 100])
    plt.legend()
    plt.show()


def gather_data(df):
    y_rf = get_means_rf(df)
    y_zr = get_means_zr(df)
    x = [i for i in range(10, 110, 10)]

    return x, y_rf, y_zr


plot_data()
