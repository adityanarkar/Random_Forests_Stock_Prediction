import json
import os
from functools import reduce
from threading import Lock
import multiprocessing
import KNN.knn as knn
import ZeroR.zr as zr
import definitions
from RandomForest.parallel import rf
from SVM import svm
from data_prep import data_preparation as dp

future_day_start = 10
future_day_stop = 110
estimator_end = 60
depth = 60
initial_no_of_features = 10
max_features = 21
feature_window_size = 50
discretize = True



def selectTop(top, x):
    if x['score'] == -1:
        return top
    elif x["score"] > top["score"]:
        return x
    return top


def getInitial():
    initial = []
    for i in range(3):
        initial.append({"our_test_score": 0})
    return initial


def get_top_rf_result_csv_format(STOCK, top):
    top_estimator = top["estimators"]
    top_depth = top['max_depth']
    top_model_score = top['score']
    top_future_day = top['future_day']
    top_our_test_score = top['our_test_score']
    top_no_of_features = top['no_of_features']
    result = result_in_csv(STOCK, 'RF', top_estimator, top_depth, Model_Score=top_model_score,
                           Future_day=top_future_day, Our_test_score=top_our_test_score,
                           No_of_features=top_no_of_features)
    return result


def result_in_csv(STOCK, Algo, Estimator=-1, Depth=-1, Metric='0', No_of_features=0, Model_Score=0.0,
                  Future_day=0, C=-1, degree=-1, No_of_neighbors=-1,
                  Our_test_score=-1.0):
    return f"{STOCK},{Algo},{Estimator},{Depth},{Metric},{No_of_features},{Model_Score},{Future_day},{C},{degree},{No_of_neighbors},{Our_test_score}\n "


def testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes,
                      actual_data_to_predict, p):
    list_of_dicts = []
    n_estimators = range(10, estimator_end, 10)
    max_depth = range(10, depth, 10)
    for no_of_features in range(10, data_for_algos.shape[1], 1):
        print(f"Predicting for future days: {future_day} No of features: {no_of_features}")
        for i in n_estimators:
            for j in max_depth:
                list_of_dicts.append({'estimators': i, 'max_depth': j, 'no_of_features': no_of_features, 'data': data_for_algos})
    results = p.map(rf.random_forest_classifier, list_of_dicts)
    top = reduce(lambda top, dict_of_scores: selectTop(top, dict_of_scores), results, get_top_rf())
    result = get_final_result_with_pred_rf(STOCK, top, data_to_predict_for_algos,  test_classes, future_day)
    return result


def get_final_result_with_pred_rf(STOCK, top, data_to_predict_for_algos,  test_classes, future_day):
    if 'clf' not in top:
        return get_top_rf_result_csv_format(STOCK, get_top_rf(future_day=future_day))
    clf = top['clf']
    predictions = clf.predict(data_to_predict_for_algos)
    our_test_score = sum(
        [1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
    top.update({'our_test_score': our_test_score})
    top.update({"future_day": future_day})
    result = get_top_rf_result_csv_format(STOCK, top)
    print(f"final Result RF: {result}")
    return result


def get_top_rf(estimators=-1, max_depth=-1, future_day=-1, no_of_features=-1, score=-1, our_test_score=-1):
    return {"estimators": estimators, "max_depth": max_depth, "score": score,
            "future_day": future_day,
            "no_of_features": no_of_features,
            "our_test_score": our_test_score}


def create_dir_and_store_result(dir_to_create, result_path, result):
    if not os.path.isdir(dir_to_create):
        os.mkdir(dir_to_create)
        with open(result_path, 'w') as f:
            f.write(json.dumps(result))


def get_prepared_data(STOCK_FILE, window_size, feature_window_size, discretize):
    # window_size = no of days in future you want to predict
    # feature_window_size = the window size use to calculate the features

    df, actual_data_to_predict = dp.data_preparation(os.path.join(f"{definitions.ROOT_DIR}", f"data/{STOCK_FILE}"),
                                                     window_size=window_size,
                                                     feature_window_size=feature_window_size,
                                                     discretize=discretize).data_frame_with_features()
    complete_data = df.to_numpy()
    data_for_algos, data_to_predict_for_algos, test_classes = complete_data[:-window_size], complete_data[
                                                                                            -window_size:,
                                                                                            :-1], complete_data[
                                                                                                  -window_size:,
                                                                                                  -1]
    return data_for_algos, data_to_predict_for_algos, test_classes, actual_data_to_predict


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
                "C,degree,No_of_neighbors,Our_test_score\n")


def get_config_from_dict(dictionary):
    RESULT_FILE = dictionary["RESULT_FILE"]
    COMPLETED_FILE = dictionary["COMPLETED_FILE"]
    algos = dictionary["algos"]
    future_day_start = dictionary["future_day_start"]
    future_day_stop = dictionary["future_day_stop"]
    estimator_start = dictionary["estimator_start"]
    estimator_stop = dictionary["estimator_stop"]
    depth_start = dictionary["depth_start"]
    depth_stop = dictionary["depth_stop"]
    initial_no_of_features = dictionary["initial_no_of_features"]
    max_features = dictionary["max_features"]
    feature_window_size = dictionary["feature_window_size"]
    discretize = True if dictionary["discretize"] == 1 else False
    C = dictionary["C"]

    return RESULT_FILE, COMPLETED_FILE, algos, future_day_start, future_day_stop, estimator_start, \
           estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, \
           feature_window_size, discretize, C


def pre_run_tasks(RESULT_FILE, COMPLETED_FILE):
    make_missing_dirs(RESULT_FILE)
    make_missing_dirs(COMPLETED_FILE)
    add_headers(RESULT_FILE)


def run_tests_for_a_stock(filename, algos, RESULT_FILE, p):
    STOCK = filename.split(".csv")[0]
    result = ""
    for future_day in range(future_day_start, future_day_stop, 10):
        try:
            data_for_algos, data_to_predict_for_algos, test_classes, actual_data_to_predict = get_prepared_data(
                filename, future_day, feature_window_size, discretize)
        except:
            continue

        if 'RF' in algos:
            print('Starting RF testing')
            result += testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes,
                                        actual_data_to_predict, p)
            print(result)
            print('Finished RF testing')

        # if 'SVM' in algos:
        #     print('Starting SVM testing')
        #     result += testSVM(STOCK, data_for_algos, data_to_predict_for_algos, test_classes, future_day)
        #     print(result)
        #     print('Finished SVM testing')
        #
        # if 'KNN' in algos:
        #     print('Starting KNN testing')
        #     result += testKNN(STOCK, data_for_algos, data_to_predict_for_algos, test_classes, future_day)
        #     print(result)
        #     print('Finished KNN testing')
        #
        # if 'ZR' in algos:
        #     print('Starting ZR testing')
        #     result += testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes)
        #     print(result)
        #     print('Finished ZR testing')

    write_result_to_file(Lock(), RESULT_FILE, result)
    return filename


def main():
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    file = open('configs/config_all_fs.json')
    configs = json.load(file)

    for dictionary in configs:
        RESULT_FILE, COMPLETED_FILE, algos, future_day_start, future_day_stop, estimator_start, \
        estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, \
        feature_window_size, discretize, C = get_config_from_dict(dictionary)
        pre_run_tasks(RESULT_FILE, COMPLETED_FILE)

        files = list(map(lambda x: x.replace("\n", ""), open('10stocks.txt', 'r').readlines()))
        files.reverse()
        print(files)
        for filename in files:
            run_tests_for_a_stock(filename, algos, RESULT_FILE, p)


if __name__ == '__main__':
    main()
