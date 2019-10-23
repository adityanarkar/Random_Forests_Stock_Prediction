import json
import os
import ZeroR.zr as zr
import KNN.knn as knn
import data_prep.data_collection as dc
import definitions
from RandomForest import RF_with_predictions as rf
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


def get_top_rf_result_csv_format(STOCK, top):
    top_estimator = top["estimators"]
    top_depth = top['max_depth']
    top_model_score = top['model_score']
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
                      actual_data_to_predict):
    n_estimators = range(10, estimator_end, 10)
    max_depth = range(10, depth, 10)
    top = get_top_rf()
    for no_of_features in range(initial_no_of_features, data_for_algos.shape[1] + 1, 1):
        print(f"Predicting for future days: {future_day} No of features: {no_of_features}")
        for i in n_estimators:
            for j in max_depth:
                try:
                    model, score = rf.random_forest_classifier(data_for_algos, i, j, no_of_features)
                    predictions = model.predict(data_to_predict_for_algos)
                    our_test_score = sum(
                        [1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
                    our_test_score = 0 if our_test_score is None else our_test_score
                    if our_test_score > top['our_test_score']:
                        top = get_top_rf(estimators=i, max_depth=j, future_day=future_day,
                                         no_of_features=no_of_features, score=score, our_test_score=our_test_score)
                    elif our_test_score == top['our_test_score'] and score > top['model_score']:
                        top = get_top_rf(estimators=i, max_depth=j, future_day=future_day,
                                         no_of_features=no_of_features, score=score, our_test_score=our_test_score)
                except:
                    continue

    result = get_top_rf_result_csv_format(STOCK, top)

    print(f"final Result RF: {result}")
    return result


def get_top_rf(estimators=-1, max_depth=-1, future_day=-1, no_of_features=-1, score=-1, our_test_score=-1):
    return {"estimators": estimators, "max_depth": max_depth, "model_score": score,
            "future_day": future_day,
            "no_of_features": no_of_features,
            "our_test_score": our_test_score}


def testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos, test_classes):
    try:
        model, score = zr.zr(data_for_algos)
        predictions = model.predict(data_to_predict_for_algos)
        our_test_score = sum([1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
        our_test_score = 0 if our_test_score is None else our_test_score
        result = result_in_csv(STOCK, 'ZR', Future_day=future_day, Model_Score=score, Our_test_score=our_test_score)
    except:
        result = result_in_csv(STOCK, 'ZR', Future_day=future_day, Our_test_score=-1)
    print(result)
    return result


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


def testSVM(STOCK, data_for_algos, data_to_predict_for_algos, test_classes, future_day):
    top = get_top_svm(-1, -1, -1, future_day, -1, 'linear')
    for no_of_features in range(initial_no_of_features, data_for_algos.shape[1]):
        print(f"SVM {STOCK} {no_of_features}")
        for C in [0.5, 1, 5, 10, 100]:
            for kernel in ['linear', 'poly', 'rbf']:
                for degree in [1, 2, 3, 4]:
                    try:
                        clf, score = svm.svm_classifier(data_for_algos, no_of_features, C, kernel, degree)
                        predictions = clf.predict(data_to_predict_for_algos)
                        our_test_score = sum(
                            [1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
                        our_test_score = 0 if our_test_score is None else our_test_score
                        if (our_test_score > top['our_test_score']) or (
                                our_test_score == top['our_test_score'] and score > top['model_score']):
                            top = get_top_svm(C, no_of_features, score, future_day, our_test_score, kernel, degree)
                        if kernel != 'poly':
                            break
                    except:
                        continue
    result = get_svm_result_csv(STOCK, top)
    print(result)
    return result


def get_svm_result_csv(STOCK, top):
    return result_in_csv(STOCK=STOCK, Algo='SVM', No_of_features=top['no_of_features'], Model_Score=top['model_score'],
                         Future_day=top['future_day'], C=top['C'], Metric=top['metric'], degree=top['degree'],
                         Our_test_score=top['our_test_score'])


def get_top_svm(C, no_of_features, score, future_day, our_test_score, kernel, degree=-1):
    return {'C': C, 'no_of_features': no_of_features, "model_score": score, 'metric': kernel,
            "future_day": future_day,
            "degree": degree,
            "our_test_score": our_test_score}


def testKNN(STOCK, data_for_algos, data_to_predict_for_algos, test_classes, future_day):
    top = get_top_knn(-1, -1, -1, future_day, -1, -1)
    for no_of_features in range(initial_no_of_features, data_for_algos.shape[1]):
        for neighbors in [3, 5, 7, 9, 11]:
            for metric in ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']:
                try:
                    clf, selector, score = knn.knn_classifier(data_for_algos, metric, neighbors, data_for_algos.shape[
                                                                                                     1] - 1 if no_of_features == 'all' else no_of_features)
                    X_new = selector.transform(data_to_predict_for_algos)
                    predictions = clf.predict(X_new)
                    our_test_score = sum(
                        [1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
                    our_test_score = 0 if our_test_score is None else our_test_score
                    if (our_test_score > top['our_test_score']) or (
                            our_test_score == top['our_test_score'] and score > top['model_score']):
                        top = get_top_knn(data_for_algos.shape[
                                              1] - 1 if no_of_features == 'all' else no_of_features, neighbors, metric,
                                          future_day, score, our_test_score)
                except:
                    continue
    result = get_knn_result_csv(STOCK, top)
    print(result)
    return result


def get_knn_result_csv(STOCK, top):
    return result_in_csv(STOCK, 'KNN', Metric=top['metric'], No_of_features=top['no_of_features'],
                         Model_Score=top['model_score'], Future_day=top['future_day'], No_of_neighbors=top['neighbors'],
                         Our_test_score=top['our_test_score'])


def get_top_knn(no_of_features, neighbors, metric, future_day, score, our_test_score):
    return {'no_of_features': no_of_features, 'neighbors': neighbors, 'metric': metric, "model_score": score,
            "future_day": future_day,
            "our_test_score": our_test_score}


def runExperiment(lock, STOCK_FILE, RESULT_FILE):
    STOCK = STOCK_FILE.split(".csv")[0]
    print(STOCK)

    result = ""
    for future_day in range(future_day_start, future_day_stop, 10):
        try:
            data_for_algos, data_to_predict_for_algos, test_classes, actual_data_to_predict = get_prepared_data(
                STOCK_FILE, future_day, feature_window_size, discretize)
        except:
            continue

        # result += testRandomForests(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
        #                             test_classes, actual_data_to_predict)
        result += testSVM(STOCK=STOCK, data_for_algos=data_for_algos,
                          data_to_predict_for_algos=data_to_predict_for_algos, test_classes=test_classes,
                          future_day=future_day)
        #
        result += testKNN(STOCK, data_for_algos, data_to_predict_for_algos, test_classes, future_day)
        #
        # result += testZeroHour(STOCK, future_day, data_for_algos, data_to_predict_for_algos,
        #                        test_classes)

    write_result_to_file(lock, RESULT_FILE, result)


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

# tickr_file = "data/TICKR.txt"
# collect_data(300, tickr_file)
# print(sys.argv[1])
# runExperiment(sys.argv[1])
