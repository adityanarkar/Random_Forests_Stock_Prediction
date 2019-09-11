import os
import matplotlib.pyplot as plt
import numpy as np
import json


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def drawOutput(stock, rfScore, zhScore, labels):
    x = np.arange(len(labels))
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, zhScore, width, label='Zero Hour')
    rects2 = ax.bar(x + width / 2, rfScore, width, label='Random Forests')
    ax.set_ylabel('Scores')
    ax.set_title(f"Scores for {stock}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)

    fig.tight_layout()
    plt.show()


def getZHResultsFromFile(filepath):
    # results = []
    with open(filepath, 'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            return data[i]["our_test_score"]
            # results.append(score)
    # return results


def getRFResultsFromFile(filepath):
    # results = []
    with open(filepath, 'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            if "model" in data[i].keys():
                return data[i]["model"]["our_test_score"]
            else:
                return 0
            # results.append(score)
    # return results


def prepareData(rfResultsDir, zhResultsDir, listOfStocks):
    labels = range(10, 110, 10)
    for tickr in listOfStocks:
        zhFilePath = os.path.join(zhResultsDir, f"{tickr}.JSON")
        rfFilePath = os.path.join(rfResultsDir, f"{tickr}.JSON")
        if os.path.isfile(zhFilePath) and os.path.isfile(rfFilePath):
            zhRes = getZHResultsFromFile(zhFilePath)
            rfRes = getRFResultsFromFile(rfFilePath)
            drawOutput(tickr, rfRes, zhRes, labels)
        else:
            print(f"File {zhFilePath} or {rfFilePath} does not exist.")


def prepare_data_all_stock_with_selection(rfResultsDir, zhResultsDir, listOfStocks):
    all_zh_results = []
    all_rf_results = []
    for tickr in listOfStocks:
        zhFilePath = os.path.join(zhResultsDir, f"{tickr}.JSON")
        rfFilePath = os.path.join(rfResultsDir, f"{tickr}.JSON")
        if os.path.isfile(zhFilePath) and os.path.isfile(rfFilePath):
            zhRes = getZHResultsFromFile(zhFilePath)
            all_zh_results.append(zhRes)
            rfRes = getRFResultsFromFile(rfFilePath)
            all_rf_results.append(rfRes)
        else:
            print(f"File {zhFilePath} or {rfFilePath} does not exist.")
        all_zh_results = [0 if x is None else x for x in all_zh_results]
    drawOutput("All stocks", all_rf_results, all_zh_results, listOfStocks)


# prepareData("./Results/RF", "./Results/ZH", main.get_requested_tickrs())

def get_requested_tickrs():
    result = []
    with open("TICKR.txt") as f:
        for line in f.readlines():
            if not line.startswith("#"):
                result.append(line.replace("\n", ""))
    return result


prepare_data_all_stock_with_selection("./Results/Selection/RF", "./Results/Selection/ZH", get_requested_tickrs())
