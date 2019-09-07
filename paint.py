import os
import matplotlib
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
    # rects2 = ax.bar(x + width / 2, rfScore, width, label='Random Forests')
    ax.set_ylabel('Scores')
    ax.set_title(f"Scores for {stock}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    # autolabel(rects2)

    fig.tight_layout()
    plt.show()


def getZHResultsFromFile(filepath):
    results = []
    with open(filepath, 'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            score = data[i]["our_test_score"]
            results.append(score)
        return results


def getRFResultsFromFile(filepath):
    results = []
    return results


def prepareData(rfResultsDir, zhResultsDir, listOfStocks):
    labels = range(10, 110, 10)
    for tickr in listOfStocks:
        zhRes = getZHResultsFromFile(os.path.join(zhResultsDir, f"{tickr}.JSON"))
        # rfRes = getRFResultsFromFile(os.path.join(zhResultsDir, f"{tickr}.JSON"))
        drawOutput(tickr, 0, zhRes, labels)


stocks = []

with open("TICKR.txt", 'r') as f:
    for line in f.readlines():
        stocks.append(line.replace("\n",""))

prepareData("", "./Results/ZH", stocks)
