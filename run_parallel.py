import multiprocessing
import os, random
from multiprocessing import Process, Lock

import main
import main_updated


def f(l, i):
    l.acquire()
    print('hello world', i)
    l.release()


if __name__ == '__main__':
    max_cpus = multiprocessing.cpu_count()
    lock = Lock()
    completed = []

    RESULT_FILE = "Results/EndGame/Shuffle/FS/result_svm_knn_zr.csv"
    COMPLETED_FILE = "Results/EndGame/Shuffle/FS/completed_svm_knn_zr.qwe"
    main.make_missing_dirs(RESULT_FILE)
    main.make_missing_dirs(COMPLETED_FILE)

    main.add_headers(RESULT_FILE)
    processes = []
    files = list(map(lambda x: x.replace("\n", ""), open('10stocks.txt', 'r').readlines()))
    for filename in files:
        p = Process(target=main.runExperiment, args=(lock, filename, RESULT_FILE))
        p.start()
        p.join()
        processes.append({"process": p, "stock": filename})
        if len(processes) % 8 == 0:
            processes[0]["process"].join()
            open(COMPLETED_FILE, 'a').write(f"{processes[0]['stock']}\n")
            processes.remove(processes[0])
