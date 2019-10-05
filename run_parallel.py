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

    RESULT_FILE = "Results/Test/Shuffle/Discretize/result_parallel_up_down.csv"
    COMPLETED_FILE = "Results/Test/Shuffle/Discretize/completed_up_down.qwe"
    main.make_missing_dirs(RESULT_FILE)
    main.make_missing_dirs(COMPLETED_FILE)

    main.add_headers(RESULT_FILE)
    algos = ['RF']
    processes = []
    counter = 0
    files = os.listdir('data/')
    random.shuffle(files)
    for filename in files:
        if not filename in completed and counter < 100:
            counter += 1
            p = Process(target=main_updated.runExperiment, args=(lock, filename, RESULT_FILE, algos))
            p.start()
            processes.append({"process": p, "stock": filename})
            if len(processes) % 8 == 0:
                processes[0]["process"].join()
                open(COMPLETED_FILE, 'a').write(f"{processes[0]['stock']}\n")
                processes.remove(processes[0])
