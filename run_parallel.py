import multiprocessing
from multiprocessing import Process, Lock
import main
import os


def f(l, i):
    l.acquire()
    print('hello world', i)
    l.release()


if __name__ == '__main__':
    max_cpus = multiprocessing.cpu_count()
    lock = Lock()

    RESULT_FILE = "Results/result-parallel.csv"
    COMPLETED_FILE = "Results/completed.qwe"
    main.add_headers(RESULT_FILE)
    completed = open(COMPLETED_FILE, 'r').readlines()
    print(completed)

    processes = []
    for filename in os.listdir("data/"):
        if not filename in completed:
            p = Process(target=main.runExperiment, args=(lock, filename, RESULT_FILE))
            p.start()
            processes.append({"process": p, "stock": filename})
            if len(processes) % 8 == 0:
                processes[0]["process"].join()
                open(COMPLETED_FILE, 'a').write(f"{processes[0]['stock']}\n")
                processes.remove(processes[0])
