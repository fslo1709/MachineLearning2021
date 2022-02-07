import numpy as np
import multiprocessing
import random
from libsvm.svmutil import *

def my_process():
    options = [0.1, 1, 10, 100, 1000]
    selected = [0]*5
    for i in range(125):
        y, x = svm_read_problem('./data/satimage.scale')
        z = list(zip(x, y))
        random.shuffle(z)
        x[:], y[:] = zip(*z)
        t = 0
        for i in y:
            if y[t] == 1.0:
                y[t] = 1.0
            else:
                y[t] = -1.0
            t += 1
        index = 0
        winner_index = 0
        winner_gamma = 0.0
        for option in options:
            m = svm_train(y[:200], x[:200], '-s 0 -t 2 -g '+ str(option) +' -c 0.1 -q')
            p_labs, p_acc, pvals = svm_predict(y[200:], x[200:], m)
            if (p_acc[0] > winner_gamma):
                winner_gamma = p_acc[0]
                winner_index = index
            index += 1
        selected[winner_index] += 1
    print(selected)

if __name__ == '__main__':
    p1 = multiprocessing.Process(target = my_process)
    p2 = multiprocessing.Process(target = my_process)
    p3 = multiprocessing.Process(target = my_process)
    p4 = multiprocessing.Process(target = my_process)
    p5 = multiprocessing.Process(target = my_process)
    p6 = multiprocessing.Process(target = my_process)
    p7 = multiprocessing.Process(target = my_process)
    p8 = multiprocessing.Process(target = my_process)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()

    print("Finished")
    