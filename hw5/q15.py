import numpy as np 
from libsvm.svmutil import *

options = [0.1, 1, 10, 100, 1000]

for option in options:
    y, x = svm_read_problem('./data/satimage.scale')
    test_y, test_x = svm_read_problem('./data/satimage.scale.t')

    #change labels
    z = 0
    for i in y:
        if y[z] == 1.0:
            y[z] = 1.0
        else:
            y[z] = -1.0
        z += 1

    z = 0
    for i in test_y:
        if test_y[z] == 1.0:
            test_y[z] = 1.0
        else:
            test_y[z] = -1.0
        z += 1

    m = svm_train(y, x, '-s 0 -t 2 -g '+ str(option) +' -c 0.1 -q')
    p_labs, p_acc, pvals = svm_predict(test_y, test_x, m)

