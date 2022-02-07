import numpy as np 
import math
from libsvm.svmutil import *

options = [0.01, 0.1, 1, 10, 100]

for option in options:
    y, x = svm_read_problem('./data/satimage.scale')
    test_y, test_x = svm_read_problem('./data/satimage.scale.t')
    z = 0
    for i in y:
        if y[z] == 1.0:
            y[z] = 1.0
        else:
            y[z] = -1.0
        z += 1
    m = svm_train(y, x, '-s 0 -t 2 -g 10 -c '+ str(option) +' -q')
    p_labs, p_acc, pvals = svm_predict(test_y, test_x, m)