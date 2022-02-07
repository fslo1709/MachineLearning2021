import numpy as np 
import math
import scipy.sparse as sp
from libsvm.svmutil import *

options = [2.0, 3.0, 4.0, 5.0, 6.0]

for option in options:
    y, x = svm_read_problem('./data/satimage.scale')
    z = 0
    for i in y:
        if y[z] == option:
            y[z] = 1.0
        else:
            y[z] = -1.0
        z += 1
    m = svm_train(y, x, '-s 0 -t 1 -d 3 -g 1 -r 1 -c 10 -q')
    sv_coef = m.get_sv_coef()
    print(np.shape(sv_coef))