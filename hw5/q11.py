import numpy as np 
import math
import scipy.sparse as sp
from libsvm.svmutil import *

y, x = svm_read_problem('./data/satimage.scale')
z = 0
for i in y:
    if y[z] == 5.0:
        y[z] = 1.0
    else:
        y[z] = -1.0
    z += 1

m = svm_train(y, x, '-s 0 -t 0 -c 10 -q')
sv_coef = m.get_sv_coef()
sv = m.get_SV()

new_sv = []
for element in sv:
    temp = [0.0]*36
    for i in element:
        temp[i-1] = element[i]
    new_sv.append(temp)

final_sv_coef = np.array(sv_coef)
final_sv = np.array(new_sv)
dot_product = np.dot(np.matrix.transpose(final_sv_coef), final_sv)

print(math.sqrt(np.inner(dot_product, dot_product)[0][0]))